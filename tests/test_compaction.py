"""Tests for session event compaction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.events.filters import apply_compaction
from orxhestra.models.part import Content, ToolCallPart, ToolResponsePart
from orxhestra.sessions.compaction import (
    CompactionConfig,
    _events_to_text,
    _find_compaction_boundary,
    compact_session,
)
from orxhestra.sessions.in_memory_session_service import InMemorySessionService
from orxhestra.sessions.session import Session


def _make_event(
    text: str = "hello",
    *,
    event_type: EventType = EventType.AGENT_MESSAGE,
    agent_name: str = "bot",
    session_id: str = "s1",
    timestamp: float | None = None,
    content: Content | None = None,
    actions: EventActions | None = None,
) -> Event:
    """Create a simple event for testing."""
    return Event(
        type=event_type,
        agent_name=agent_name,
        session_id=session_id,
        content=content or Content.from_text(text),
        actions=actions or EventActions(),
        **({"timestamp": timestamp} if timestamp is not None else {}),
    )


def _make_session(event_count: int, session_id: str = "s1") -> Session:
    """Create a session with N text events."""
    events = [
        _make_event(f"msg-{i}", timestamp=float(i))
        for i in range(event_count)
    ]
    return Session(id=session_id, app_name="app", user_id="u1", events=events)


def _find_compaction_event(session: Session) -> Event:
    """Return the compaction event from the session."""
    for e in session.events:
        if e.actions.compaction is not None:
            return e
    raise AssertionError("No compaction event found in session")


# --- CompactionConfig ---


def test_compaction_config_defaults() -> None:
    config = CompactionConfig()
    assert config.max_events == 50
    assert config.retention_count == 20
    assert config.llm is None


def test_compaction_config_custom_values() -> None:
    config = CompactionConfig(max_events=100, retention_count=30)
    assert config.max_events == 100
    assert config.retention_count == 30


# --- _events_to_text ---


def test_events_to_text_with_text_events() -> None:
    events = [_make_event("hello"), _make_event("world")]
    result = _events_to_text(events)
    assert "[agent_message] (bot): hello" in result
    assert "[agent_message] (bot): world" in result


def test_events_to_text_with_tool_call() -> None:
    tc = ToolCallPart(tool_call_id="tc1", tool_name="search", args={"q": "test"})
    event = _make_event(content=Content(parts=[tc]))
    result = _events_to_text([event])
    assert "tool_call: search" in result


def test_events_to_text_with_tool_response() -> None:
    tr = ToolResponsePart(tool_call_id="tc1", tool_name="search", result="found it")
    event = _make_event(
        content=Content(parts=[tr]),
        event_type=EventType.TOOL_RESPONSE,
    )
    result = _events_to_text([event])
    assert "tool_response: search" in result
    assert "found it" in result


def test_events_to_text_truncates_long_text() -> None:
    long_text = "x" * 1000
    events = [_make_event(long_text)]
    result = _events_to_text(events)
    assert len(result) < 1000


def test_events_to_text_empty_list() -> None:
    assert _events_to_text([]) == ""


def test_events_to_text_no_agent_name() -> None:
    event = _make_event("hi", agent_name=None)
    result = _events_to_text([event])
    assert "[agent_message]:" in result
    assert "(None)" not in result


# --- _find_compaction_boundary ---


def test_find_compaction_boundary_no_compactions() -> None:
    events = [_make_event("a"), _make_event("b")]
    assert _find_compaction_boundary(events) == -1.0


def test_find_compaction_boundary_with_compaction() -> None:
    from orxhestra.events.event_actions import EventCompaction

    compaction_event = _make_event(
        "summary",
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=0.0,
                end_timestamp=39.0,
                summary="summary",
                event_count=40,
            ),
        ),
    )
    events = [compaction_event, _make_event("after")]
    assert _find_compaction_boundary(events) == 39.0


# --- compact_session (non-destructive) ---


@pytest.mark.asyncio
async def test_no_compaction_under_threshold() -> None:
    session = _make_session(10)
    service = InMemorySessionService()
    config = CompactionConfig(max_events=50, retention_count=20)

    result = await compact_session(session, service, config)

    assert result is False
    assert len(session.events) == 10


@pytest.mark.asyncio
async def test_no_compaction_at_exact_threshold() -> None:
    session = _make_session(50)
    config = CompactionConfig(max_events=50, retention_count=20)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False
    assert len(session.events) == 50


@pytest.mark.asyncio
async def test_compaction_above_threshold_no_llm() -> None:
    session = _make_session(60)
    original_count = len(session.events)
    config = CompactionConfig(max_events=50, retention_count=20)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is True
    # Non-destructive: all original events + 1 appended compaction event
    assert len(session.events) == original_count + 1
    compaction_event = _find_compaction_event(session)
    assert compaction_event.agent_name == "compaction"
    assert compaction_event.actions.compaction.event_count == 40


@pytest.mark.asyncio
async def test_compaction_preserves_all_original_events() -> None:
    session = _make_session(60)
    original_ids = [e.id for e in session.events]
    config = CompactionConfig(max_events=50, retention_count=20)

    await compact_session(session, InMemorySessionService(), config)

    # All original events are still present (non-destructive)
    current_ids = [e.id for e in session.events if e.actions.compaction is None]
    assert current_ids == original_ids


@pytest.mark.asyncio
async def test_compaction_event_appended_at_end() -> None:
    session = _make_session(60)
    config = CompactionConfig(max_events=50, retention_count=20)

    await compact_session(session, InMemorySessionService(), config)

    assert session.events[-1].actions.compaction is not None


@pytest.mark.asyncio
async def test_compaction_event_has_correct_timestamps() -> None:
    session = _make_session(60)
    config = CompactionConfig(max_events=50, retention_count=20)

    await compact_session(session, InMemorySessionService(), config)

    compaction = _find_compaction_event(session).actions.compaction
    assert compaction.start_timestamp == 0.0
    assert compaction.end_timestamp == 39.0


@pytest.mark.asyncio
async def test_compaction_event_summary_matches_content() -> None:
    session = _make_session(60)
    config = CompactionConfig(max_events=50, retention_count=20)

    await compact_session(session, InMemorySessionService(), config)

    compaction_event = _find_compaction_event(session)
    assert compaction_event.text == compaction_event.actions.compaction.summary


@pytest.mark.asyncio
async def test_compaction_with_llm() -> None:
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "This is a summary of the conversation."
    mock_llm.ainvoke.return_value = mock_response

    session = _make_session(60)
    config = CompactionConfig(max_events=50, retention_count=20, llm=mock_llm)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is True
    mock_llm.ainvoke.assert_called_once()
    compaction_event = _find_compaction_event(session)
    assert compaction_event.text == "This is a summary of the conversation."
    assert compaction_event.actions.compaction.summary == "This is a summary of the conversation."


@pytest.mark.asyncio
async def test_compaction_llm_failure_returns_false() -> None:
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = RuntimeError("LLM unavailable")

    session = _make_session(60)
    original_count = len(session.events)
    config = CompactionConfig(max_events=50, retention_count=20, llm=mock_llm)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False
    assert len(session.events) == original_count


@pytest.mark.asyncio
async def test_no_compaction_when_split_idx_zero() -> None:
    """retention_count >= event count means nothing to compact."""
    session = _make_session(55)
    config = CompactionConfig(max_events=50, retention_count=55)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False


@pytest.mark.asyncio
async def test_no_compaction_with_pending_tool_calls() -> None:
    """Events with unresolved tool calls should not be compacted."""
    events = [
        _make_event(f"msg-{i}", timestamp=float(i))
        for i in range(40)
    ]
    # Add a tool call in old events (idx 10) with no matching response
    tc = ToolCallPart(tool_call_id="pending-tc", tool_name="search", args={})
    events[10] = _make_event(
        content=Content(parts=[tc]),
        timestamp=10.0,
    )
    # Add retained events
    events.extend(
        _make_event(f"recent-{i}", timestamp=float(40 + i))
        for i in range(20)
    )
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    config = CompactionConfig(max_events=50, retention_count=20)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is False
    assert len(session.events) == 60


@pytest.mark.asyncio
async def test_compaction_allows_resolved_tool_calls() -> None:
    """Tool calls with matching responses should be compactable."""
    events = [
        _make_event(f"msg-{i}", timestamp=float(i))
        for i in range(38)
    ]
    # Add tool call in old events
    tc = ToolCallPart(tool_call_id="resolved-tc", tool_name="search", args={})
    events.append(_make_event(content=Content(parts=[tc]), timestamp=38.0))
    # Add matching response
    tr = ToolResponsePart(tool_call_id="resolved-tc", tool_name="search", result="ok")
    events.append(
        _make_event(
            content=Content(parts=[tr]),
            event_type=EventType.TOOL_RESPONSE,
            timestamp=39.0,
        )
    )
    # Add retained events
    events.extend(
        _make_event(f"recent-{i}", timestamp=float(40 + i))
        for i in range(20)
    )
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    config = CompactionConfig(max_events=50, retention_count=20)

    result = await compact_session(session, InMemorySessionService(), config)

    assert result is True
    # Non-destructive: 60 originals + 1 compaction = 61
    assert len(session.events) == 61


@pytest.mark.asyncio
async def test_compaction_fallback_truncates_at_2000_chars() -> None:
    """Without an LLM, fallback summary is truncated to 2000 chars."""
    events = [
        _make_event("x" * 200, timestamp=float(i))
        for i in range(80)
    ]
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    config = CompactionConfig(max_events=50, retention_count=20)

    await compact_session(session, InMemorySessionService(), config)

    summary = _find_compaction_event(session).actions.compaction.summary
    assert len(summary) <= 2000


@pytest.mark.asyncio
async def test_repeated_compaction_is_idempotent() -> None:
    """Second compaction should be skipped — non-compacted events are under threshold."""
    session = _make_session(60)
    service = InMemorySessionService()
    config = CompactionConfig(max_events=50, retention_count=20)

    first = await compact_session(session, service, config)
    assert first is True
    count_after_first = len(session.events)

    second = await compact_session(session, service, config)
    assert second is False
    assert len(session.events) == count_after_first


# --- apply_compaction integration ---


@pytest.mark.asyncio
async def test_apply_compaction_filters_compacted_events() -> None:
    """Verify the view layer hides events covered by the compaction range."""
    session = _make_session(60)
    config = CompactionConfig(max_events=50, retention_count=20)

    await compact_session(session, InMemorySessionService(), config)

    # apply_compaction should filter out the 40 old events
    filtered = apply_compaction(session.events)

    # 20 retained events + 1 compaction summary = 21
    assert len(filtered) == 21

    # The compaction summary should be present
    summary_events = [e for e in filtered if "[Compacted summary" in e.text]
    assert len(summary_events) == 1
    assert "40 events" in summary_events[0].text

    # None of the old events (timestamps 0-39) should be in filtered
    for e in filtered:
        if "[Compacted summary" not in e.text:
            assert e.timestamp >= 40.0


@pytest.mark.asyncio
async def test_apply_compaction_after_multiple_compactions() -> None:
    """Multiple compaction rounds should produce correct filtered view."""
    # Create 120 events (timestamps 0..119)
    events = [
        _make_event(f"msg-{i}", timestamp=float(i))
        for i in range(120)
    ]
    session = Session(id="s1", app_name="app", user_id="u1", events=events)
    service = InMemorySessionService()
    # Compact at 50, retain 20 — first round compacts 100 old events
    config = CompactionConfig(max_events=50, retention_count=20)

    first = await compact_session(session, service, config)
    assert first is True

    # Simulate more events arriving (timestamps 120..179)
    for i in range(120, 180):
        session.events.append(_make_event(f"msg-{i}", timestamp=float(i)))

    second = await compact_session(session, service, config)
    assert second is True

    # View layer should produce a manageable set
    filtered = apply_compaction(session.events)
    # Should have 2 compaction summaries + 20 retained events
    summaries = [e for e in filtered if "[Compacted summary" in e.text]
    assert len(summaries) == 2
