"""Tests for DatabaseSessionService (SQLite backend)."""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from orxhestra.events.event import Event, EventType  # noqa: E402
from orxhestra.events.event_actions import EventActions  # noqa: E402
from orxhestra.models.part import Content  # noqa: E402
from orxhestra.sessions.database_session_service import DatabaseSessionService  # noqa: E402


@pytest.fixture
async def svc(tmp_path):
    """Create a fresh SQLite-backed session service."""
    db_path = tmp_path / "test.db"
    service = DatabaseSessionService(f"sqlite+aiosqlite:///{db_path}")
    await service.initialize()
    return service


@pytest.mark.asyncio
async def test_create_and_get_session(svc):
    """Create a session, then retrieve it."""
    session = await svc.create_session(
        app_name="app", user_id="u1", session_id="s1",
    )
    assert session.id == "s1"
    assert session.app_name == "app"

    loaded = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert loaded is not None
    assert loaded.id == "s1"


@pytest.mark.asyncio
async def test_get_session_not_found(svc):
    """Non-existent session returns None."""
    result = await svc.get_session(app_name="app", user_id="u1", session_id="nope")
    assert result is None


@pytest.mark.asyncio
async def test_create_session_with_state(svc):
    """Initial state should be persisted and reloaded."""
    await svc.create_session(
        app_name="app", user_id="u1", session_id="s1",
        state={"counter": 42},
    )
    loaded = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert loaded.state["counter"] == 42


@pytest.mark.asyncio
async def test_append_event_persists(svc):
    """Events should be persisted and reloaded in order."""
    session = await svc.create_session(app_name="app", user_id="u1", session_id="s1")

    event = Event(
        type=EventType.USER_MESSAGE,
        session_id="s1",
        author="user",
        content=Content.from_text("Hello"),
    )
    await svc.append_event(session, event)

    loaded = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert len(loaded.events) == 1
    assert loaded.events[0].text == "Hello"


@pytest.mark.asyncio
async def test_append_event_applies_state_delta(svc):
    """State delta from events should be persisted."""
    session = await svc.create_session(app_name="app", user_id="u1", session_id="s1")

    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id="s1",
        agent_name="bot",
        content=Content.from_text("Done"),
        actions=EventActions(state_delta={"result": "ok"}),
    )
    await svc.append_event(session, event)

    loaded = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert loaded.state["result"] == "ok"


@pytest.mark.asyncio
async def test_delete_session(svc):
    """Deleted sessions should not be retrievable."""
    await svc.create_session(app_name="app", user_id="u1", session_id="s1")
    await svc.delete_session("s1")

    result = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert result is None


@pytest.mark.asyncio
async def test_list_sessions(svc):
    """List should return all sessions for a user."""
    await svc.create_session(app_name="app", user_id="u1", session_id="s1")
    await svc.create_session(app_name="app", user_id="u1", session_id="s2")
    await svc.create_session(app_name="app", user_id="u2", session_id="s3")

    sessions = await svc.list_sessions(app_name="app", user_id="u1")
    assert len(sessions) == 2
    ids = {s.id for s in sessions}
    assert ids == {"s1", "s2"}


@pytest.mark.asyncio
async def test_update_session_state(svc):
    """update_session should merge state."""
    await svc.create_session(
        app_name="app", user_id="u1", session_id="s1",
        state={"a": 1},
    )
    updated = await svc.update_session("s1", state={"b": 2})
    assert updated.state == {"a": 1, "b": 2}

    loaded = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert loaded.state == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_multiple_events_order(svc):
    """Multiple events should be returned in insertion order."""
    session = await svc.create_session(app_name="app", user_id="u1", session_id="s1")

    for i in range(5):
        event = Event(
            type=EventType.USER_MESSAGE,
            session_id="s1",
            author="user",
            content=Content.from_text(f"msg-{i}"),
        )
        await svc.append_event(session, event)

    loaded = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert len(loaded.events) == 5
    for i, event in enumerate(loaded.events):
        assert event.text == f"msg-{i}"


@pytest.mark.asyncio
async def test_not_initialized_raises():
    """Using service before initialize() should raise RuntimeError."""
    svc = DatabaseSessionService("sqlite+aiosqlite:///test.db")
    with pytest.raises(RuntimeError, match="initialize"):
        await svc.create_session(app_name="app", user_id="u1")
