"""Session event compaction — LLM-based summarization of old events.

When session history grows beyond a configured threshold, old events
are summarized into a compaction event appended to the session. The
original events are preserved; the view layer (``apply_compaction``)
filters them out when building LLM context.

Approach (non-destructive, sliding window):
  1. Count non-compacted events.  If under ``max_events``, do nothing.
  2. Identify events not already covered by a prior compaction.
  3. Summarize the older portion using the LLM.
  4. Append a compaction event via the session service.  The
     ``apply_compaction`` filter in ``events.filters`` replaces raw
     events in the compacted range with the summary at query time.

Example::

    config = CompactionConfig(max_events=50, retention_count=20)
    runner = Runner(agent=my_agent, ..., compaction_config=config)

The runner will automatically compact after each invocation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions, EventCompaction
from orxhestra.models.part import Content

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orxhestra.sessions.base_session_service import BaseSessionService
    from orxhestra.sessions.session import Session

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """Configuration for automatic session compaction.

    Attributes
    ----------
    max_events : int
        Compact when non-compacted events exceed this count.
    retention_count : int
        Always keep the last N events as raw (uncompacted).
    llm : BaseChatModel, optional
        LLM to use for summarization.  If ``None``, the compactor
        will use a simple text-based extraction instead of an LLM call.
    """

    max_events: int = 50
    retention_count: int = 20
    llm: BaseChatModel | None = field(default=None, repr=False)


_SUMMARIZE_PROMPT = """\
Summarize the following conversation events into a concise prose summary.
Preserve key facts, decisions, tool results, and any important data.
Remove redundant or irrelevant information.
Keep the summary under 500 words.

Events:
{events_text}

Summary:"""


def _events_to_text(events: list[Event]) -> str:
    """Convert events to a readable text block for the summarizer."""
    lines: list[str] = []
    for event in events:
        prefix = f"[{event.type.value}]"
        if event.agent_name:
            prefix += f" ({event.agent_name})"

        if event.text:
            lines.append(f"{prefix}: {event.text[:500]}")
        elif event.has_tool_calls:
            for tc in event.tool_calls:
                lines.append(f"{prefix} tool_call: {tc.tool_name}({tc.args})")
        elif event.type == EventType.TOOL_RESPONSE:
            for tr in event.content.tool_responses:
                result = tr.result[:200] if tr.result else ""
                lines.append(f"{prefix} tool_response: {tr.tool_name} → {result}")

    return "\n".join(lines)


def _find_compaction_boundary(events: list[Event]) -> float:
    """Return the end timestamp of the latest existing compaction, or -1."""
    boundary: float = -1.0
    for e in events:
        if e.actions.compaction is not None:
            boundary = max(boundary, e.actions.compaction.end_timestamp)
    return boundary


async def compact_session(
    session: Session,
    session_service: BaseSessionService,
    config: CompactionConfig,
) -> bool:
    """Compact old session events if the session exceeds the threshold.

    Compaction is non-destructive: a compaction event is appended to the
    session via the session service. Original events are preserved.  The
    ``apply_compaction`` filter hides them when building LLM context.

    Parameters
    ----------
    session : Session
        The session to potentially compact.
    session_service : BaseSessionService
        Service to persist the compaction event.
    config : CompactionConfig
        Compaction configuration.

    Returns
    -------
    bool
        True if compaction was performed, False otherwise.
    """
    events = session.events

    # Find the boundary of the last compaction so we only consider
    # events that haven't already been compacted.
    compaction_boundary = _find_compaction_boundary(events)

    candidate_events = [
        e for e in events
        if e.timestamp > compaction_boundary and e.actions.compaction is None
    ]

    if len(candidate_events) <= config.max_events:
        return False

    # Split: old candidates to compact vs. recent ones to keep raw
    split_idx = len(candidate_events) - config.retention_count
    if split_idx <= 0:
        return False

    old_events = candidate_events[:split_idx]
    if not old_events:
        return False

    # Skip if there are pending tool calls in the old events
    # (never compact events with unresolved tool calls)
    responded_ids: set[str] = set()
    for e in events:
        if e.type == EventType.TOOL_RESPONSE:
            for tr in e.content.tool_responses:
                if tr.tool_call_id:
                    responded_ids.add(tr.tool_call_id)

    for e in old_events:
        if e.has_tool_calls:
            for tc in e.tool_calls:
                if tc.tool_call_id and tc.tool_call_id not in responded_ids:
                    logger.debug("Skipping compaction: pending tool call %s", tc.tool_call_id)
                    return False

    # Generate summary
    events_text = _events_to_text(old_events)
    if not events_text.strip():
        return False

    if config.llm is not None:
        prompt = _SUMMARIZE_PROMPT.format(events_text=events_text)
        try:
            response = await config.llm.ainvoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning("Compaction LLM call failed: %s", exc)
            return False
    else:
        summary = events_text[:2000]

    # Create and append compaction event (non-destructive)
    start_ts = old_events[0].timestamp
    end_ts = old_events[-1].timestamp

    compaction_event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id=session.id,
        agent_name="compaction",
        content=Content.from_text(summary),
        actions=EventActions(
            compaction=EventCompaction(
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                summary=summary,
                event_count=len(old_events),
            ),
        ),
    )

    await session_service.append_event(session, compaction_event)

    logger.info(
        "Compacted %d events into summary (%d chars). "
        "Session has %d total events.",
        len(old_events),
        len(summary),
        len(session.events),
    )

    return True
