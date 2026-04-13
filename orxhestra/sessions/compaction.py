"""Session event compaction — LLM-based summarization of old events.

When session history grows beyond a configured character threshold, old
events are summarized into a compaction event appended to the session.
The original events are preserved; the view layer (``apply_compaction``)
filters them out when building LLM context.

Approach (non-destructive, token-aware):
  1. Estimate total character count of non-compacted events.
     If under ``char_threshold``, do nothing.
  2. Identify events not already covered by a prior compaction.
  3. Summarize the older portion using the LLM, keeping the most
     recent ``retention_chars`` worth of events raw.
  4. Append a compaction event via the session service.

Example::

    config = CompactionConfig(char_threshold=100_000, retention_chars=20_000)
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


def _estimate_event_chars(event: Event) -> int:
    """Estimate the character count of an event's content."""
    total: int = 0
    if event.text:
        total += len(event.text)
    if event.has_tool_calls:
        for tc in event.tool_calls:
            total += len(tc.tool_name) + len(str(tc.args))
    if event.type == EventType.TOOL_RESPONSE:
        for tr in event.content.tool_responses:
            if tr.result:
                total += len(tr.result)
    return total


@dataclass
class CompactionConfig:
    """Configuration for automatic session compaction.

    Parameters
    ----------
    char_threshold : int
        Compact when non-compacted event content exceeds this many
        characters.  Default 100,000 (~25k tokens).
    retention_chars : int
        Always keep the most recent events totalling at least this
        many characters as raw (uncompacted).  Default 20,000
        (~5k tokens).
    model : BaseChatModel, optional
        LLM to use for summarization.  If ``None``, the compactor
        will use a simple text-based extraction instead of an LLM call.
    """

    char_threshold: int = 100_000
    retention_chars: int = 20_000
    model: BaseChatModel | None = field(default=None, repr=False)


_SUMMARIZE_PROMPT = """\
You are a conversation summarizer. Create a structured summary of these \
conversation events. This summary will replace the original events in the \
agent's context, so it MUST preserve all actionable information.

## Required sections (include ALL that apply):

### Primary Request
What the user originally asked for. Include exact requirements.

### Key Technical Context
Languages, frameworks, patterns, architecture decisions discovered.

### Files and Code
List ALL files read, created, or modified with their paths. For modified \
files, note WHAT was changed and WHY.

### Decisions Made
Any choices, trade-offs, or approaches that were selected and why.

### Current State
What has been completed so far, what is in progress, what remains.

### Pending Tasks
If there are outstanding items, list them clearly.

## Rules
- Preserve ALL file paths, function names, error messages, and command outputs.
- Include specific code snippets only if they are essential context.
- Do NOT include conversational filler or repeated information.
- Keep the summary under 1000 words but be thorough.

Events:
{events_text}

Structured summary:"""


def _events_to_text(events: list[Event]) -> str:
    """Convert events to a readable text block for the summarizer."""
    lines: list[str] = []
    for event in events:
        prefix = f"[{event.type.value}]"
        if event.agent_name:
            prefix += f" ({event.agent_name})"

        if event.text:
            lines.append(f"{prefix}: {event.text[:2000]}")
        elif event.has_tool_calls:
            for tc in event.tool_calls:
                args_str = str(tc.args)[:500]
                lines.append(f"{prefix} tool_call: {tc.tool_name}({args_str})")
        elif event.type == EventType.TOOL_RESPONSE:
            for tr in event.content.tool_responses:
                result = tr.result[:1000] if tr.result else ""
                lines.append(f"{prefix} tool_response: {tr.tool_name} → {result}")

    return "\n".join(lines)


def _find_compaction_boundary(events: list[Event]) -> float:
    """Return the end timestamp of the latest existing compaction, or -1."""
    boundary: float = -1.0
    for e in events:
        if e.actions.compaction is not None:
            boundary = max(boundary, e.actions.compaction.end_timestamp)
    return boundary


def _split_by_retention_chars(
    events: list[Event], retention_chars: int,
) -> tuple[list[Event], list[Event]]:
    """Split events into (old, recent) keeping at least retention_chars recent.

    The split point is adjusted so that tool_call/tool_response pairs are
    never separated — if the candidate boundary falls between a tool call
    and its response, the boundary moves earlier to include the full pair.
    """
    cumulative: int = 0
    split_idx: int = len(events)
    for i in range(len(events) - 1, -1, -1):
        cumulative += _estimate_event_chars(events[i])
        if cumulative >= retention_chars:
            split_idx = i + 1
            break
    else:
        # All events fit within retention — nothing to compact
        split_idx = 0

    # Ensure tool_call/tool_response pairs are not split.
    # Collect tool_call_ids from events in the "old" portion and
    # check if any responses are in the "recent" portion.
    old_call_ids: set[str] = set()
    for event in events[:split_idx]:
        if event.has_tool_calls:
            for tc in event.tool_calls:
                if tc.tool_call_id:
                    old_call_ids.add(tc.tool_call_id)

    # Walk the recent portion backwards to find orphaned responses.
    # Move the split point earlier to include any tool calls whose
    # responses landed in the recent portion.
    while split_idx > 0:
        orphaned = False
        for event in events[split_idx:]:
            if event.type == EventType.TOOL_RESPONSE:
                for tr in event.content.tool_responses:
                    if tr.tool_call_id and tr.tool_call_id in old_call_ids:
                        orphaned = True
                        break
            if orphaned:
                break
        if not orphaned:
            break
        # Move split point earlier by one event and recompute.
        split_idx -= 1
        # Recompute old_call_ids.
        old_call_ids.clear()
        for event in events[:split_idx]:
            if event.has_tool_calls:
                for tc in event.tool_calls:
                    if tc.tool_call_id:
                        old_call_ids.add(tc.tool_call_id)

    return events[:split_idx], events[split_idx:]


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

    # Check character threshold
    total_chars = sum(_estimate_event_chars(e) for e in candidate_events)
    if total_chars <= config.char_threshold:
        return False

    # Split: old candidates to compact vs. recent ones to keep raw
    old_events, _recent = _split_by_retention_chars(
        candidate_events, config.retention_chars,
    )
    if not old_events:
        return False

    # Skip if there are pending tool calls in the old events
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

    if config.model is not None:
        prompt = _SUMMARIZE_PROMPT.format(events_text=events_text)
        try:
            response = await config.model.ainvoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.debug("Compaction LLM call failed: %s", exc)
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

    logger.debug(
        "Compacted %d events (%d chars) into summary (%d chars). "
        "Session has %d total events.",
        len(old_events),
        total_chars - sum(_estimate_event_chars(e) for e in _recent),
        len(summary),
        len(session.events),
    )

    return True
