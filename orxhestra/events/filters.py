"""Event filtering and compaction utilities.

Pure functions for preparing session events before they are converted
to LLM messages. Used by ``LlmAgent`` and available for custom agents.
"""

from __future__ import annotations

from orxhestra.events.event import Event, EventType
from orxhestra.models.part import Content


def should_include_event(event: Event) -> bool:
    """Return True if the event should be included in LLM context.

    Filters out events that carry no useful information for the LLM:
    partial (streaming) events, empty events, framework lifecycle
    events (AGENT_START/END), error-only metadata, and scratchpad notes.
    """
    if event.partial:
        return False

    if event.type in (EventType.AGENT_START, EventType.AGENT_END):
        return False

    if event.metadata.get("error"):
        return False

    if event.metadata.get("scratchpad"):
        return False

    if event.type == EventType.AGENT_MESSAGE:
        if not event.text and not event.data and not event.has_tool_calls:
            return False

    return True


def apply_compaction(events: list[Event]) -> list[Event]:
    """Replace raw events with compaction summaries where applicable.

    When events carry an ``EventActions.compaction`` entry, all raw
    events within that timestamp range are replaced by a single
    synthetic event containing the compaction summary.
    """
    compactions: list[tuple[float, float, str]] = []
    for event in events:
        c = event.actions.compaction
        if c is not None:
            compactions.append((c.start_timestamp, c.end_timestamp, c.summary))

    if not compactions:
        return events

    result: list[Event] = []
    for event in events:
        c = event.actions.compaction
        if c is not None:
            # Emit as USER_MESSAGE so the LLM sees it as context
            # injection, not as a prior AI response it never generated.
            result.append(Event(
                type=EventType.USER_MESSAGE,
                session_id=event.session_id,
                agent_name=event.agent_name,
                branch=event.branch,
                invocation_id=event.invocation_id,
                content=Content.from_text(
                    f"[Compacted summary of {c.event_count} previous events]\n\n"
                    f"{c.summary}"
                ),
            ))
            continue

        ts = event.timestamp
        compacted = any(start <= ts <= end for start, end, _ in compactions)
        if not compacted:
            result.append(event)

    return result
