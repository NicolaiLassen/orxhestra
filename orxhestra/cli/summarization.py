"""Context summarization — prevent context window overflow.

When the conversation gets too long (measured by character count, not
event count), summarize older messages to free up context space.  Also
provides a ``/compact`` command for manual summarization.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from orxhestra.events.event import Event, EventType

# Compact when conversation content exceeds this many characters (~10k tokens)
DEFAULT_CHAR_THRESHOLD: int = 40_000

# Keep the most recent events totalling at least this many characters (~2k tokens)
RETENTION_CHARS: int = 8_000

_SUMMARIZE_PROMPT: str = """\
Summarize the following conversation history concisely. Focus on:
1. What tasks were completed and their outcomes
2. What files were created/modified and why
3. Key decisions made
4. Current state of the work

Be brief but preserve all actionable context. Output only the summary.

Conversation:
{conversation}
"""


def _estimate_event_chars(event: Event) -> int:
    """Estimate the character count of an event's content."""
    if event.partial:
        return 0
    total: int = 0
    if event.text:
        total += len(event.text)
    if event.type == EventType.TOOL_RESPONSE:
        for tr in event.content.tool_responses:
            if tr.result:
                total += len(tr.result)
    return total


def _events_to_text(events: list[Event]) -> str:
    """Convert events to a readable text for summarization."""
    lines: list[str] = []
    for event in events:
        if event.partial:
            continue
        if event.type == EventType.USER_MESSAGE:
            lines.append(f"User: {event.text}")
        elif event.type == EventType.AGENT_MESSAGE and event.text:
            lines.append(f"Agent: {event.text[:500]}")
        elif event.type == EventType.TOOL_RESPONSE and event.text:
            lines.append(f"Tool result: {event.text[:200]}")
    return "\n".join(lines)


async def summarize_session(
    llm: BaseChatModel,
    events: list[Event],
    char_threshold: int = DEFAULT_CHAR_THRESHOLD,
    retention_chars: int = RETENTION_CHARS,
) -> list[Event] | None:
    """Summarize old events if the session exceeds the character threshold.

    Parameters
    ----------
    llm : BaseChatModel
        LLM to generate the summary.
    events : list[Event]
        The session's event list.
    char_threshold : int
        Minimum total characters before compaction triggers.
    retention_chars : int
        Keep the most recent events totalling at least this many characters.

    Returns
    -------
    list[Event] or None
        A new event list with old events replaced by a summary, or None
        if no summarization was needed.
    """
    total_chars: int = sum(_estimate_event_chars(e) for e in events)

    if total_chars < char_threshold:
        return None

    # Walk backwards to find the split point that retains retention_chars
    cumulative: int = 0
    split_idx: int = len(events)
    for i in range(len(events) - 1, -1, -1):
        cumulative += _estimate_event_chars(events[i])
        if cumulative >= retention_chars:
            split_idx = i + 1
            break
    else:
        split_idx = 0

    if split_idx <= 0:
        return None

    old_events: list[Event] = events[:split_idx]
    recent_events: list[Event] = events[split_idx:]

    # Generate summary
    conversation_text: str = _events_to_text(old_events)
    if not conversation_text.strip():
        return None

    prompt: str = _SUMMARIZE_PROMPT.format(conversation=conversation_text)
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a conversation summarizer."),
        HumanMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        summary_text: str = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Summarization failed", exc_info=True
        )
        return None

    # Create a summary event to replace old events
    from orxhestra.models.part import Content

    summary_event = Event(
        type=EventType.AGENT_MESSAGE,
        author="system",
        content=Content.from_text(
            f"[Previous conversation summary]\n{summary_text}"
        ),
        metadata={"summary": True},
    )

    return [summary_event, *recent_events]
