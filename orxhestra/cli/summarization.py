"""Context summarization - prevent context window overflow.

When the conversation gets too long, summarize older messages to free
up context space. Also provides a /compact command for manual summarization.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from orxhestra.events.event import Event, EventType

# Summarize when conversation exceeds this many non-partial events
DEFAULT_EVENT_THRESHOLD: int = 40

# Keep the most recent N events unsummarized
KEEP_RECENT: int = 10

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
    threshold: int = DEFAULT_EVENT_THRESHOLD,
) -> list[Event] | None:
    """Summarize old events if the session exceeds the threshold.

    Returns a new event list with old events replaced by a summary event,
    or None if no summarization was needed.
    """
    # Filter to non-partial, substantive events
    substantive: list[Event] = [
        e for e in events
        if not e.partial and e.type in (
            EventType.USER_MESSAGE,
            EventType.AGENT_MESSAGE,
            EventType.TOOL_RESPONSE,
        )
    ]

    if len(substantive) < threshold:
        return None

    # Split into old (summarize) and recent (keep)
    split_idx: int = len(events) - KEEP_RECENT
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
