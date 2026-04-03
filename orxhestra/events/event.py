"""Unified event model for agent execution.

A single ``Event`` class carries all information via ``Content`` parts
and ``metadata``. No subclasses — use ``EventType`` and helper methods
like ``is_final_response()`` to classify events.

Conversion to/from LangChain messages is built in via
``to_langchain_message()`` and ``from_langchain_message()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from orxhestra.events.event_actions import EventActions
from orxhestra.models.llm_response import LlmResponse
from orxhestra.models.part import Content, ToolCallPart, ToolResponsePart


class EventType(str, Enum):
    """All possible event types emitted during agent execution."""

    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_RESPONSE = "tool_response"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"




class Event(BaseModel):
    """Single event type for everything emitted during agent execution.

    Carries a ``content: Content`` field with typed parts (TextPart,
    DataPart, FilePart, ToolCallPart, ToolResponsePart). Use
    ``metadata`` for extra context (react steps, error info, etc.).

    Attributes
    ----------
    id : str
        Unique identifier for this event.
    type : EventType
        The kind of event.
    timestamp : float
        Unix timestamp of when this event was created.
    invocation_id : str
        ID of the agent invocation that produced this event.
    session_id : str, optional
        The session this event belongs to.
    author : str
        Who produced this event: "user" or the agent name.
    agent_name : str, optional
        Name of the agent that emitted this event.
    branch : str
        Dot-separated path showing which agent in the tree produced this.
    partial : bool, optional
        When True, the event is an incomplete streaming chunk.
    turn_complete : bool
        When False, more events are expected in this turn (streaming).
    content : Content
        The event payload as typed parts.
    actions : EventActions
        Side-effects to apply when this event is committed to the session.
    metadata : dict[str, Any]
        Arbitrary extra metadata (react_step, error, scratchpad, etc.).
    llm_response : LlmResponse, optional
        The raw LLM response (internal use).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    timestamp: float = Field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )
    invocation_id: str = ""
    session_id: str | None = None
    author: str = ""
    agent_name: str | None = None
    branch: str = ""
    partial: bool | None = None
    turn_complete: bool = True
    content: Content = Field(default_factory=Content)
    actions: EventActions = Field(default_factory=EventActions)
    metadata: dict[str, Any] = Field(default_factory=dict)
    llm_response: LlmResponse | None = None

    @property
    def text(self) -> str:
        """Concatenate all text parts in content."""
        return self.content.text

    @property
    def data(self) -> dict[str, Any] | None:
        """Return the first DataPart's data, or None."""
        return self.content.data

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Return all ToolCallPart entries from content."""
        return self.content.tool_calls

    @property
    def has_tool_calls(self) -> bool:
        """Return True if content contains any tool call parts."""
        return self.content.has_tool_calls

    def is_final_response(self) -> bool:
        """Return True if this event represents the agent's final answer.

        A final response is a complete (non-partial) AGENT_MESSAGE that
        has text or data content, no pending tool calls, and is not an
        intermediate step (thought, action, observation, error).
        """
        if self.partial:
            return False
        if self.type != EventType.AGENT_MESSAGE:
            return False
        if self.has_tool_calls:
            return False
        if self.metadata.get("react_step"):
            return False
        if self.metadata.get("error"):
            return False
        if self.actions.skip_summarization:
            return True
        return bool(self.text or self.data)

    def to_langchain_message(self) -> BaseMessage:
        """Convert this event to the appropriate LangChain message type."""
        if self.type == EventType.USER_MESSAGE:
            return HumanMessage(content=self.text)
        elif self.type == EventType.TOOL_RESPONSE:
            parts = self.content.tool_responses
            if parts:
                return ToolMessage(
                    content=parts[0].result or parts[0].error or "",
                    tool_call_id=parts[0].tool_call_id,
                )
            return ToolMessage(content=self.text, tool_call_id="")
        else:  # AGENT_MESSAGE
            if self.has_tool_calls:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": tc.tool_call_id,
                            "name": tc.tool_name,
                            "args": tc.args,
                        }
                        for tc in self.tool_calls
                    ],
                )
            return AIMessage(content=self.text)

    @staticmethod
    def from_langchain_message(msg: BaseMessage, **kwargs: Any) -> Event:
        """Create an Event from a LangChain message."""
        if isinstance(msg, HumanMessage):
            return Event(
                type=EventType.USER_MESSAGE,
                author="user",
                content=Content.from_text(
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                ),
                **kwargs,
            )
        elif isinstance(msg, ToolMessage):
            return Event(
                type=EventType.TOOL_RESPONSE,
                content=Content(
                    parts=[
                        ToolResponsePart(
                            tool_call_id=getattr(msg, "tool_call_id", ""),
                            tool_name="",
                            result=msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content),
                        )
                    ]
                ),
                **kwargs,
            )
        else:  # AIMessage
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                parts = [
                    ToolCallPart(
                        tool_call_id=tc.get("id", ""),
                        tool_name=tc.get("name", ""),
                        args=tc.get("args", {}),
                    )
                    for tc in tool_calls
                ]
                return Event(
                    type=EventType.AGENT_MESSAGE,
                    content=Content(parts=parts),
                    **kwargs,
                )
            return Event(
                type=EventType.AGENT_MESSAGE,
                content=Content.from_text(
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                ),
                **kwargs,
            )

    @property
    def tool_name(self) -> str | None:
        """First tool call or tool response name, or None."""
        if self.content.tool_calls:
            return self.content.tool_calls[0].tool_name
        if self.content.tool_responses:
            return self.content.tool_responses[0].tool_name
        return None

    @property
    def tool_input(self) -> dict | None:
        """First tool call args, or None."""
        if self.content.tool_calls:
            return self.content.tool_calls[0].args
        return None

    @property
    def error(self) -> str | None:
        """Error message from tool response or metadata."""
        if self.content.tool_responses:
            return self.content.tool_responses[0].error
        if self.metadata.get("error"):
            return self.text
        return None

    @staticmethod
    def new_id() -> str:
        """Generate a new unique event ID."""
        return str(uuid4())
