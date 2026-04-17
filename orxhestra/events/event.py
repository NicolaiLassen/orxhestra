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
    """All possible event types emitted during agent execution.

    Values
    ------
    USER_MESSAGE
        Input from the user that starts or continues a turn.
    AGENT_MESSAGE
        Output from an agent — either a streaming token chunk
        (``partial=True``) or a final answer.
    TOOL_RESPONSE
        Result from a tool invocation. Matches a prior tool call
        by ``ToolCallPart.tool_call_id``.
    AGENT_START
        Internal lifecycle marker — an agent has begun its turn.
    AGENT_END
        Internal lifecycle marker — an agent has finished its turn.

    See Also
    --------
    Event : Envelope that carries one of these types.
    """

    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    TOOL_RESPONSE = "tool_response"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"


class Event(BaseModel):
    """Single event type for everything emitted during agent execution.

    Carries a :class:`Content` payload with typed parts
    (:class:`TextPart`, :class:`DataPart`, :class:`FilePart`,
    :class:`ToolCallPart`, :class:`ToolResponsePart`). Use ``metadata``
    for extra context (react steps, error info, etc.).

    See Also
    --------
    EventType : Enum of event categories.
    EventActions : Side-effects attached to the event.
    Content : Container holding the event's typed parts.
    Runner.astream : Primary producer of events.
    Middleware.on_event : Hook for transforming or dropping events.

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
    signature : str, optional
        Base64url-encoded Ed25519 signature over the canonical JSON
        representation of the event's signable fields (id, type,
        timestamp, agent_name, content text).  Set automatically
        when the emitting agent has a signing key.
    signer_did : str
        The ``did:key`` identifier of the signing agent.  Verifiers
        use this to resolve the public key via
        ``orxhestra.auth.did_key_to_public_key()``.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
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
    signature: str | None = None
    signer_did: str = ""

    @property
    def text(self) -> str:
        """Concatenate all text parts in content."""
        return self.content.text

    @property
    def thinking(self) -> str:
        """Concatenate all thinking parts in content."""
        return self.content.thinking

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
        """Convert this event to the appropriate LangChain message type.

        Returns
        -------
        BaseMessage
            ``HumanMessage`` for ``USER_MESSAGE``, ``ToolMessage`` for
            ``TOOL_RESPONSE``, or ``AIMessage`` otherwise. Tool calls
            are preserved on ``AIMessage`` via its ``tool_calls`` field.
        """
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
        """Create an Event from a LangChain message.

        Parameters
        ----------
        msg : BaseMessage
            A ``HumanMessage``, ``ToolMessage``, or ``AIMessage``.
        **kwargs : Any
            Extra fields forwarded to the :class:`Event` constructor
            (e.g. ``session_id``, ``invocation_id``, ``agent_name``).

        Returns
        -------
        Event
            ``USER_MESSAGE`` for ``HumanMessage``, ``TOOL_RESPONSE`` for
            ``ToolMessage``, ``AGENT_MESSAGE`` otherwise. Tool calls on
            ``AIMessage`` are preserved as ``ToolCallPart`` entries.
        """
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

    def signable_payload(self) -> dict[str, Any]:
        """Return the canonical dict of fields included in the signature.

        Returns
        -------
        dict[str, Any]
            Deterministic subset of event fields used for signing
            and verification.
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name or "",
            "branch": self.branch,
            "content_text": self.text,
        }

    @property
    def is_signed(self) -> bool:
        """Return ``True`` if this event carries a signature."""
        return self.signature is not None and bool(self.signer_did)

    def verify_signature(self) -> bool:
        """Verify this event's signature using the signer's DID.

        Resolves the public key from ``signer_did`` via
        ``orxhestra.auth.did_key_to_public_key()`` and verifies the
        signature over :meth:`signable_payload`.

        Returns
        -------
        bool
            ``True`` if the signature is valid.  ``False`` if the
            event is unsigned, the DID is invalid, or verification
            fails.

        Raises
        ------
        ImportError
            If ``orxhestra[auth]`` is not installed.
        """
        if not self.is_signed:
            return False

        from orxhestra.auth.crypto import (
            did_key_to_public_key,
            verify_json_signature,
        )

        try:
            public_key = did_key_to_public_key(self.signer_did)
        except ValueError:
            return False

        return verify_json_signature(
            public_key,
            self.signable_payload(),
            self.signature,
        )

    @staticmethod
    def new_id() -> str:
        """Generate a new unique event ID.

        Returns
        -------
        str
            UUID4 hex string suitable for :attr:`Event.id`.
        """
        return str(uuid4())
