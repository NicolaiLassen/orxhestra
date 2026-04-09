"""LlmResponse - a structured wrapper around a LangChain AIMessage.

LangChain's ``BaseChatModel.ainvoke()`` returns an ``AIMessage``. LlmResponse
normalises that into a stable, serialisable Pydantic model so that agents,
tools, and events never need to import LangChain types directly.

Events that are direct model outputs (AGENT_MESSAGE events with tool calls)
carry an optional ``llm_response`` field of this type, preserving token usage,
model version, and tool-call metadata alongside the event payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage


class LlmResponse(BaseModel):
    """Wrapper around a LangChain AIMessage returned by the model.

    Preserves token usage, model version, and raw message content without
    coupling the rest of the event model to LangChain types directly. Events
    store this as an optional field - it is only populated for events that
    are the direct result of an LLM call.

    Parameters
    ----------
    raw : AIMessage
        The raw LangChain AIMessage returned by the model.

    Attributes
    ----------
    text : str
        Plain text content extracted from the message. Empty string if the
        message has no text content (e.g. a pure tool-call response).
    model_version : str, optional
        Model identifier returned in the response metadata, if available.
    input_tokens : int, optional
        Number of prompt tokens used, from usage metadata if available.
    output_tokens : int, optional
        Number of completion tokens used, from usage metadata if available.
    tool_calls : list[dict[str, Any]]
        Tool calls extracted from the message. Each entry has keys:
        ``id``, ``name``, ``args``.
    partial : bool
        True if this is a streaming chunk that is not yet complete.
    raw_message : Any
        The original AIMessage object for callers that need the full
        LangChain representation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str = ""
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    partial: bool = False
    raw_message: Any = None

    @classmethod
    def from_ai_message(cls, message: AIMessage) -> LlmResponse:
        """Build an LlmResponse from a LangChain AIMessage.

        Parameters
        ----------
        message : AIMessage
            The message returned by ``llm.ainvoke()``.

        Returns
        -------
        LlmResponse
            A populated response wrapper.
        """
        from orxhestra.models.content_parser import parse_content_blocks

        text, _ = parse_content_blocks(message.content)

        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens") if usage else None
        output_tokens = usage.get("output_tokens") if usage else None

        model_version = (
            message.response_metadata.get("model_name") or message.response_metadata.get("model")
            if hasattr(message, "response_metadata") and message.response_metadata
            else None
        )

        tool_calls = [
            {"id": tc.get("id", ""), "name": tc["name"], "args": tc["args"]}
            for tc in (message.tool_calls or [])
        ]

        return cls(
            text=text,
            model_version=model_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            partial=False,
            raw_message=message,
        )

    @property
    def has_tool_calls(self) -> bool:
        """Return True if the model requested one or more tool calls.

        Returns
        -------
        bool
        """
        return bool(self.tool_calls)
