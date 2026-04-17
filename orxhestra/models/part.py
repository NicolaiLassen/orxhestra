"""Content and Part models — typed multimodal content for events.

Aligns with the A2A protocol's Part types (TextPart, DataPart, FilePart).
Events carry a ``Content`` object with a list of typed parts instead of
loose string fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TextPart(BaseModel):
    """A plain text content part.

    Attributes
    ----------
    type : str
        Part discriminator, always ``"text"``.
    text : str
        The text payload.
    metadata : dict[str, Any]
        Arbitrary key-value metadata attached to this part.
    """

    type: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataPart(BaseModel):
    """A structured data content part (JSON-serializable dict).

    Use this when the payload is structured (e.g. a schema-constrained
    LLM response, a tool result with fields). Use :class:`TextPart`
    for free-form prose and :class:`FilePart` for binary/file data.

    Attributes
    ----------
    type : str
        Part discriminator, always ``"data"``.
    data : dict[str, Any]
        The structured payload. Must be JSON-serializable.
    metadata : dict[str, Any]
        Arbitrary key-value metadata attached to this part.
    """

    type: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class FilePart(BaseModel):
    """A file content part — either inline bytes or a URI reference.

    Use ``inline_bytes`` for small payloads that should travel with
    the event (images under ~100 KB, brief attachments). Use ``uri``
    for larger files stored in an artifact service or object store —
    this keeps the event small and lets consumers fetch on demand.

    Attributes
    ----------
    type : str
        Part discriminator, always ``"file"``.
    uri : str, optional
        URI pointing to the file (e.g. GCS, S3, HTTP).
    inline_bytes : str, optional
        Base64-encoded file content for inline transfer.
    mime_type : str, optional
        MIME type of the file (e.g. ``"image/png"``).
    name : str, optional
        Filename.
    metadata : dict[str, Any]
        Arbitrary key-value metadata attached to this part.
    """

    type: Literal["file"] = "file"
    uri: str | None = None
    inline_bytes: str | None = None
    mime_type: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ThinkingPart(BaseModel):
    """A thinking/reasoning content part from extended thinking models.

    Attributes
    ----------
    type : str
        Part discriminator, always ``"thinking"``.
    thinking : str
        The thinking/reasoning text.
    metadata : dict[str, Any]
        Arbitrary key-value metadata attached to this part.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallPart(BaseModel):
    """A tool/function call part — the agent wants to invoke a tool.

    Attributes
    ----------
    type : str
        Part discriminator, always ``"tool_call"``.
    tool_call_id : str
        Unique identifier for this tool call, used to match the
        corresponding ``ToolResponsePart``.
    tool_name : str
        Name of the tool to invoke.
    args : dict[str, Any]
        Arguments to pass to the tool.
    metadata : dict[str, Any]
        Arbitrary key-value metadata (e.g. ``{"interactive": True}``).
    """

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResponsePart(BaseModel):
    """A tool/function response part — result from a tool execution.

    Attributes
    ----------
    type : str
        Part discriminator, always ``"tool_response"``.
    tool_call_id : str
        Identifier matching the originating ``ToolCallPart``.
    tool_name : str
        Name of the tool that produced this response.
    result : str
        Successful result text. Empty string when the call errored.
    error : str, optional
        Error message if the tool call failed.
    metadata : dict[str, Any]
        Arbitrary key-value metadata attached to this part.
    """

    type: Literal["tool_response"] = "tool_response"
    tool_call_id: str
    tool_name: str
    result: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


Part = TextPart | DataPart | FilePart | ThinkingPart | ToolCallPart | ToolResponsePart


class Content(BaseModel):
    """Container for multimodal content — a list of typed parts.

    Mirrors the A2A protocol's :attr:`Message.parts`.

    Attributes
    ----------
    role : str, optional
        Who produced this content: "user", "model", or "agent".
    parts : list[Part]
        Ordered list of content parts.

    See Also
    --------
    TextPart : Plain text.
    DataPart : Structured JSON-serializable data.
    FilePart : File reference (URL or inline bytes).
    ThinkingPart : Reasoning/chain-of-thought payload.
    ToolCallPart : Tool-call request from the model.
    ToolResponsePart : Tool-call result payload.
    Event.content : Field that carries a ``Content`` on every event.
    """

    role: str | None = None
    parts: list[Part] = Field(default_factory=list)

    @staticmethod
    def from_text(text: str, *, role: str | None = None) -> Content:
        """Create a Content with a single :class:`TextPart`.

        Parameters
        ----------
        text : str
            Text payload.
        role : str, optional
            Message role (``"user"``, ``"model"``, ``"agent"``).

        Returns
        -------
        Content
            A new ``Content`` wrapping a single ``TextPart``.
        """
        return Content(role=role, parts=[TextPart(text=text)])

    @staticmethod
    def from_thinking(thinking: str, *, role: str | None = None) -> Content:
        """Create a Content with a single :class:`ThinkingPart`.

        Parameters
        ----------
        thinking : str
            Chain-of-thought or reasoning text.
        role : str, optional
            Message role.

        Returns
        -------
        Content
            A new ``Content`` wrapping a single ``ThinkingPart``.
        """
        return Content(role=role, parts=[ThinkingPart(thinking=thinking)])

    @staticmethod
    def from_data(data: dict[str, Any], *, role: str | None = None) -> Content:
        """Create a Content with a single :class:`DataPart`.

        Parameters
        ----------
        data : dict[str, Any]
            JSON-serializable structured payload.
        role : str, optional
            Message role.

        Returns
        -------
        Content
            A new ``Content`` wrapping a single ``DataPart``.
        """
        return Content(role=role, parts=[DataPart(data=data)])

    @property
    def text(self) -> str:
        """Concatenate all text parts."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def thinking(self) -> str:
        """Concatenate all thinking parts."""
        return "".join(p.thinking for p in self.parts if isinstance(p, ThinkingPart))

    @property
    def data(self) -> dict[str, Any] | None:
        """Return the first DataPart's data, or None."""
        for p in self.parts:
            if isinstance(p, DataPart):
                return p.data
        return None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """Return all ToolCallPart entries."""
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def tool_responses(self) -> list[ToolResponsePart]:
        """Return all ToolResponsePart entries."""
        return [p for p in self.parts if isinstance(p, ToolResponsePart)]

    @property
    def has_tool_calls(self) -> bool:
        """Return True if content contains any tool call parts."""
        return any(isinstance(p, ToolCallPart) for p in self.parts)
