"""A2A protocol types — spec-compliant models (v1.0).

Implements the core A2A wire-format types based on the v1.0 specification:
  https://a2a-protocol.org/
  https://github.com/a2aproject/A2A

All models use camelCase aliases for JSON serialization to match the spec.
Enum values use SCREAMING_SNAKE_CASE as defined in the proto.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class A2AModel(BaseModel):
    """Base for all A2A models with camelCase alias support."""

    model_config = {"populate_by_name": True, "alias_generator": _to_camel}




class Part(A2AModel):
    """A2A v1.0 Part — exactly one of text/raw/url/data must be set.

    Common fields ``media_type`` and ``filename`` apply to any content type.
    """

    text: str | None = None
    raw: str | None = None  # base64-encoded bytes
    url: str | None = None
    data: dict[str, Any] | Any | None = None
    media_type: str | None = None
    filename: str | None = None
    metadata: dict[str, Any] | None = None


def text_part(text: str, media_type: str = "text/plain") -> Part:
    """Create a text Part.

    Parameters
    ----------
    text : str
        The text content.
    media_type : str
        MIME type, defaults to ``"text/plain"``.

    Returns
    -------
    Part
        A Part with the ``text`` field set.
    """
    return Part(text=text, media_type=media_type)


def file_part(
    *,
    url: str | None = None,
    raw: str | None = None,
    media_type: str | None = None,
    filename: str | None = None,
) -> Part:
    """Create a file Part (by URL or raw bytes).

    Parameters
    ----------
    url : str | None
        Remote URL pointing to the file content.
    raw : str | None
        Base64-encoded file bytes.
    media_type : str | None
        MIME type of the file.
    filename : str | None
        Optional filename hint.

    Returns
    -------
    Part
        A Part with the ``url`` or ``raw`` field set.
    """
    return Part(url=url, raw=raw, media_type=media_type, filename=filename)


def data_part(data: dict[str, Any], media_type: str = "application/json") -> Part:
    """Create a structured data Part.

    Parameters
    ----------
    data : dict[str, Any]
        Structured data payload.
    media_type : str
        MIME type, defaults to ``"application/json"``.

    Returns
    -------
    Part
        A Part with the ``data`` field set.
    """
    return Part(data=data, media_type=media_type)




class Role(str, Enum):
    """A2A v1.0 Message roles per spec."""

    USER = "user"
    AGENT = "agent"




class Message(A2AModel):
    """A2A v1.0 Message — a single turn from a user or agent.

    Attributes
    ----------
    message_id : str
        Unique ID for this message. Auto-generated.
    role : Role
        Speaker role: ``USER`` or ``AGENT``.
    parts : list[Part]
        Ordered content parts (text, data, files).
    context_id : str, optional
        Conversation/context identifier grouping related messages.
    task_id : str, optional
        ID of the task this message belongs to.
    reference_task_ids : list[str], optional
        IDs of earlier tasks this message references.
    extensions : list[str], optional
        A2A protocol extensions in use.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: Role
    parts: list[Part]
    context_id: str | None = None
    task_id: str | None = None
    reference_task_ids: list[str] | None = None
    extensions: list[str] | None = None
    metadata: dict[str, Any] | None = None




class Artifact(A2AModel):
    """A2A v1.0 Artifact — a durable output produced by a task.

    Attributes
    ----------
    artifact_id : str
        Unique ID for this artifact. Auto-generated.
    name : str, optional
        Human-readable filename or identifier.
    description : str, optional
        Short description of what the artifact contains.
    parts : list[Part]
        Ordered content parts holding the artifact payload.
    extensions : list[str], optional
        A2A protocol extensions attached to this artifact.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.
    """

    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str | None = None
    description: str | None = None
    parts: list[Part]
    extensions: list[str] | None = None
    metadata: dict[str, Any] | None = None




class TaskState(str, Enum):
    """A2A v1.0 task lifecycle states.

    States progress roughly as ``SUBMITTED`` → ``WORKING`` →
    (``COMPLETED`` | ``FAILED`` | ``CANCELED`` | ``REJECTED``). The
    ``INPUT_REQUIRED`` and ``AUTH_REQUIRED`` states represent pauses
    waiting for external input; the task returns to ``WORKING`` once
    the input is provided.

    Values
    ------
    UNSPECIFIED
        Placeholder when state has not been set.
    SUBMITTED
        Task accepted by the server, not yet started.
    WORKING
        Task actively executing.
    COMPLETED
        Terminal — task finished successfully.
    FAILED
        Terminal — task errored.
    CANCELED
        Terminal — task was cancelled.
    INPUT_REQUIRED
        Task is paused waiting for user input.
    REJECTED
        Terminal — task was refused before starting.
    AUTH_REQUIRED
        Task is paused waiting for authentication.
    """

    UNSPECIFIED = "TASK_STATE_UNSPECIFIED"
    SUBMITTED = "TASK_STATE_SUBMITTED"
    WORKING = "TASK_STATE_WORKING"
    COMPLETED = "TASK_STATE_COMPLETED"
    FAILED = "TASK_STATE_FAILED"
    CANCELED = "TASK_STATE_CANCELED"
    INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    REJECTED = "TASK_STATE_REJECTED"
    AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"


TERMINAL_STATES = {
    TaskState.COMPLETED,
    TaskState.CANCELED,
    TaskState.FAILED,
    TaskState.REJECTED,
}


class TaskStatus(A2AModel):
    """A2A v1.0 TaskStatus — current state snapshot of a task.

    Attributes
    ----------
    state : TaskState
        Lifecycle state of the task.
    message : Message, optional
        Last message emitted by the task (often an error or prompt).
    timestamp : str, optional
        ISO 8601 timestamp of the status transition.
    """

    state: TaskState
    message: Message | None = None
    timestamp: str | None = None


class Task(A2AModel):
    """A2A v1.0 Task — a unit of work tracked on an A2A server.

    Attributes
    ----------
    id : str
        Unique task ID. Auto-generated.
    context_id : str
        Conversation/context identifier grouping related tasks.
    status : TaskStatus
        Current lifecycle state.
    history : list[Message], optional
        Chronological messages exchanged during execution.
    artifacts : list[Artifact], optional
        Outputs produced by the task.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus
    history: list[Message] | None = None
    artifacts: list[Artifact] | None = None
    metadata: dict[str, Any] | None = None




class TaskStatusUpdateEvent(A2AModel):
    """A2A v1.0 TaskStatusUpdateEvent."""

    task_id: str
    context_id: str
    status: TaskStatus
    final: bool = False
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(A2AModel):
    """A2A v1.0 TaskArtifactUpdateEvent."""

    task_id: str
    context_id: str
    artifact: Artifact
    append: bool | None = None
    last_chunk: bool | None = None
    metadata: dict[str, Any] | None = None




class AgentProvider(A2AModel):
    """Agent provider information."""

    organization: str
    url: str


class AgentSkill(A2AModel):
    """A2A v1.0 AgentSkill — a capability advertised on an agent card.

    Attributes
    ----------
    id : str
        Unique skill identifier within the agent.
    name : str
        Human-readable skill name.
    description : str
        What the skill does. Used by remote callers for selection.
    tags : list[str]
        Tags for discovery and filtering.
    examples : list[str], optional
        Sample prompts demonstrating the skill.
    input_modes : list[str], optional
        MIME types the skill accepts as input.
    output_modes : list[str], optional
        MIME types the skill produces as output.
    """

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None
    input_modes: list[str] | None = None
    output_modes: list[str] | None = None


class AgentCapabilities(A2AModel):
    """A2A v1.0 AgentCapabilities — feature flags on an agent card.

    Attributes
    ----------
    streaming : bool, optional
        True if the agent supports streaming task updates.
    push_notifications : bool, optional
        True if the agent can push updates to a callback URL.
    extended_agent_card : bool, optional
        True if the agent exposes extended card metadata beyond v1.0.
    """

    streaming: bool | None = None
    push_notifications: bool | None = None
    extended_agent_card: bool | None = None


class AgentInterface(A2AModel):
    """A2A v1.0 AgentInterface — protocol binding endpoint."""

    url: str
    protocol_binding: str = "JSONRPC"
    protocol_version: str = "1.0"
    tenant: str | None = None


class AgentCard(A2AModel):
    """A2A v1.0 Agent Card — discovery manifest for a remote agent.

    Attributes
    ----------
    name : str
        Agent display name.
    description : str
        What this agent does.
    supported_interfaces : list[AgentInterface]
        Protocol bindings (e.g. JSON-RPC endpoints) where the agent
        can be reached.
    version : str
        Semantic version of the agent.
    capabilities : AgentCapabilities
        Feature flags (streaming, push notifications, etc.).
    skills : list[AgentSkill]
        Skills advertised by the agent.
    default_input_modes : list[str]
        MIME types accepted by default when a skill does not override.
    default_output_modes : list[str]
        MIME types produced by default when a skill does not override.
    provider : AgentProvider, optional
        Organization hosting the agent.
    documentation_url : str, optional
        Link to agent documentation.
    icon_url : str, optional
        Link to an icon displayed in clients.
    """

    name: str
    description: str
    supported_interfaces: list[AgentInterface] = Field(default_factory=list)
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    default_input_modes: list[str] = Field(
        default_factory=lambda: ["application/json"],
    )
    default_output_modes: list[str] = Field(
        default_factory=lambda: ["application/json"],
    )
    provider: AgentProvider | None = None
    documentation_url: str | None = None
    icon_url: str | None = None




class JSONRPCError(A2AModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any | None = None


class JSONRPCRequest(A2AModel):
    """JSON-RPC 2.0 request."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(A2AModel):
    """JSON-RPC 2.0 response."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    result: Any | None = None
    error: JSONRPCError | None = None




class SendMessageConfiguration(A2AModel):
    """A2A v1.0 SendMessageConfiguration."""

    accepted_output_modes: list[str] | None = None
    history_length: int | None = None
    return_immediately: bool | None = None


class MessageSendParams(A2AModel):
    """Parameters for SendMessage / SendStreamingMessage."""

    message: Message
    configuration: SendMessageConfiguration | None = None
    metadata: dict[str, Any] | None = None


class TaskQueryParams(A2AModel):
    """Parameters for GetTask."""

    id: str
    history_length: int | None = None


class TaskIdParams(A2AModel):
    """Parameters for CancelTask."""

    id: str
    metadata: dict[str, Any] | None = None




class A2AErrorCode:
    """A2A v1.0 JSON-RPC error codes."""

    # Standard JSON-RPC
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # A2A-specific
    TASK_NOT_FOUND = -32001
    TASK_NOT_CANCELABLE = -32002
    PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
    UNSUPPORTED_OPERATION = -32004
    CONTENT_TYPE_NOT_SUPPORTED = -32005
    INVALID_AGENT_RESPONSE = -32006
    EXTENDED_AGENT_CARD_NOT_CONFIGURED = -32007
    EXTENSION_SUPPORT_REQUIRED = -32008
    VERSION_NOT_SUPPORTED = -32009
