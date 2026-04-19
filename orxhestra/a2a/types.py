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
    """Convert a ``snake_case`` identifier to ``camelCase``.

    Used as the :mod:`pydantic` alias generator for
    :class:`A2AModel` so models serialise with the camelCase field
    names the A2A wire format expects.

    Parameters
    ----------
    name : str
        A ``snake_case`` field name.

    Returns
    -------
    str
        The ``camelCase`` equivalent.  Single-word names pass through
        unchanged.
    """
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class A2AModel(BaseModel):
    """Base class for every A2A wire type in orxhestra.

    Enables population by either Python (snake_case) or JSON
    (camelCase) names and sets :func:`_to_camel` as the alias
    generator so every field serialises in camelCase for the wire.

    See Also
    --------
    _to_camel : Alias function used by this model.
    """

    model_config = {"populate_by_name": True, "alias_generator": _to_camel}




class Part(A2AModel):
    """A2A v1.0 :class:`Part` — a single content chunk on a :class:`Message`.

    Exactly one of ``text``, ``raw``, ``url``, or ``data`` must be
    set.  The common ``media_type`` and ``filename`` fields apply to
    any content type.

    Attributes
    ----------
    text : str, optional
        Plain text content when the part is textual.
    raw : str, optional
        Base64-encoded binary payload.
    url : str, optional
        URL pointing to an external payload.
    data : dict[str, Any], optional
        Structured JSON-serializable payload.
    media_type : str, optional
        MIME type of the payload.
    filename : str, optional
        Original filename, for binary and URL parts.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.

    See Also
    --------
    text_part : Helper constructor for ``text`` parts.
    file_part : Helper constructor for binary / URL parts.
    data_part : Helper constructor for structured ``data`` parts.
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
    """A2A v1.0 message role — who produced the :class:`Message`.

    Values
    ------
    USER
        The message originated from a human or upstream caller.
    AGENT
        The message was produced by an agent (local or remote).

    See Also
    --------
    Message.role : Field that carries this value on the wire.
    """

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
    """A2A v1.0 streaming event — a change in :class:`TaskStatus`.

    Emitted by ``SendStreamingMessage`` whenever a task transitions
    to a new state or surfaces a progress message.

    Attributes
    ----------
    task_id : str
        Task whose status is changing.
    context_id : str
        Conversation context the task belongs to.
    status : TaskStatus
        New status snapshot (state + optional message + timestamp).
    final : bool
        When ``True``, no further status updates will be emitted for
        this task.  Used by clients to close the stream.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.

    See Also
    --------
    TaskArtifactUpdateEvent : Sibling event for artifact deltas.
    events_to_a2a_stream : Produces these events from SDK events.
    """

    task_id: str
    context_id: str
    status: TaskStatus
    final: bool = False
    metadata: dict[str, Any] | None = None


class TaskArtifactUpdateEvent(A2AModel):
    """A2A v1.0 streaming event — an :class:`Artifact` delta for a task.

    Emitted by ``SendStreamingMessage`` when an agent produces a
    piece of final output (answer, tool result).  Streamed in chunks
    with ``append`` / ``last_chunk`` flags.

    Attributes
    ----------
    task_id : str
        Task the artifact belongs to.
    context_id : str
        Conversation context the task belongs to.
    artifact : Artifact
        Artifact payload (new or extending an earlier chunk).
    append : bool, optional
        When ``True``, ``artifact.parts`` extend the previously sent
        chunk rather than replacing it.
    last_chunk : bool, optional
        When ``True``, this is the final chunk for the artifact.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.

    See Also
    --------
    TaskStatusUpdateEvent : Sibling event for status transitions.
    Artifact : The payload carried by this event.
    """

    task_id: str
    context_id: str
    artifact: Artifact
    append: bool | None = None
    last_chunk: bool | None = None
    metadata: dict[str, Any] | None = None




class AgentProvider(A2AModel):
    """Organization responsible for hosting an agent.

    Attributes
    ----------
    organization : str
        Human-readable organization name.
    url : str
        URL to the organization's home page or documentation.
    """

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
    """A2A v1.0 :class:`AgentInterface` — a protocol binding endpoint.

    Describes one way the agent can be reached.  An :class:`AgentCard`
    typically advertises a single :class:`AgentInterface`, but may
    advertise several for multi-tenant deployments.

    Attributes
    ----------
    url : str
        URL where the binding is reachable.
    protocol_binding : str
        Wire format.  Currently ``"JSONRPC"``.
    protocol_version : str
        Spec version the endpoint implements.  ``"1.0"`` for A2A v1.
    tenant : str, optional
        Tenant identifier for multi-tenant deployments.
    """

    url: str
    protocol_binding: str = "JSONRPC"
    protocol_version: str = "1.0"
    tenant: str | None = None


class VerificationMethod(A2AModel):
    """W3C DID Core verification method advertised on an :class:`AgentCard`.

    Allows A2A peers to cryptographically identify the server that
    published the card.  Consumers resolve this via
    :class:`orxhestra.security.did.DidResolver`.

    Attributes
    ----------
    id : str
        Fully-qualified verification method identifier, typically
        ``"<did>#<fragment>"``.
    type : str
        Key type.  ``"Ed25519VerificationKey2020"`` for the keys
        orxhestra produces.
    controller : str
        DID that owns this key.
    public_key_multibase : str
        Multibase-encoded public key (z-prefixed base58btc).
    """

    id: str
    type: str = "Ed25519VerificationKey2020"
    controller: str
    public_key_multibase: str


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
    controller : str, optional
        DID of the entity controlling this agent.  Populated when the
        server has a signing identity configured.  Peers use this to
        resolve the matching :class:`VerificationMethod`.
    verification_method : list[VerificationMethod], optional
        Ed25519 keys peers can use to verify signed messages from the
        server.  Populated when the server has a signing identity
        configured.
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
    controller: str | None = None
    verification_method: list[VerificationMethod] | None = None




class JSONRPCError(A2AModel):
    """JSON-RPC 2.0 error object attached to an error :class:`JSONRPCResponse`.

    Attributes
    ----------
    code : int
        Numeric error code.  See :class:`A2AErrorCode` for the
        A2A-specific values.
    message : str
        Short human-readable error message.
    data : Any, optional
        Free-form error payload (stack trace excerpts, offending
        parameter, etc.).
    """

    code: int
    message: str
    data: Any | None = None


class JSONRPCRequest(A2AModel):
    """JSON-RPC 2.0 request envelope sent to an A2A server.

    Attributes
    ----------
    jsonrpc : str
        Protocol marker, always ``"2.0"``.
    id : str or int
        Correlation id echoed on the response.
    method : str
        RPC method name (``"SendMessage"``, ``"GetTask"``, ...).
    params : dict[str, Any], optional
        Method-specific parameters.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] | None = None


class JSONRPCResponse(A2AModel):
    """JSON-RPC 2.0 response envelope returned by an A2A server.

    Attributes
    ----------
    jsonrpc : str
        Protocol marker, always ``"2.0"``.
    id : str or int
        Correlation id echoed from the originating request.
    result : Any, optional
        Successful result payload.  Mutually exclusive with ``error``.
    error : JSONRPCError, optional
        Error detail when the call failed.
    """

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    result: Any | None = None
    error: JSONRPCError | None = None




class SendMessageConfiguration(A2AModel):
    """Per-call send options for ``SendMessage`` / ``SendStreamingMessage``.

    Attributes
    ----------
    accepted_output_modes : list[str], optional
        Output MIME types the caller will accept.  Overrides the
        agent card's ``default_output_modes`` for this call.
    history_length : int, optional
        Maximum number of prior messages to include in the task
        ``history`` field.
    return_immediately : bool, optional
        When ``True``, the server returns the newly created task
        without waiting for completion.
    """

    accepted_output_modes: list[str] | None = None
    history_length: int | None = None
    return_immediately: bool | None = None


class MessageSendParams(A2AModel):
    """Parameter object for ``SendMessage`` / ``SendStreamingMessage``.

    Attributes
    ----------
    message : Message
        The message to deliver to the agent.
    configuration : SendMessageConfiguration, optional
        Per-call overrides.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.
    """

    message: Message
    configuration: SendMessageConfiguration | None = None
    metadata: dict[str, Any] | None = None


class TaskQueryParams(A2AModel):
    """Parameter object for ``GetTask``.

    Attributes
    ----------
    id : str
        Task identifier to fetch.
    history_length : int, optional
        Maximum number of messages to include in the returned
        ``task.history``.
    """

    id: str
    history_length: int | None = None


class TaskIdParams(A2AModel):
    """Parameter object for ``CancelTask``.

    Attributes
    ----------
    id : str
        Task identifier to cancel.
    metadata : dict[str, Any], optional
        Arbitrary implementation-specific metadata.
    """

    id: str
    metadata: dict[str, Any] | None = None




class A2AErrorCode:
    """A2A v1.0 JSON-RPC error codes.

    Groups the standard JSON-RPC codes (``-32700`` through
    ``-32603``) with the A2A-specific extensions (``-32001`` through
    ``-32009``) per the v1.0 spec.  Values are class attributes so
    they can be referenced as ``A2AErrorCode.TASK_NOT_FOUND`` without
    instantiating.

    Attributes
    ----------
    PARSE_ERROR : int
        JSON could not be parsed.
    INVALID_REQUEST : int
        JSON-RPC envelope is malformed.
    METHOD_NOT_FOUND : int
        Requested RPC method is not implemented.
    INVALID_PARAMS : int
        Parameters failed validation.
    INTERNAL_ERROR : int
        Unhandled server-side exception.
    TASK_NOT_FOUND : int
        ``GetTask`` / ``CancelTask`` referenced an unknown task.
    TASK_NOT_CANCELABLE : int
        ``CancelTask`` targeted a task in a terminal state.
    PUSH_NOTIFICATION_NOT_SUPPORTED : int
        Agent does not implement push notifications.
    UNSUPPORTED_OPERATION : int
        Operation not supported by this agent.
    CONTENT_TYPE_NOT_SUPPORTED : int
        A part's ``media_type`` is outside the agent's accepted set.
    INVALID_AGENT_RESPONSE : int
        Agent returned content that does not match the expected
        schema.
    EXTENDED_AGENT_CARD_NOT_CONFIGURED : int
        Requested extended card metadata is not exposed.
    EXTENSION_SUPPORT_REQUIRED : int
        Message relies on an unsupported A2A extension.
    VERSION_NOT_SUPPORTED : int
        Client advertised a spec version the server cannot speak.
    """

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
