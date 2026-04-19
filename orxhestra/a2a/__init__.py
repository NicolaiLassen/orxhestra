"""A2A (Agent-to-Agent) protocol v1.0 support.

Provides spec-compliant wire types (:class:`Message`, :class:`Task`,
:class:`AgentCard`, ...), a FastAPI-based :class:`A2AServer` that
adapts any :class:`BaseAgent` to a JSON-RPC endpoint, and an
A2A-to-SDK event-stream converter (:func:`events_to_a2a_stream`).

Optional identity layer (requires ``orxhestra[auth]``):

- :class:`VerificationMethod` advertised on :class:`AgentCard` so
  peers can resolve the server's public key.
- :mod:`orxhestra.a2a.signing` — detached Ed25519 signatures on
  :class:`Message` envelopes via metadata-level
  ``orxSignature`` / ``orxSignerDid`` fields.  Signing is strictly
  opt-in — unsigned peers keep working unless the server or client
  sets ``require_signed``.

Requires the ``a2a`` extra: ``pip install orxhestra[a2a]``.

See Also
--------
orxhestra.a2a.server.A2AServer : Server-side adapter.
orxhestra.agents.a2a_agent.A2AAgent : Client-side adapter.
orxhestra.a2a.signing : Message signing / verification helpers.
orxhestra.security.did : DID resolvers used to verify remote peers.
"""

from orxhestra.a2a.signing import (
    SIGNATURE_KEY,
    SIGNER_DID_KEY,
    TIMESTAMP_KEY,
    extract_signature,
    message_signable_payload,
    sign_message,
    verify_message,
)
from orxhestra.a2a.types import (
    A2AModel,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Artifact,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageConfiguration,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    VerificationMethod,
    data_part,
    file_part,
    text_part,
)


def __getattr__(name: str):
    """Lazy-load :class:`A2AServer` and :func:`events_to_a2a_stream`.

    Keeps the import graph light so users who only need the wire
    types don't pay for FastAPI.  The converter lives in
    :mod:`orxhestra.a2a.converters` and is imported the same way for
    symmetry.

    Parameters
    ----------
    name : str
        Attribute requested from this module.

    Returns
    -------
    Any
        The imported symbol.

    Raises
    ------
    AttributeError
        If ``name`` is not a known lazy symbol.
    """
    if name == "A2AServer":
        from orxhestra.a2a.server import A2AServer
        return A2AServer
    if name == "events_to_a2a_stream":
        from orxhestra.a2a.converters import events_to_a2a_stream
        return events_to_a2a_stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "A2AModel",
    "A2AServer",
    "AgentCapabilities",
    "AgentCard",
    "AgentInterface",
    "AgentProvider",
    "AgentSkill",
    "Artifact",
    "JSONRPCError",
    "JSONRPCRequest",
    "JSONRPCResponse",
    "Message",
    "MessageSendParams",
    "Part",
    "Role",
    "SIGNATURE_KEY",
    "SIGNER_DID_KEY",
    "SendMessageConfiguration",
    "TIMESTAMP_KEY",
    "Task",
    "TaskArtifactUpdateEvent",
    "TaskState",
    "TaskStatus",
    "TaskStatusUpdateEvent",
    "VerificationMethod",
    "data_part",
    "events_to_a2a_stream",
    "extract_signature",
    "file_part",
    "message_signable_payload",
    "sign_message",
    "text_part",
    "verify_message",
]
