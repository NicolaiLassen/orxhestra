"""Detached Ed25519 signatures for A2A v1.0 messages.

The signature lives under ``message.metadata[SIGNATURE_KEY]`` so an
unsigned peer can strip it (or ignore it entirely) without breaking
the underlying A2A v1.0 wire format.  The signed payload covers every
field that matters for identity and content:

- ``message_id`` — stable identity
- ``role`` — who is speaking
- ``parts_hash`` — SHA-256 of the canonical JSON of the parts list
- ``context_id`` / ``task_id`` / ``reference_task_ids`` — conversation
- ``extensions`` — protocol add-ons in use
- ``timestamp`` — moment of signing (replay protection)
- ``signer_did`` — key the verifier should resolve

Routines here are pure functions — they do not mutate the input
message.  :func:`sign_message` returns a modified copy.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from orxhestra.a2a.types import Message, Role

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from orxhestra.security.did import DidResolver


SIGNATURE_KEY = "orxSignature"
"""Metadata key under which the base64url Ed25519 signature is stored."""

SIGNER_DID_KEY = "orxSignerDid"
"""Metadata key under which the signer DID is stored."""

TIMESTAMP_KEY = "orxSignedAt"
"""Metadata key under which the signing timestamp (epoch seconds) lives."""


def message_signable_payload(
    message: Message,
    *,
    signer_did: str,
    timestamp: float,
) -> dict[str, Any]:
    """Return the canonical dict that gets signed for an A2A Message.

    Parameters
    ----------
    message : Message
        The message to fingerprint.  Not mutated.
    signer_did : str
        DID bound into the payload so stealing the signature and
        attaching a different DID is detectable.
    timestamp : float
        Unix seconds at the moment of signing.

    Returns
    -------
    dict[str, Any]
        Deterministic signable dictionary suitable for
        :func:`orxhestra.security.crypto.sign_json_payload`.
    """
    from orxhestra.security.crypto import canonicalize_json

    parts_payload = [
        p.model_dump(by_alias=True, exclude_none=True) for p in message.parts
    ]
    parts_hash = hashlib.sha256(canonicalize_json(parts_payload)).hexdigest()
    role_value = message.role.value if isinstance(message.role, Role) else str(message.role)

    return {
        "message_id": message.message_id,
        "role": role_value,
        "parts_hash": parts_hash,
        "context_id": message.context_id or "",
        "task_id": message.task_id or "",
        "reference_task_ids": message.reference_task_ids or [],
        "extensions": message.extensions or [],
        "signer_did": signer_did,
        "timestamp": timestamp,
    }


def sign_message(
    message: Message,
    signing_key: Ed25519PrivateKey,
    signer_did: str,
    *,
    timestamp: float | None = None,
) -> Message:
    """Return a copy of ``message`` with a detached Ed25519 signature attached.

    Parameters
    ----------
    message : Message
        The message to sign.  Not mutated.
    signing_key : Ed25519PrivateKey
        Private key corresponding to ``signer_did``.
    signer_did : str
        DID identifying the signer — written into the metadata so the
        verifier knows which key to resolve.
    timestamp : float, optional
        Unix seconds stamped into the payload.  Defaults to now.

    Returns
    -------
    Message
        A model copy carrying the signature metadata.

    Raises
    ------
    ImportError
        If ``orxhestra[auth]`` is not installed.
    """
    from orxhestra.security.crypto import sign_json_payload

    ts = timestamp if timestamp is not None else time.time()
    payload = message_signable_payload(message, signer_did=signer_did, timestamp=ts)
    signature = sign_json_payload(signing_key, payload)

    existing = dict(message.metadata or {})
    existing[SIGNATURE_KEY] = signature
    existing[SIGNER_DID_KEY] = signer_did
    existing[TIMESTAMP_KEY] = ts
    return message.model_copy(update={"metadata": existing})


def extract_signature(message: Message) -> tuple[str, str, float] | None:
    """Return ``(signature, signer_did, timestamp)`` or ``None`` when unsigned.

    Parameters
    ----------
    message : Message

    Returns
    -------
    tuple[str, str, float] or None
    """
    metadata = message.metadata or {}
    signature = metadata.get(SIGNATURE_KEY)
    signer_did = metadata.get(SIGNER_DID_KEY)
    timestamp = metadata.get(TIMESTAMP_KEY)
    if not signature or not signer_did or timestamp is None:
        return None
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return None
    return signature, signer_did, ts


async def verify_message(
    message: Message,
    resolver: DidResolver,
) -> bool:
    """Return ``True`` when ``message`` carries a valid detached signature.

    Parameters
    ----------
    message : Message
        The message to verify.
    resolver : DidResolver
        Resolver used to turn the signer DID into a public key.

    Returns
    -------
    bool
        ``False`` when the message is unsigned, the DID cannot be
        resolved, or the signature does not match the payload.
    """
    extracted = extract_signature(message)
    if extracted is None:
        return False

    signature, signer_did, ts = extracted

    try:
        public_key = await resolver.resolve(signer_did)
    except (ValueError, LookupError, ImportError):
        return False

    from orxhestra.security.crypto import verify_json_signature

    payload = message_signable_payload(message, signer_did=signer_did, timestamp=ts)
    return verify_json_signature(public_key, payload, signature)
