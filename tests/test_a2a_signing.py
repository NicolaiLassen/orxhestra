"""Tests for A2A v1.0 message signing / verification helpers."""

from __future__ import annotations

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.a2a.signing import (  # noqa: E402
    SIGNATURE_KEY,
    SIGNER_DID_KEY,
    TIMESTAMP_KEY,
    extract_signature,
    message_signable_payload,
    sign_message,
    verify_message,
)
from orxhestra.a2a.types import Message, Part, Role, VerificationMethod  # noqa: E402
from orxhestra.security.crypto import (  # noqa: E402
    generate_ed25519_keypair,
    public_key_to_did_key,
)
from orxhestra.security.did import DidKeyResolver  # noqa: E402


def _message(text: str = "hello") -> Message:
    return Message(
        role=Role.USER,
        parts=[Part(text=text, media_type="text/plain")],
        context_id="ctx-1",
        task_id="task-1",
    )


@pytest.mark.asyncio
class TestMessageSigning:
    """End-to-end sign → verify round-trip."""

    async def test_round_trip(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        signed = sign_message(_message(), priv, did)
        assert await verify_message(signed, DidKeyResolver())

    async def test_tampered_text_fails_verification(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        signed = sign_message(_message("original"), priv, did)
        # Edit the underlying part — parts_hash changes.
        signed.parts[0].text = "modified"
        assert not await verify_message(signed, DidKeyResolver())

    async def test_forged_did_fails_verification(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        signed = sign_message(_message(), priv, did)
        other_priv, other_pub = generate_ed25519_keypair()
        other_did = public_key_to_did_key(other_pub)
        # Swap signer DID — verifier resolves the *other* key, signature mismatches.
        signed.metadata = dict(signed.metadata or {})
        signed.metadata[SIGNER_DID_KEY] = other_did
        assert not await verify_message(signed, DidKeyResolver())

    async def test_unsigned_message_fails_verification(self) -> None:
        msg = _message()
        assert await verify_message(msg, DidKeyResolver()) is False


class TestSignableHelpers:
    """Deterministic shape of the signable payload."""

    def test_payload_is_stable(self) -> None:
        msg = _message()
        p1 = message_signable_payload(msg, signer_did="did:key:z1", timestamp=1.0)
        p2 = message_signable_payload(msg, signer_did="did:key:z1", timestamp=1.0)
        assert p1 == p2

    def test_payload_changes_with_timestamp(self) -> None:
        msg = _message()
        p1 = message_signable_payload(msg, signer_did="did:key:z1", timestamp=1.0)
        p2 = message_signable_payload(msg, signer_did="did:key:z1", timestamp=2.0)
        assert p1 != p2

    def test_extract_signature_returns_none_when_unsigned(self) -> None:
        assert extract_signature(_message()) is None

    def test_extract_signature_round_trip(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        signed = sign_message(_message(), priv, did)
        extracted = extract_signature(signed)
        assert extracted is not None
        sig, extracted_did, ts = extracted
        assert signed.metadata[SIGNATURE_KEY] == sig
        assert extracted_did == did
        assert signed.metadata[TIMESTAMP_KEY] == ts


class TestAgentCardVerificationMethod:
    """Agent cards can publish :class:`VerificationMethod` entries."""

    def test_serializes_as_camelcase(self) -> None:
        vm = VerificationMethod(
            id="did:key:z1#key-1",
            controller="did:key:z1",
            public_key_multibase="z6Mk...",
        )
        dumped = vm.model_dump(by_alias=True, exclude_none=True)
        assert dumped["publicKeyMultibase"] == "z6Mk..."
        assert dumped["type"] == "Ed25519VerificationKey2020"
