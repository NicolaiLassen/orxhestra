"""Tests for signed events — Ed25519 signatures on Event objects."""

from __future__ import annotations

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.agents.base_agent import BaseAgent  # noqa: E402
from orxhestra.agents.invocation_context import InvocationContext  # noqa: E402
from orxhestra.auth.crypto import (  # noqa: E402
    generate_ed25519_keypair,
    public_key_to_did_key,
)
from orxhestra.events.event import Event, EventType  # noqa: E402
from orxhestra.models.part import Content  # noqa: E402


def _signed_ctx(**kwargs) -> InvocationContext:
    """Create a context with a fresh signing key."""
    priv, pub = generate_ed25519_keypair()
    did = public_key_to_did_key(pub)
    defaults = {
        "session_id": "test",
        "agent_name": "agent",
        "signing_key": priv,
        "signing_did": did,
    }
    defaults.update(kwargs)
    return InvocationContext(**defaults)


def _unsigned_ctx(**kwargs) -> InvocationContext:
    """Create a context without a signing key."""
    defaults = {"session_id": "test", "agent_name": "agent"}
    defaults.update(kwargs)
    return InvocationContext(**defaults)


class _DummyAgent(BaseAgent):
    """Minimal agent for testing _emit_event."""

    async def astream(self, input, config=None, *, ctx=None):
        yield  # pragma: no cover


class TestSignedEvents:
    """Tests for automatic event signing via _emit_event."""

    def test_event_signed_when_key_present(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("hello"),
        )
        assert event.is_signed
        assert event.signature is not None
        assert event.signer_did.startswith("did:key:z")

    def test_event_unsigned_when_no_key(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _unsigned_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("hello"),
        )
        assert not event.is_signed
        assert event.signature is None
        assert event.signer_did == ""

    def test_signed_event_verifies(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("trusted output"),
        )
        assert event.verify_signature()

    def test_tampered_event_fails_verification(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("original"),
        )
        event.content = Content.from_text("tampered")
        assert not event.verify_signature()

    def test_wrong_did_fails_verification(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("hello"),
        )
        _, other_pub = generate_ed25519_keypair()
        event.signer_did = public_key_to_did_key(other_pub)
        assert not event.verify_signature()

    def test_unsigned_event_verify_returns_false(self) -> None:
        event = Event(
            type=EventType.AGENT_MESSAGE,
            content=Content.from_text("no sig"),
        )
        assert not event.verify_signature()

    def test_signing_key_propagates_through_derive(self) -> None:
        parent_ctx = _signed_ctx()
        child_ctx = parent_ctx.derive(agent_name="child")
        assert child_ctx.signing_key is parent_ctx.signing_key
        assert child_ctx.signing_did == parent_ctx.signing_did

    def test_tool_response_event_signed(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.TOOL_RESPONSE,
            content=Content.from_text("tool result"),
        )
        assert event.is_signed
        assert event.verify_signature()

    def test_signable_payload_deterministic(self) -> None:
        agent = _DummyAgent(name="agent")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("deterministic"),
        )
        p1 = event.signable_payload()
        p2 = event.signable_payload()
        assert p1 == p2
