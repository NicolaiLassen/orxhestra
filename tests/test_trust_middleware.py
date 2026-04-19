"""Tests for :class:`TrustMiddleware` + :class:`TrustPolicy`."""

from __future__ import annotations

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.agents.base_agent import BaseAgent  # noqa: E402
from orxhestra.agents.invocation_context import InvocationContext  # noqa: E402
from orxhestra.events.event import Event, EventType  # noqa: E402
from orxhestra.middleware import TrustMiddleware  # noqa: E402
from orxhestra.models.part import Content  # noqa: E402
from orxhestra.security.crypto import (  # noqa: E402
    generate_ed25519_keypair,
    public_key_to_did_key,
)
from orxhestra.security.did import DidKeyResolver  # noqa: E402
from orxhestra.trust import PolicyDecision, TrustPolicy  # noqa: E402


class _A(BaseAgent):
    async def astream(self, *_a, **_kw):  # type: ignore[override]
        yield  # pragma: no cover


def _signed_event(ctx_did: str, ctx_key, text: str = "hello") -> tuple[InvocationContext, Event]:
    ctx = InvocationContext(
        session_id="s", agent_name="a", branch="root",
        signing_key=ctx_key, signing_did=ctx_did,
    )
    agent = _A(name="a")
    event = agent._emit_event(
        ctx, EventType.AGENT_MESSAGE, content=Content.from_text(text),
    )
    return ctx, event


@pytest.mark.asyncio
class TestTrustMiddleware:
    """End-to-end on_event semantics."""

    async def test_valid_signature_passes_strict(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        ctx, event = _signed_event(did, priv)

        mw = TrustMiddleware(TrustPolicy(mode="strict"), DidKeyResolver())
        result = await mw.on_event(ctx, event)
        assert result is event
        assert event.metadata["trust"]["verified"] is True

    async def test_tampered_event_dropped_in_strict_mode(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        ctx, event = _signed_event(did, priv)
        event.content = Content.from_text("tampered")

        mw = TrustMiddleware(TrustPolicy(mode="strict"), DidKeyResolver())
        result = await mw.on_event(ctx, event)
        assert result is None

    async def test_tampered_event_annotated_in_permissive_mode(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        ctx, event = _signed_event(did, priv)
        event.content = Content.from_text("tampered")

        mw = TrustMiddleware(TrustPolicy(mode="permissive"), DidKeyResolver())
        result = await mw.on_event(ctx, event)
        assert result is event
        assert event.metadata["trust"]["verified"] is False
        assert "signature" in event.metadata["trust"]["reason"]

    async def test_denylist_takes_precedence(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        ctx, event = _signed_event(did, priv)

        policy = TrustPolicy(
            mode="strict",
            trusted_dids={did},
            denied_dids={did},
        )
        mw = TrustMiddleware(policy, DidKeyResolver())
        assert await mw.on_event(ctx, event) is None

    async def test_allowlist_filters_strangers(self) -> None:
        priv1, pub1 = generate_ed25519_keypair()
        priv2, pub2 = generate_ed25519_keypair()
        trusted = public_key_to_did_key(pub1)
        other = public_key_to_did_key(pub2)
        ctx, event = _signed_event(other, priv2)

        policy = TrustPolicy(mode="strict", trusted_dids={trusted})
        mw = TrustMiddleware(policy, DidKeyResolver())
        assert await mw.on_event(ctx, event) is None

    async def test_unsigned_event_allowed_by_default(self) -> None:
        ctx = InvocationContext(session_id="s", agent_name="a", branch="root")
        event = Event(type=EventType.USER_MESSAGE, content=Content.from_text("hi"))

        mw = TrustMiddleware(TrustPolicy(mode="strict"), DidKeyResolver())
        result = await mw.on_event(ctx, event)
        assert result is event

    async def test_unsigned_event_dropped_when_required(self) -> None:
        ctx = InvocationContext(session_id="s", agent_name="a", branch="root")
        event = Event(
            type=EventType.AGENT_MESSAGE, content=Content.from_text("hi"),
        )

        policy = TrustPolicy(
            mode="strict",
            require_signed_event_types={EventType.AGENT_MESSAGE},
            allow_unsigned=True,
        )
        mw = TrustMiddleware(policy, DidKeyResolver())
        assert await mw.on_event(ctx, event) is None

    async def test_require_chain_rejects_gap(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        ctx, first = _signed_event(did, priv, "one")
        second_agent = _A(name="a")
        # Emit a second event on the same branch. Then reset prev_signature
        # to simulate a gap / reordering attack.
        second = second_agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("two"),
        )
        mw = TrustMiddleware(
            TrustPolicy(mode="strict", require_chain=True),
            DidKeyResolver(),
        )
        # First one flows through and sets the chain head.
        assert await mw.on_event(ctx, first) is first
        # Forge prev_signature and re-sign so the signature still validates
        # but chain continuity is broken.
        from orxhestra.security.crypto import sign_json_payload

        second.prev_signature = "deadbeef"
        second.signature = sign_json_payload(priv, second.signable_payload())
        result = await mw.on_event(ctx, second)
        assert result is None


class TestPolicyHelpers:
    """Unit coverage for :class:`TrustPolicy` without middleware."""

    def test_check_signer_allowlist(self) -> None:
        policy = TrustPolicy(trusted_dids={"did:key:z1"})
        assert policy.check_signer("did:key:z1") is None
        rejected = policy.check_signer("did:key:z2")
        assert isinstance(rejected, PolicyDecision)
        assert not rejected.verified

    def test_check_signer_denylist(self) -> None:
        policy = TrustPolicy(denied_dids={"did:key:z2"})
        assert policy.check_signer("did:key:z1") is None
        rejected = policy.check_signer("did:key:z2")
        assert rejected is not None
        assert "denylisted" in rejected.reason

    def test_requires_signature(self) -> None:
        policy = TrustPolicy(
            require_signed_event_types={EventType.AGENT_MESSAGE},
        )
        assert policy.requires_signature(EventType.AGENT_MESSAGE)
        assert not policy.requires_signature(EventType.USER_MESSAGE)

    def test_allow_unsigned_false_requires_every_type(self) -> None:
        policy = TrustPolicy(allow_unsigned=False)
        assert policy.requires_signature(EventType.USER_MESSAGE)
