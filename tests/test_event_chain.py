"""Tests for Event hash-chain and extended signable payload coverage."""

from __future__ import annotations

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.agents.base_agent import BaseAgent  # noqa: E402
from orxhestra.agents.invocation_context import InvocationContext  # noqa: E402
from orxhestra.events.event import EventType  # noqa: E402
from orxhestra.events.event_actions import EventActions  # noqa: E402
from orxhestra.models.part import Content, ToolCallPart  # noqa: E402
from orxhestra.security.crypto import (  # noqa: E402
    generate_ed25519_keypair,
    public_key_to_did_key,
)


def _signed_ctx() -> InvocationContext:
    priv, pub = generate_ed25519_keypair()
    did = public_key_to_did_key(pub)
    return InvocationContext(
        session_id="s",
        agent_name="agent",
        branch="root",
        signing_key=priv,
        signing_did=did,
    )


class _Agent(BaseAgent):
    async def astream(self, *_a, **_kw):  # type: ignore[override]
        yield  # pragma: no cover


class TestHashChain:
    """Every emitted event links to the previous one on the same branch."""

    def test_first_event_has_no_prev_signature(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("one"),
        )
        assert event.prev_signature is None
        assert event.is_signed
        assert event.verify_signature()

    def test_subsequent_events_chain_by_signature(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        first = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("one"),
        )
        second = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("two"),
        )
        assert second.prev_signature == first.signature
        assert first.verify_signature()
        assert second.verify_signature()

    def test_partial_events_do_not_advance_chain_head(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        first = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("one"),
        )
        streaming_chunk = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("partial"),
            partial=True,
        )
        second_final = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("two"),
        )
        # The partial sees the current head...
        assert streaming_chunk.prev_signature == first.signature
        # ...but does not move it, so the next non-partial still points at first.
        assert second_final.prev_signature == first.signature

    def test_branches_track_chain_heads_independently(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()

        parent = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("parent"),
        )
        child_ctx = ctx.derive(agent_name="child")
        child_event = agent._emit_event(
            child_ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("child"),
        )
        # Different branch → starts its own chain.
        assert child_event.prev_signature is None
        back_on_parent = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("second"),
        )
        assert back_on_parent.prev_signature == parent.signature


class TestExpandedSigningCoverage:
    """signable_payload covers tool calls, actions, and llm_response."""

    def test_tampered_tool_call_breaks_signature(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content(parts=[
                ToolCallPart(
                    tool_call_id="t1", tool_name="shell", args={"cmd": "ls"},
                ),
            ]),
        )
        assert event.verify_signature()
        # Rewrite the tool call args — verification must fail.
        event.content.tool_calls[0].args = {"cmd": "rm -rf /"}
        assert not event.verify_signature()

    def test_tampered_transfer_action_breaks_signature(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("hello"),
            actions=EventActions(transfer_to_agent="legitimate"),
        )
        assert event.verify_signature()
        event.actions.transfer_to_agent = "attacker"
        assert not event.verify_signature()

    def test_tampered_state_delta_breaks_signature(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("noop"),
            actions=EventActions(state_delta={"role": "user"}),
        )
        assert event.verify_signature()
        event.actions.state_delta["role"] = "admin"
        assert not event.verify_signature()

    def test_empty_event_still_deterministic(self) -> None:
        agent = _Agent(name="a")
        ctx = _signed_ctx()
        event = agent._emit_event(
            ctx, EventType.AGENT_START,
        )
        assert event.signable_payload() == event.signable_payload()

    def test_backward_compat_unsigned_events(self) -> None:
        """Events without a signing key still behave as before."""
        agent = _Agent(name="a")
        ctx = InvocationContext(session_id="s", agent_name="a")
        event = agent._emit_event(
            ctx, EventType.AGENT_MESSAGE, content=Content.from_text("hi"),
        )
        assert not event.is_signed
        assert event.prev_signature is None
