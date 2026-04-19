"""Tests for :class:`AttestationMiddleware`."""

from __future__ import annotations

from typing import Any

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.agents.invocation_context import InvocationContext  # noqa: E402
from orxhestra.events.event import Event, EventType  # noqa: E402
from orxhestra.events.event_actions import EventActions  # noqa: E402
from orxhestra.middleware import AttestationMiddleware  # noqa: E402
from orxhestra.models.part import Content, ToolCallPart  # noqa: E402
from orxhestra.trust import Claim  # noqa: E402


class _RecorderProvider:
    """Capture provider calls for assertions."""

    def __init__(self) -> None:
        self.audited: list[Event] = []
        self.claims: list[tuple[str, str, dict[str, Any]]] = []
        self.revoked: list[tuple[str, str]] = []

    async def issue_claim(
        self, subject_did: str, claim_type: str, claims: dict[str, Any],
    ) -> Claim:
        self.claims.append((subject_did, claim_type, dict(claims)))
        return Claim(
            id=f"c{len(self.claims)}",
            subject_did=subject_did,
            type=claim_type,
            issued_at=0.0,
            claims=dict(claims),
        )

    async def verify_claim(self, claim: Claim) -> bool:
        return True

    async def append_audit(self, event: Event) -> None:
        self.audited.append(event)

    async def revoke(self, claim_id: str, reason: str) -> None:
        self.revoked.append((claim_id, reason))


def _ctx() -> InvocationContext:
    return InvocationContext(session_id="s", agent_name="a", branch="root")


@pytest.mark.asyncio
class TestAttestationMiddleware:
    """Audit + claim issuance on_event."""

    async def test_every_event_is_audited(self) -> None:
        provider = _RecorderProvider()
        mw = AttestationMiddleware(provider)
        event = Event(
            type=EventType.AGENT_MESSAGE, content=Content.from_text("hi"),
        )
        returned = await mw.on_event(_ctx(), event)
        assert returned is event
        assert provider.audited == [event]
        assert provider.claims == []

    async def test_transfer_action_issues_claim(self) -> None:
        provider = _RecorderProvider()
        mw = AttestationMiddleware(provider)
        event = Event(
            type=EventType.AGENT_MESSAGE,
            agent_name="router",
            content=Content.from_text("hand off"),
            actions=EventActions(transfer_to_agent="researcher"),
        )
        await mw.on_event(_ctx(), event)
        transfer_claims = [c for c in provider.claims if c[1] == "agent.transfer"]
        assert len(transfer_claims) == 1
        _, _, claim_payload = transfer_claims[0]
        assert claim_payload["to_agent"] == "researcher"
        assert claim_payload["from_agent"] == "router"

    async def test_tool_calls_issue_claims(self) -> None:
        provider = _RecorderProvider()
        mw = AttestationMiddleware(provider)
        event = Event(
            type=EventType.AGENT_MESSAGE,
            content=Content(parts=[
                ToolCallPart(tool_call_id="t1", tool_name="shell", args={"cmd": "ls"}),
                ToolCallPart(tool_call_id="t2", tool_name="grep", args={"q": "x"}),
            ]),
        )
        await mw.on_event(_ctx(), event)
        tool_claims = [c for c in provider.claims if c[1] == "tool.invoke"]
        assert len(tool_claims) == 2
        names = sorted(c[2]["tool_name"] for c in tool_claims)
        assert names == ["grep", "shell"]

    async def test_partial_events_skip_claim_issuance(self) -> None:
        provider = _RecorderProvider()
        mw = AttestationMiddleware(provider)
        event = Event(
            type=EventType.AGENT_MESSAGE,
            partial=True,
            content=Content(parts=[
                ToolCallPart(tool_call_id="t1", tool_name="shell"),
            ]),
        )
        await mw.on_event(_ctx(), event)
        assert provider.audited == [event]
        assert provider.claims == []

    async def test_provider_exception_does_not_propagate(self) -> None:
        class _Broken:
            async def issue_claim(self, *_a, **_kw):
                raise RuntimeError("boom")

            async def verify_claim(self, *_a, **_kw):
                return False

            async def append_audit(self, *_a, **_kw):
                raise RuntimeError("boom")

            async def revoke(self, *_a, **_kw):
                return None

        mw = AttestationMiddleware(_Broken())
        event = Event(
            type=EventType.AGENT_MESSAGE,
            content=Content(parts=[
                ToolCallPart(tool_call_id="t1", tool_name="shell"),
            ]),
        )
        returned = await mw.on_event(_ctx(), event)
        assert returned is event  # never drops the event.
