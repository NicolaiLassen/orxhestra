"""Tests for :class:`LocalAttestationProvider` + :class:`NoOpAttestationProvider`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.events.event import Event, EventType  # noqa: E402
from orxhestra.models.part import Content  # noqa: E402
from orxhestra.security.crypto import (  # noqa: E402
    generate_ed25519_keypair,
    public_key_to_did_key,
)
from orxhestra.trust import (  # noqa: E402
    Claim,
    LocalAttestationProvider,
    NoOpAttestationProvider,
)


def _fresh_event() -> Event:
    return Event(
        type=EventType.AGENT_MESSAGE,
        agent_name="a",
        branch="root",
        content=Content.from_text("hello"),
    )


def _provider(tmp_path: Path) -> LocalAttestationProvider:
    priv, pub = generate_ed25519_keypair()
    did = public_key_to_did_key(pub)
    return LocalAttestationProvider(tmp_path, priv, did)


@pytest.mark.asyncio
class TestLocalAttestationProvider:
    """Hash-chained audit log and Ed25519-signed claims on disk."""

    async def test_issue_and_verify_claim_round_trip(self, tmp_path: Path) -> None:
        provider = _provider(tmp_path)
        claim = await provider.issue_claim(
            subject_did="did:key:zSubject",
            claim_type="tool.invoke",
            claims={"tool_name": "shell"},
        )
        assert isinstance(claim, Claim)
        assert claim.signature
        assert claim.issuer_did.startswith("did:key:z")
        assert await provider.verify_claim(claim)

    async def test_verify_claim_rejects_tampered_payload(self, tmp_path: Path) -> None:
        provider = _provider(tmp_path)
        claim = await provider.issue_claim(
            subject_did="did:key:zSubject",
            claim_type="tool.invoke",
            claims={"tool_name": "shell"},
        )
        claim.claims["tool_name"] = "rm-rf"
        assert not await provider.verify_claim(claim)

    async def test_audit_log_chain_detects_tampering(self, tmp_path: Path) -> None:
        provider = _provider(tmp_path)
        await provider.append_audit(_fresh_event())
        await provider.append_audit(_fresh_event())
        await provider.append_audit(_fresh_event())
        assert provider.verify_audit_log()

        # Tamper with the second line.
        log_path = tmp_path / "audit.log"
        lines = log_path.read_text().splitlines()
        parsed = json.loads(lines[1])
        parsed["agent_name"] = "attacker"
        lines[1] = json.dumps(parsed)
        log_path.write_text("\n".join(lines) + "\n")

        assert not provider.verify_audit_log()

    async def test_revoke_removes_claim_file(self, tmp_path: Path) -> None:
        provider = _provider(tmp_path)
        claim = await provider.issue_claim(
            subject_did="did:key:zSubject",
            claim_type="tool.invoke",
            claims={"tool_name": "shell"},
        )
        claim_file = tmp_path / "claims" / f"{claim.id}.json"
        assert claim_file.exists()

        await provider.revoke(claim.id, "policy-violation")
        assert not claim_file.exists()
        revocations = (tmp_path / "revocations.log").read_text().strip().splitlines()
        assert len(revocations) == 1
        assert json.loads(revocations[0])["reason"] == "policy-violation"


@pytest.mark.asyncio
class TestNoOpAttestationProvider:
    """The default provider passes through without touching disk."""

    async def test_issue_claim_returns_empty_signature(self) -> None:
        provider = NoOpAttestationProvider()
        claim = await provider.issue_claim(
            subject_did="did:key:zSubject",
            claim_type="tool.invoke",
            claims={"tool_name": "shell"},
        )
        assert claim.signature == ""
        assert claim.subject_did == "did:key:zSubject"

    async def test_verify_claim_accepts_everything(self) -> None:
        provider = NoOpAttestationProvider()
        bogus = Claim(
            id="x",
            subject_did="did:key:z",
            type="anything",
            issued_at=0.0,
        )
        assert await provider.verify_claim(bogus)

    async def test_append_audit_is_noop(self) -> None:
        provider = NoOpAttestationProvider()
        await provider.append_audit(_fresh_event())
