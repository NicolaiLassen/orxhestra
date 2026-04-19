"""Local filesystem attestation provider — hash-chained JSON log.

A reference implementation of :class:`AttestationProvider` that uses
only stdlib + ``orxhestra[auth]``.  Good for local development,
single-host deployments, and tests.

Layout under ``path``::

    path/
      audit.log         JSON lines, each entry linked by ``prev_hash``
      claims/           one JSON file per claim, keyed by claim id
      revocations.log   JSON lines of revocation records

Every audit entry and claim is signed with the provider's Ed25519 key
so the log is tamper-evident after the fact.  The SHA-256 chain over
``prev_hash`` detects reordering or deletion.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from orxhestra.trust.attestation.protocol import Claim

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from orxhestra.events.event import Event


class LocalAttestationProvider:
    """Append-only, hash-chained, signed audit + claim store on disk.

    Parameters
    ----------
    path : Path or str
        Directory to hold ``audit.log``, ``claims/``, and
        ``revocations.log``.  Created on demand.
    signing_key : Ed25519PrivateKey
        Key used to sign every audit entry and every issued claim.
    issuer_did : str
        The ``did:key`` corresponding to ``signing_key``.  Recorded on
        each claim under :attr:`Claim.issuer_did`.

    See Also
    --------
    orxhestra.trust.attestation.protocol.AttestationProvider : Protocol.
    NoOpAttestationProvider : No-persistence alternative.
    orxhestra.middleware.attestation.AttestationMiddleware : Runtime
        consumer.
    """

    def __init__(
        self,
        path: Path | str,
        signing_key: Ed25519PrivateKey,
        issuer_did: str,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "claims").mkdir(exist_ok=True)
        self._audit_log = self.path / "audit.log"
        self._revocations_log = self.path / "revocations.log"
        self._signing_key = signing_key
        self._issuer_did = issuer_did
        self._lock = asyncio.Lock()

    async def issue_claim(
        self,
        subject_did: str,
        claim_type: str,
        claims: dict[str, Any],
    ) -> Claim:
        """Issue, sign, and persist a :class:`Claim` as a JSON file.

        Parameters
        ----------
        subject_did : str
        claim_type : str
        claims : dict[str, Any]

        Returns
        -------
        Claim
            The issued claim with :attr:`Claim.signature` and
            :attr:`Claim.issuer_did` populated.
        """
        from orxhestra.security.crypto import sign_json_payload

        claim_id = str(uuid4())
        issued_at = time.time()
        payload = {
            "id": claim_id,
            "subject_did": subject_did,
            "type": claim_type,
            "issued_at": issued_at,
            "claims": claims,
            "issuer_did": self._issuer_did,
        }
        signature = sign_json_payload(self._signing_key, payload)
        claim = Claim(
            id=claim_id,
            subject_did=subject_did,
            type=claim_type,
            issued_at=issued_at,
            claims=dict(claims),
            signature=signature,
            issuer_did=self._issuer_did,
        )

        async with self._lock:
            (self.path / "claims" / f"{claim_id}.json").write_text(
                json.dumps({**payload, "signature": signature}, indent=2),
            )
        return claim

    async def verify_claim(self, claim: Claim) -> bool:
        """Verify a claim's Ed25519 signature using the issuer's DID.

        Returns ``False`` on any failure — malformed DID, missing
        signature, or mismatched payload.

        Parameters
        ----------
        claim : Claim

        Returns
        -------
        bool
        """
        if not claim.signature or not claim.issuer_did:
            return False

        from orxhestra.security.crypto import (
            did_key_to_public_key,
            verify_json_signature,
        )

        try:
            public_key = did_key_to_public_key(claim.issuer_did)
        except ValueError:
            return False

        payload = {
            "id": claim.id,
            "subject_did": claim.subject_did,
            "type": claim.type,
            "issued_at": claim.issued_at,
            "claims": claim.claims,
            "issuer_did": claim.issuer_did,
        }
        return verify_json_signature(public_key, payload, claim.signature)

    async def append_audit(self, event: Event) -> None:
        """Append a signed, hash-chained audit entry for ``event``.

        The entry links to the previous one via ``prev_hash`` so any
        middle-of-log tampering invalidates the chain.  Layout on
        disk (one JSON line per entry)::

            {signed_payload..., "signature": ..., "hash": ...}

        where the signature is an Ed25519 signature over
        ``signed_payload`` and ``hash`` is SHA-256 of
        ``signed_payload + signature``.

        Parameters
        ----------
        event : Event
            Event to audit.  Its :meth:`Event.signable_payload` is
            used as the canonical fingerprint.
        """
        from orxhestra.security.crypto import canonicalize_json, sign_json_payload

        async with self._lock:
            prev_hash = self._read_last_hash()
            signed_payload = {
                "prev_hash": prev_hash,
                "event_id": event.id,
                "event_type": event.type.value,
                "agent_name": event.agent_name or "",
                "branch": event.branch,
                "timestamp": event.timestamp,
                "signer_did": event.signer_did,
                "event_fingerprint": event.signable_payload(),
                "issuer_did": self._issuer_did,
            }
            signature = sign_json_payload(self._signing_key, signed_payload)
            entry_hash = hashlib.sha256(
                canonicalize_json({**signed_payload, "signature": signature}),
            ).hexdigest()
            line = json.dumps({**signed_payload, "signature": signature, "hash": entry_hash})
            with self._audit_log.open("a") as fp:
                fp.write(line + "\n")

    async def revoke(self, claim_id: str, reason: str) -> None:
        """Record a revocation and delete the claim file.

        Parameters
        ----------
        claim_id : str
        reason : str
        """
        async with self._lock:
            record = {
                "claim_id": claim_id,
                "reason": reason,
                "revoked_at": time.time(),
                "issuer_did": self._issuer_did,
            }
            with self._revocations_log.open("a") as fp:
                fp.write(json.dumps(record) + "\n")
            claim_file = self.path / "claims" / f"{claim_id}.json"
            if claim_file.exists():
                claim_file.unlink()

    def _read_last_hash(self) -> str:
        """Return the ``hash`` of the last audit entry, or ``""`` when empty."""
        if not self._audit_log.exists():
            return ""
        last_hash = ""
        with self._audit_log.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_hash = entry.get("hash", "") or last_hash
        return last_hash

    def verify_audit_log(self) -> bool:
        """Replay the audit log and confirm every entry chains correctly.

        Synchronous — intended for post-hoc verification / tests.

        Returns
        -------
        bool
            ``True`` when every ``prev_hash`` matches the previous
            entry's ``hash`` and every signature verifies against the
            recorded ``issuer_did``.
        """
        if not self._audit_log.exists():
            return True

        from orxhestra.security.crypto import (
            canonicalize_json,
            did_key_to_public_key,
            verify_json_signature,
        )

        prev_hash = ""
        with self._audit_log.open() as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    return False

                if entry.get("prev_hash", "") != prev_hash:
                    return False

                signature = entry.pop("signature", "")
                claimed_hash = entry.pop("hash", "")
                issuer_did = entry.get("issuer_did", "")

                try:
                    public_key = did_key_to_public_key(issuer_did)
                except ValueError:
                    return False

                if not verify_json_signature(public_key, entry, signature):
                    return False

                recomputed = hashlib.sha256(
                    canonicalize_json({**entry, "signature": signature}),
                ).hexdigest()
                if recomputed != claimed_hash:
                    return False

                prev_hash = claimed_hash
        return True
