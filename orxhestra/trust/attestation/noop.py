"""No-op attestation provider — records nothing, returns empty claims.

Useful as the default when
:class:`~orxhestra.middleware.attestation.AttestationMiddleware` is
registered but the user hasn't wired up a real backend yet.  Also a
convenient stand-in for tests that exercise the middleware path
without asserting on audit behavior.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from orxhestra.trust.attestation.protocol import Claim

if TYPE_CHECKING:
    from orxhestra.events.event import Event


class NoOpAttestationProvider:
    """A provider that satisfies the :class:`AttestationProvider` protocol.

    Every method returns a trivial successful result without touching
    external state.  Ideal as the default when attestation is required
    by configuration but no backend has been selected.

    See Also
    --------
    orxhestra.trust.attestation.protocol.AttestationProvider : Protocol implemented here.
    LocalAttestationProvider : Reference implementation with persistence.
    """

    async def issue_claim(
        self,
        subject_did: str,
        claim_type: str,
        claims: dict[str, Any],
    ) -> Claim:
        """Return a :class:`Claim` with a fresh id and no signature."""
        return Claim(
            id=str(uuid4()),
            subject_did=subject_did,
            type=claim_type,
            issued_at=time.time(),
            claims=dict(claims),
            signature="",
        )

    async def verify_claim(self, claim: Claim) -> bool:
        """Accept every claim.  Returns ``True`` unconditionally."""
        return True

    async def append_audit(self, event: Event) -> None:
        """Discard ``event`` without persisting anything."""
        return None

    async def revoke(self, claim_id: str, reason: str) -> None:
        """No-op revocation."""
        return None
