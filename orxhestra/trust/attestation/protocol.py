"""Attestation provider protocol and :class:`Claim` data model.

The protocol is intentionally small and provider-agnostic so that
external attestation services (W3C Verifiable Credentials backends,
blockchain anchoring systems, enterprise audit stores, compliance
frameworks) can implement it in their own packages without any
coupling to orxhestra internals.

See Also
--------
NoOpAttestationProvider : Default implementation that records nothing.
LocalAttestationProvider : JSON-on-disk hash-chained audit log.
orxhestra.middleware.attestation.AttestationMiddleware : Runtime
    consumer that drives providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from orxhestra.events.event import Event


@dataclass
class Claim:
    """A signed assertion issued by an :class:`AttestationProvider`.

    Claims are the provider's receipt for an event or action — a
    stable identifier the verifier can later present to prove that a
    particular agent did a particular thing at a particular time.

    Attributes
    ----------
    id : str
        Provider-assigned unique identifier.
    subject_did : str
        The DID of the agent the claim is about.
    type : str
        Dotted type name — e.g. ``"tool.invoke"``,
        ``"agent.transfer"``, or ``"event.audit"``.  Providers should
        document the set of types they recognise.
    issued_at : float
        Unix timestamp when the claim was issued.
    claims : dict[str, Any]
        Provider-specific claim payload.  JSON-serializable.
    signature : str
        Detached signature or provider-specific receipt.  Opaque to
        orxhestra — the provider owns the format.
    issuer_did : str, optional
        DID of the issuing authority, when different from
        ``subject_did``.  Omitted for self-issued claims.
    """

    id: str
    subject_did: str
    type: str
    issued_at: float
    claims: dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    issuer_did: str = ""


@runtime_checkable
class AttestationProvider(Protocol):
    """Pluggable attestation backend.

    Every method is async so adapters can speak to remote services
    (HTTPS endpoints, blockchain RPCs, queue-backed audit pipelines)
    without blocking the agent loop.  Synchronous implementations can
    still be wrapped in plain ``async def`` methods.

    See Also
    --------
    NoOpAttestationProvider : Default.
    LocalAttestationProvider : Bundled reference implementation.
    AttestationMiddleware : Middleware that drives a provider.
    """

    async def issue_claim(
        self,
        subject_did: str,
        claim_type: str,
        claims: dict[str, Any],
    ) -> Claim:
        """Issue and sign a new claim.

        Parameters
        ----------
        subject_did : str
            DID of the agent the claim is about.
        claim_type : str
            Dotted claim type (e.g. ``"tool.invoke"``).
        claims : dict[str, Any]
            JSON-serializable payload.

        Returns
        -------
        Claim
        """
        ...

    async def verify_claim(self, claim: Claim) -> bool:
        """Verify a previously issued claim.

        Parameters
        ----------
        claim : Claim
            The claim to verify.

        Returns
        -------
        bool
            ``True`` when the signature / receipt is valid.
        """
        ...

    async def append_audit(self, event: Event) -> None:
        """Append an :class:`Event` to the audit log.

        Parameters
        ----------
        event : Event
            Any event — signed or unsigned.  Providers typically
            persist a fingerprint rather than the full payload.
        """
        ...

    async def revoke(self, claim_id: str, reason: str) -> None:
        """Revoke a previously issued claim.

        Parameters
        ----------
        claim_id : str
            Identifier returned by :meth:`issue_claim`.
        reason : str
            Short human-readable justification stored with the
            revocation record.
        """
        ...
