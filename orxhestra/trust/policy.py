"""Trust policy — declarative rules for event verification.

A :class:`TrustPolicy` decides whether an incoming event is acceptable
based on:

- Whether the signer DID is on an allowlist / denylist.
- Whether signatures are required for particular event types.
- Whether hash-chain continuity is required.
- Whether unrecognised events are rejected (strict mode) or merely
  annotated (permissive mode).

Policies are pure values — they carry no crypto state.  The actual
verification work happens in
:class:`~orxhestra.middleware.trust.TrustMiddleware`, which consults
this policy after resolving the signer's public key.

See Also
--------
orxhestra.middleware.trust.TrustMiddleware : Consumer that enforces
    these rules on_event.
orxhestra.events.event.Event : Target of policy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from orxhestra.events.event import EventType


PolicyMode = Literal["strict", "permissive"]


@dataclass(frozen=True)
class PolicyDecision:
    """Verdict from evaluating an event against a :class:`TrustPolicy`.

    Produced by :meth:`TrustPolicy.check_signer` and by
    :class:`~orxhestra.middleware.trust.TrustMiddleware` during
    :meth:`~orxhestra.middleware.base.Middleware.on_event`.

    Attributes
    ----------
    allow : bool
        Whether the event should be delivered to downstream consumers.
        In permissive mode ``allow`` stays ``True`` even when
        verification fails — the event is passed through with a
        ``metadata["trust"]`` annotation.
    reason : str
        Short human-readable explanation of the decision.  Empty when
        the event passes cleanly.
    verified : bool
        Whether the event's signature was successfully verified.
    """

    allow: bool
    reason: str = ""
    verified: bool = False


@dataclass
class TrustPolicy:
    """Declarative policy evaluated on every event by the trust middleware.

    See :class:`~orxhestra.middleware.trust.TrustMiddleware` for the
    enforcement pipeline that consults this policy.

    Parameters
    ----------
    mode : {"strict", "permissive"}
        ``"strict"`` drops events that fail any rule.  ``"permissive"``
        keeps delivering them but annotates
        ``event.metadata["trust"]`` with the failure reason so
        downstream consumers can flag or filter them.
    trusted_dids : set[str]
        When non-empty, only events whose ``signer_did`` is in this
        set may pass.  An empty set means *any* valid signer is
        acceptable (subject to ``denied_dids``).
    denied_dids : set[str]
        DIDs whose events are always rejected, even if they appear in
        ``trusted_dids``.  Denylist takes precedence.
    require_signed_event_types : set[EventType]
        Event types that must carry a valid signature.  Other event
        types pass through unsigned without complaint.  An empty set
        imposes no signing requirement at all.
    require_chain : bool
        When ``True``, every signed event must link to the previous
        signed event on the same branch via :attr:`Event.prev_signature`.
        Gaps or forks are rejected.  Disable for environments where
        events are re-emitted out of order (e.g. replayed fixtures).
    allow_unsigned : bool
        When ``True``, events with no signature are allowed through
        (subject to ``require_signed_event_types``).  When ``False``
        every event must be signed.

    See Also
    --------
    PolicyDecision : Return type of :meth:`decide`.
    TrustMiddleware : Middleware that applies this policy.
    """

    mode: PolicyMode = "permissive"
    trusted_dids: set[str] = field(default_factory=set)
    denied_dids: set[str] = field(default_factory=set)
    require_signed_event_types: set[EventType] = field(default_factory=set)
    require_chain: bool = False
    allow_unsigned: bool = True

    def check_signer(self, signer_did: str) -> PolicyDecision | None:
        """Check the DID against the allow/deny lists.

        Parameters
        ----------
        signer_did : str
            The ``did:key`` / ``did:web`` identifier of the signer.

        Returns
        -------
        PolicyDecision or None
            ``None`` when the DID is acceptable.  A rejecting
            ``PolicyDecision`` when denied or absent from the
            allowlist.
        """
        if signer_did in self.denied_dids:
            return PolicyDecision(
                allow=self.mode == "permissive",
                reason=f"signer {signer_did} is denylisted",
                verified=False,
            )
        if self.trusted_dids and signer_did not in self.trusted_dids:
            return PolicyDecision(
                allow=self.mode == "permissive",
                reason=f"signer {signer_did} not in allowlist",
                verified=False,
            )
        return None

    def requires_signature(self, event_type: EventType) -> bool:
        """Return ``True`` when events of ``event_type`` must be signed.

        Parameters
        ----------
        event_type : EventType
            The type of the incoming event.

        Returns
        -------
        bool
        """
        if not self.allow_unsigned:
            return True
        return event_type in self.require_signed_event_types
