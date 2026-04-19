"""Signature-verification middleware.

:class:`TrustMiddleware` hooks into
:meth:`~orxhestra.middleware.base.Middleware.on_event` to:

1. Check the event type against
   :attr:`~orxhestra.trust.policy.TrustPolicy.require_signed_event_types`.
2. Resolve the signer's public key via a
   :class:`~orxhestra.security.did.DidResolver`.
3. Verify the event's Ed25519 signature over its canonical payload.
4. Verify hash-chain continuity when
   :attr:`~orxhestra.trust.policy.TrustPolicy.require_chain` is set.
5. Consult the policy's allow/deny lists.

Failures drop the event in strict mode or annotate it under
``event.metadata["trust"]`` in permissive mode.

See Also
--------
orxhestra.trust.policy.TrustPolicy : Declarative rules this middleware
    enforces.
orxhestra.security.did : DID resolvers used to look up signer keys.
orxhestra.middleware.attestation.AttestationMiddleware : Sibling —
    pair for verify + audit in the same stack.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from orxhestra.middleware.base import BaseMiddleware

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event
    from orxhestra.security.did import DidResolver
    from orxhestra.trust.policy import PolicyDecision, TrustPolicy


logger: logging.Logger = logging.getLogger(__name__)

_CHAIN_HEADS_KEY = "_orx_trust_chain_heads"


class TrustMiddleware(BaseMiddleware):
    """Verify event signatures and enforce a :class:`~orxhestra.trust.policy.TrustPolicy`.

    Parameters
    ----------
    policy : TrustPolicy
        Rules to apply.  See
        :class:`~orxhestra.trust.policy.TrustPolicy` for mode semantics.
    resolver : DidResolver
        Resolver used to turn ``signer_did`` into an Ed25519 public
        key.  For offline usage,
        :class:`~orxhestra.security.did.DidKeyResolver` is sufficient;
        for institutional identity, wrap a
        :class:`~orxhestra.security.did.CompositeResolver` around both
        ``did:key`` and ``did:web`` resolvers.

    See Also
    --------
    TrustPolicy : Declarative rule set consulted per event.
    orxhestra.security.did.DidResolver : Public-key lookup contract.
    orxhestra.middleware.base.Middleware : Parent protocol.

    Examples
    --------
    >>> from orxhestra import Runner
    >>> from orxhestra.middleware import TrustMiddleware
    >>> from orxhestra.security.did import DidKeyResolver
    >>> from orxhestra.trust import TrustPolicy
    >>> policy = TrustPolicy(mode="strict", allow_unsigned=False)
    >>> runner = Runner(
    ...     agent=my_agent,
    ...     middleware=[TrustMiddleware(policy, DidKeyResolver())],
    ... )
    """

    def __init__(self, policy: TrustPolicy, resolver: DidResolver) -> None:
        self.policy = policy
        self.resolver = resolver

    async def on_event(
        self, ctx: InvocationContext, event: Event,
    ) -> Event | None:
        """Verify ``event`` against the policy.

        Parameters
        ----------
        ctx : InvocationContext
            Propagates the per-branch chain-head store on
            ``ctx.state[_CHAIN_HEADS_KEY]``.
        event : Event
            Event emitted by the agent.

        Returns
        -------
        Event or None
            The event (possibly annotated) in permissive mode, or
            ``None`` to drop it in strict mode on any failure.
        """
        decision = await self._evaluate(ctx, event)

        if decision.verified and event.signature and not event.partial:
            chain_heads = ctx.state.setdefault(_CHAIN_HEADS_KEY, {})
            chain_heads[event.branch] = event.signature

        if decision.allow:
            if not decision.verified and decision.reason:
                # Permissive pass-through — annotate for downstream consumers.
                event.metadata = dict(event.metadata)
                event.metadata["trust"] = {
                    "verified": False,
                    "reason": decision.reason,
                }
            elif decision.verified:
                event.metadata = dict(event.metadata)
                event.metadata["trust"] = {"verified": True}
            return event

        logger.warning(
            "TrustMiddleware dropped event %s from %s: %s",
            event.id, event.signer_did or "<unsigned>", decision.reason,
        )
        return None

    async def _evaluate(
        self, ctx: InvocationContext, event: Event,
    ) -> PolicyDecision:
        """Return a :class:`~orxhestra.trust.policy.PolicyDecision` for ``event``.

        Parameters
        ----------
        ctx : InvocationContext
        event : Event

        Returns
        -------
        PolicyDecision
        """
        from orxhestra.trust.policy import PolicyDecision

        # 1. Unsigned events.
        if not event.is_signed:
            if self.policy.requires_signature(event.type):
                return PolicyDecision(
                    allow=self.policy.mode == "permissive",
                    reason=(
                        f"{event.type.value} requires a signature but event is unsigned"
                    ),
                    verified=False,
                )
            return PolicyDecision(allow=True, reason="", verified=False)

        # 2. Signer allow/deny.
        signer_check = self.policy.check_signer(event.signer_did)
        if signer_check is not None:
            return signer_check

        # 3. Resolve public key.
        try:
            public_key = await self.resolver.resolve(event.signer_did)
        except (ValueError, LookupError, ImportError) as exc:
            return PolicyDecision(
                allow=self.policy.mode == "permissive",
                reason=f"could not resolve signer DID: {exc}",
                verified=False,
            )

        # 4. Verify signature.
        from orxhestra.security.crypto import verify_json_signature

        if not verify_json_signature(
            public_key, event.signable_payload(), event.signature or "",
        ):
            return PolicyDecision(
                allow=self.policy.mode == "permissive",
                reason="signature verification failed",
                verified=False,
            )

        # 5. Chain continuity.
        if self.policy.require_chain and not event.partial:
            chain_heads = ctx.state.get(_CHAIN_HEADS_KEY, {})
            expected = chain_heads.get(event.branch)
            if expected != event.prev_signature:
                return PolicyDecision(
                    allow=self.policy.mode == "permissive",
                    reason=(
                        f"chain broken on branch {event.branch!r}: "
                        f"expected prev_signature={expected!r} "
                        f"got {event.prev_signature!r}"
                    ),
                    verified=False,
                )

        return PolicyDecision(allow=True, reason="", verified=True)
