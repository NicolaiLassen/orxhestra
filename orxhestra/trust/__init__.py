"""Trust domain — policy, attestation providers, and verification middleware.

The domain is split across two concerns under one umbrella:

- :class:`TrustPolicy` / :class:`PolicyDecision` — declarative rules
  for signature verification (allow/deny lists, strict vs permissive,
  chain-continuity requirements).
- :mod:`orxhestra.trust.attestation` — pluggable claim-issuing
  providers (:class:`AttestationProvider` protocol, plus
  :class:`NoOpAttestationProvider` and :class:`LocalAttestationProvider`
  reference implementations).

The middleware that *uses* these primitives —
:class:`~orxhestra.middleware.TrustMiddleware` and
:class:`~orxhestra.middleware.AttestationMiddleware` — lives under
:mod:`orxhestra.middleware` alongside every other middleware.  Both
are re-exported here for single-line-import convenience.

The whole layer is **opt-in**.  Agents run unchanged with no trust or
attestation middleware registered; enforcement only activates when
you explicitly pass a middleware to :class:`~orxhestra.Runner`
(or declare ``identity:`` / ``trust:`` / ``attestation:`` blocks in
composer YAML).

See Also
--------
orxhestra.middleware : Home of :class:`TrustMiddleware` and
    :class:`AttestationMiddleware`.
orxhestra.security : Cryptographic primitives (Ed25519, DID codec,
    SSRF guard) that the trust layer builds on.
"""

from orxhestra.middleware.attestation import AttestationMiddleware
from orxhestra.middleware.trust import TrustMiddleware
from orxhestra.trust.attestation import (
    AttestationProvider,
    Claim,
    LocalAttestationProvider,
    NoOpAttestationProvider,
)
from orxhestra.trust.policy import PolicyDecision, TrustPolicy

__all__ = [
    "AttestationMiddleware",
    "AttestationProvider",
    "Claim",
    "LocalAttestationProvider",
    "NoOpAttestationProvider",
    "PolicyDecision",
    "TrustMiddleware",
    "TrustPolicy",
]
