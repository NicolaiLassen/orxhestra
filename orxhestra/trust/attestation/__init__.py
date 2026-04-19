"""Attestation providers — the claim-issuing sub-domain of trust.

Orxhestra ships a protocol plus two reference implementations:

- :class:`AttestationProvider` — the stable seam.  External services
  (blockchain anchors, enterprise audit stores, compliance backends)
  plug in by implementing this protocol in their own packages — no
  named-provider adapter is shipped in the core repo.
- :class:`NoOpAttestationProvider` — drop-in default that records
  nothing.  Keeps the framework lean when attestation isn't needed.
- :class:`LocalAttestationProvider` — append-only JSON-on-disk audit
  log with SHA-256 hash chaining and detached Ed25519 signatures on
  each entry.  Zero external dependencies beyond
  :mod:`orxhestra.security.crypto`.

The companion :class:`~orxhestra.middleware.AttestationMiddleware`
drives a provider from the agent loop; it lives under
:mod:`orxhestra.middleware` alongside every other middleware.

See Also
--------
orxhestra.middleware.attestation.AttestationMiddleware : Runtime
    consumer of these providers.
orxhestra.trust.policy.TrustPolicy : Sibling trust primitive for
    signature verification.
"""

from orxhestra.trust.attestation.local import LocalAttestationProvider
from orxhestra.trust.attestation.noop import NoOpAttestationProvider
from orxhestra.trust.attestation.protocol import (
    AttestationProvider,
    Claim,
)

__all__ = [
    "AttestationProvider",
    "Claim",
    "LocalAttestationProvider",
    "NoOpAttestationProvider",
]
