"""DID resolvers — pluggable backends for ``did:key`` and ``did:web``.

A :class:`DidResolver` turns a DID string into an Ed25519 public key so
downstream components — :class:`~orxhestra.middleware.trust.TrustMiddleware`,
:class:`~orxhestra.a2a.server.A2AServer` — can verify signatures without
hardcoding a DID method.

Resolvers are registered on the :class:`~orxhestra.runner.Runner` (or
passed explicitly to
:class:`~orxhestra.middleware.trust.TrustMiddleware`) and selected by
DID prefix:

- ``did:key:z...``  →  :class:`DidKeyResolver` (stateless, offline).
- ``did:web:host:path``  →  :class:`DidWebResolver` (HTTPS fetch of
  ``/.well-known/did.json``).
- ``did:...`` mix  →  :class:`CompositeResolver` picks by prefix.

Requires ``orxhestra[auth]``.  :class:`DidWebResolver` additionally
needs ``httpx``; the import is lazy and errors with an actionable
install hint if missing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from orxhestra.security.ssrf import validate_url_host

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


logger: logging.Logger = logging.getLogger(__name__)


@runtime_checkable
class DidResolver(Protocol):
    """Resolve a DID into its Ed25519 public key.

    Implementations must be async-safe and re-entrant.  They should
    raise :class:`ValueError` on malformed DIDs and
    :class:`LookupError` when the DID document cannot be located or
    does not contain a suitable Ed25519 verification method.

    See Also
    --------
    DidKeyResolver : Offline resolver for ``did:key``.
    DidWebResolver : HTTPS resolver for ``did:web``.
    CompositeResolver : Dispatch by DID prefix.
    orxhestra.middleware.trust.TrustMiddleware : Primary consumer.
    """

    async def resolve(self, did: str) -> Ed25519PublicKey:
        """Return the Ed25519 public key for ``did``.

        Parameters
        ----------
        did : str
            The DID string to resolve.

        Returns
        -------
        Ed25519PublicKey

        Raises
        ------
        ValueError
            If ``did`` is not in a format this resolver understands.
        LookupError
            If the DID document cannot be fetched or parsed.
        """
        ...


class DidKeyResolver:
    """Resolver for ``did:key:z...`` identifiers.

    Thin adapter around :func:`orxhestra.security.crypto.did_key_to_public_key`.
    Stateless and offline — the key is encoded directly in the DID.

    See Also
    --------
    DidResolver : Protocol implemented here.
    orxhestra.security.crypto.did_key_to_public_key : Underlying codec.
    """

    async def resolve(self, did: str) -> Ed25519PublicKey:
        """Decode the public key embedded in a ``did:key`` identifier.

        Parameters
        ----------
        did : str
            A ``did:key:z...`` string.

        Returns
        -------
        Ed25519PublicKey

        Raises
        ------
        ValueError
            If ``did`` is not a ``did:key`` or uses a non-Ed25519
            multicodec prefix.
        """
        from orxhestra.security.crypto import did_key_to_public_key

        return did_key_to_public_key(did)


class DidWebResolver:
    """Resolver for ``did:web:host:path`` identifiers.

    Fetches ``https://<host>/<path...>/did.json`` (per W3C DID
    Method: Web), validates SSRF constraints via
    :func:`orxhestra.security.ssrf.validate_url_host`, and extracts the
    first Ed25519 verification method.

    Parameters
    ----------
    http_client : httpx.AsyncClient, optional
        Shared HTTP client.  When omitted, a short-lived client is
        created per request.
    timeout : float
        HTTP timeout in seconds (default 5.0).

    See Also
    --------
    DidResolver : Protocol implemented here.
    orxhestra.security.ssrf : SSRF guard used before fetching.
    """

    def __init__(
        self,
        *,
        http_client: object | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._client = http_client
        self._timeout = timeout

    async def resolve(self, did: str) -> Ed25519PublicKey:
        """Fetch and parse the ``did.json`` document for ``did``.

        Parameters
        ----------
        did : str
            A ``did:web:...`` string.

        Returns
        -------
        Ed25519PublicKey

        Raises
        ------
        ValueError
            If ``did`` is not a ``did:web`` or the document lacks a
            usable Ed25519 verification method.
        LookupError
            If the host is blocked by the SSRF guard, the HTTP fetch
            fails, or the response is not valid JSON.
        ImportError
            If ``httpx`` is not installed.
        """
        if not did.startswith("did:web:"):
            raise ValueError(f"Not a did:web identifier: {did}")

        url = _did_web_to_url(did)
        host = url.split("://", 1)[1].split("/", 1)[0]
        ssrf_error = validate_url_host(host)
        if ssrf_error:
            raise LookupError(f"did:web host rejected: {ssrf_error}")

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "did:web resolution requires httpx. Install with: "
                "pip install httpx"
            ) from exc

        if self._client is not None:
            response = await self._client.get(url, timeout=self._timeout)  # type: ignore[union-attr]
        else:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)

        if response.status_code != 200:
            raise LookupError(
                f"did:web fetch returned HTTP {response.status_code} for {url}"
            )

        try:
            document: dict = response.json()
        except ValueError as exc:
            raise LookupError(f"did:web document at {url} is not JSON") from exc

        return _extract_ed25519_key(document, did)


class CompositeResolver:
    """Dispatch DID resolution by method prefix.

    Walks the provided resolvers in order and delegates to the first
    one that does not raise :class:`ValueError` for the given DID.
    If every resolver raises :class:`ValueError`, the composite
    re-raises the last one.

    Parameters
    ----------
    resolvers : list[DidResolver]
        Resolvers to try in order.

    See Also
    --------
    DidKeyResolver : Offline ``did:key``.
    DidWebResolver : HTTPS ``did:web``.
    """

    def __init__(self, resolvers: list[DidResolver]) -> None:
        if not resolvers:
            raise ValueError("CompositeResolver requires at least one resolver.")
        self._resolvers = resolvers

    async def resolve(self, did: str) -> Ed25519PublicKey:
        """Resolve ``did`` using the first matching resolver.

        Parameters
        ----------
        did : str

        Returns
        -------
        Ed25519PublicKey

        Raises
        ------
        ValueError
            If no registered resolver recognises the DID method.
        LookupError
            Propagated from the matching resolver when the document
            cannot be retrieved.
        """
        last_error: Exception | None = None
        for resolver in self._resolvers:
            try:
                return await resolver.resolve(did)
            except ValueError as exc:
                last_error = exc
                continue
        raise ValueError(
            f"No resolver accepted DID {did!r}: {last_error}"
        ) from last_error


def _did_web_to_url(did: str) -> str:
    """Convert a ``did:web`` identifier to its HTTPS document URL.

    Per the W3C DID Method: Web spec, colons after the method act as
    path separators and ``:`` inside the identifier map to ``/`` in
    the URL.  Omitted paths default to ``/.well-known/did.json``.

    Parameters
    ----------
    did : str
        A ``did:web:host[:path...]`` string.

    Returns
    -------
    str
        The canonical HTTPS URL to ``did.json``.
    """
    remainder = did[len("did:web:"):]
    segments = remainder.split(":")
    host = segments[0]
    if len(segments) == 1:
        return f"https://{host}/.well-known/did.json"
    path = "/".join(segments[1:])
    return f"https://{host}/{path}/did.json"


def _extract_ed25519_key(document: dict, did: str) -> Ed25519PublicKey:
    """Pick the first Ed25519 verification method out of a DID document.

    Accepts ``Ed25519VerificationKey2020`` and the legacy
    ``Ed25519VerificationKey2018`` types, preferring
    ``publicKeyMultibase`` encoding and falling back to
    ``publicKeyBase58`` when that is what the document supplies.

    Parameters
    ----------
    document : dict
        The parsed DID document.
    did : str
        The DID that was resolved — used for clearer error messages.

    Returns
    -------
    Ed25519PublicKey

    Raises
    ------
    ValueError
        If no usable Ed25519 verification method is present.
    """
    import base58

    from orxhestra.security.crypto import (
        ED25519_MULTICODEC_PREFIX,
        deserialize_public_key,
    )

    methods = document.get("verificationMethod") or []
    for method in methods:
        vm_type = method.get("type", "")
        if vm_type not in {
            "Ed25519VerificationKey2020",
            "Ed25519VerificationKey2018",
        }:
            continue

        multibase = method.get("publicKeyMultibase")
        if multibase and multibase.startswith("z"):
            decoded = base58.b58decode(multibase[1:])
            if decoded[:2] == ED25519_MULTICODEC_PREFIX:
                return deserialize_public_key(decoded[2:])
            if len(decoded) == 32:
                return deserialize_public_key(decoded)

        base58_key = method.get("publicKeyBase58")
        if base58_key:
            decoded = base58.b58decode(base58_key)
            if len(decoded) == 32:
                return deserialize_public_key(decoded)

    raise ValueError(f"No Ed25519 verification method found in DID document for {did}")
