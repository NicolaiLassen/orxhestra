"""Security primitives — cryptography, DID resolvers, SSRF guard, JWT parser.

Low-level building blocks that higher-level layers
(:mod:`orxhestra.trust`, :mod:`orxhestra.middleware`,
:mod:`orxhestra.a2a.signing`) compose on top of.  Split cleanly from
the trust *domain* because these primitives are useful on their own
(signing an arbitrary payload, resolving a DID, validating an outbound
URL) without pulling in the full policy/attestation stack.

Availability:

- :mod:`~orxhestra.security.ssrf` is always available (stdlib only).
- :mod:`~orxhestra.security.crypto`, :mod:`~orxhestra.security.did`,
  and :mod:`~orxhestra.security.token_parser` require the optional
  ``orxhestra[auth]`` extra (``cryptography``, ``base58``, ``PyJWT``).
  Symbols from those modules are lazy-loaded through this package so
  importing :mod:`orxhestra.security` alone is cheap.

Install::

    pip install orxhestra[auth]

See Also
--------
orxhestra.trust : Higher-level trust domain (policy + attestation
    providers) that consumes these primitives.
orxhestra.middleware : Middleware implementations (``TrustMiddleware``,
    ``AttestationMiddleware``) that drive the primitives at runtime.
"""

from orxhestra.security.ssrf import (
    validate_and_pin_url,
    validate_redirect_target,
    validate_url_host,
)

# Symbols that require optional deps are lazy-loaded below.
_CRYPTO_SYMBOLS: dict[str, tuple[str, str]] = {
    "generate_ed25519_keypair": ("orxhestra.security.crypto", "generate_ed25519_keypair"),
    "serialize_private_key": ("orxhestra.security.crypto", "serialize_private_key"),
    "deserialize_private_key": ("orxhestra.security.crypto", "deserialize_private_key"),
    "serialize_public_key": ("orxhestra.security.crypto", "serialize_public_key"),
    "deserialize_public_key": ("orxhestra.security.crypto", "deserialize_public_key"),
    "sign_message": ("orxhestra.security.crypto", "sign_message"),
    "verify_signature": ("orxhestra.security.crypto", "verify_signature"),
    "public_key_to_did_key": ("orxhestra.security.crypto", "public_key_to_did_key"),
    "did_key_to_public_key": ("orxhestra.security.crypto", "did_key_to_public_key"),
    "did_key_fragment": ("orxhestra.security.crypto", "did_key_fragment"),
    "load_or_create_signing_key": ("orxhestra.security.crypto", "load_or_create_signing_key"),
    "canonicalize_json": ("orxhestra.security.crypto", "canonicalize_json"),
    "sign_json_payload": ("orxhestra.security.crypto", "sign_json_payload"),
    "verify_json_signature": ("orxhestra.security.crypto", "verify_json_signature"),
    "ED25519_MULTICODEC_PREFIX": ("orxhestra.security.crypto", "ED25519_MULTICODEC_PREFIX"),
    "TokenType": ("orxhestra.security.token_parser", "TokenType"),
    "detect_token_type": ("orxhestra.security.token_parser", "detect_token_type"),
    "parse_jwt_claims": ("orxhestra.security.token_parser", "parse_jwt_claims"),
    "extract_identity_from_token": (
        "orxhestra.security.token_parser",
        "extract_identity_from_token",
    ),
}


def __getattr__(name: str):
    """Lazy-load crypto and token_parser symbols to avoid hard dependency."""
    if name in _CRYPTO_SYMBOLS:
        module_path, attr = _CRYPTO_SYMBOLS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # ssrf (always available)
    "validate_url_host",
    "validate_and_pin_url",
    "validate_redirect_target",
    # crypto (requires orxhestra[auth])
    "generate_ed25519_keypair",
    "serialize_private_key",
    "deserialize_private_key",
    "serialize_public_key",
    "deserialize_public_key",
    "sign_message",
    "verify_signature",
    "public_key_to_did_key",
    "did_key_to_public_key",
    "did_key_fragment",
    "load_or_create_signing_key",
    "canonicalize_json",
    "sign_json_payload",
    "verify_json_signature",
    "ED25519_MULTICODEC_PREFIX",
    # token_parser (requires orxhestra[auth])
    "TokenType",
    "detect_token_type",
    "parse_jwt_claims",
    "extract_identity_from_token",
]
