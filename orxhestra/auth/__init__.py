"""Authentication, cryptography, and security utilities.

SSRF protection is always available (stdlib only).  Ed25519 crypto and
JWT parsing require optional dependencies::

    pip install orxhestra[auth]
"""

from orxhestra.auth.ssrf import (
    validate_and_pin_url,
    validate_redirect_target,
    validate_url_host,
)

# Symbols that require optional deps are lazy-loaded below.
_CRYPTO_SYMBOLS: dict[str, tuple[str, str]] = {
    "generate_ed25519_keypair": ("orxhestra.auth.crypto", "generate_ed25519_keypair"),
    "serialize_private_key": ("orxhestra.auth.crypto", "serialize_private_key"),
    "deserialize_private_key": ("orxhestra.auth.crypto", "deserialize_private_key"),
    "serialize_public_key": ("orxhestra.auth.crypto", "serialize_public_key"),
    "deserialize_public_key": ("orxhestra.auth.crypto", "deserialize_public_key"),
    "sign_message": ("orxhestra.auth.crypto", "sign_message"),
    "verify_signature": ("orxhestra.auth.crypto", "verify_signature"),
    "public_key_to_did_key": ("orxhestra.auth.crypto", "public_key_to_did_key"),
    "did_key_to_public_key": ("orxhestra.auth.crypto", "did_key_to_public_key"),
    "did_key_fragment": ("orxhestra.auth.crypto", "did_key_fragment"),
    "load_or_create_signing_key": ("orxhestra.auth.crypto", "load_or_create_signing_key"),
    "canonicalize_json": ("orxhestra.auth.crypto", "canonicalize_json"),
    "sign_json_payload": ("orxhestra.auth.crypto", "sign_json_payload"),
    "verify_json_signature": ("orxhestra.auth.crypto", "verify_json_signature"),
    "ED25519_MULTICODEC_PREFIX": ("orxhestra.auth.crypto", "ED25519_MULTICODEC_PREFIX"),
    "TokenType": ("orxhestra.auth.token_parser", "TokenType"),
    "detect_token_type": ("orxhestra.auth.token_parser", "detect_token_type"),
    "parse_jwt_claims": ("orxhestra.auth.token_parser", "parse_jwt_claims"),
    "extract_identity_from_token": (
        "orxhestra.auth.token_parser",
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
