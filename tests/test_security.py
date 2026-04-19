"""Tests for orxhestra.security — SSRF, crypto, and token parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orxhestra.security.ssrf import (
    validate_and_pin_url,
    validate_redirect_target,
    validate_url_host,
)


class TestSSRF:
    """Tests for SSRF protection utilities."""

    def test_empty_hostname_blocked(self) -> None:
        assert validate_url_host("") is not None

    def test_localhost_blocked(self) -> None:
        assert validate_url_host("localhost") is not None

    def test_metadata_endpoint_blocked(self) -> None:
        assert validate_url_host("169.254.169.254") is not None

    def test_metadata_google_blocked(self) -> None:
        assert validate_url_host("metadata.google.internal") is not None

    def test_private_ip_blocked(self) -> None:
        assert validate_url_host("10.0.0.1") is not None

    def test_loopback_blocked(self) -> None:
        assert validate_url_host("127.0.0.1") is not None

    def test_local_suffix_blocked(self) -> None:
        assert validate_url_host("myhost.local") is not None

    def test_internal_suffix_blocked(self) -> None:
        assert validate_url_host("api.internal") is not None

    def test_public_hostname_allowed(self) -> None:
        # google.com should not be blocked
        assert validate_url_host("google.com") is None

    def test_validate_and_pin_private_ip(self) -> None:
        error, ips = validate_and_pin_url("http://127.0.0.1/secret")
        assert error is not None
        assert ips == []

    def test_validate_and_pin_invalid_url(self) -> None:
        error, ips = validate_and_pin_url("")
        assert error is not None

    def test_validate_and_pin_raw_public_ip(self) -> None:
        error, ips = validate_and_pin_url("http://8.8.8.8/dns")
        assert error is None
        assert "8.8.8.8" in ips

    def test_redirect_to_private_blocked(self) -> None:
        error = validate_redirect_target("http://127.0.0.1/", "example.com")
        assert error is not None

    def test_redirect_to_public_allowed(self) -> None:
        error = validate_redirect_target("https://example.com/callback", "example.com")
        assert error is None


crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.security.crypto import (  # noqa: E402
    canonicalize_json,
    deserialize_private_key,
    deserialize_public_key,
    did_key_fragment,
    did_key_to_public_key,
    generate_ed25519_keypair,
    load_or_create_signing_key,
    public_key_to_did_key,
    serialize_private_key,
    serialize_public_key,
    sign_json_payload,
    sign_message,
    verify_json_signature,
    verify_signature,
)


class TestCrypto:
    """Tests for Ed25519 cryptographic operations."""

    def test_keypair_generation(self) -> None:
        priv, pub = generate_ed25519_keypair()
        assert priv is not None
        assert pub is not None

    def test_private_key_round_trip(self) -> None:
        priv, _ = generate_ed25519_keypair()
        raw: bytes = serialize_private_key(priv)
        assert len(raw) == 32
        restored = deserialize_private_key(raw)
        assert serialize_private_key(restored) == raw

    def test_public_key_round_trip(self) -> None:
        _, pub = generate_ed25519_keypair()
        raw: bytes = serialize_public_key(pub)
        assert len(raw) == 32
        restored = deserialize_public_key(raw)
        assert serialize_public_key(restored) == raw

    def test_sign_and_verify(self) -> None:
        priv, pub = generate_ed25519_keypair()
        message: bytes = b"hello orxhestra"
        sig: bytes = sign_message(priv, message)
        assert verify_signature(pub, sig, message)

    def test_verify_wrong_message_fails(self) -> None:
        priv, pub = generate_ed25519_keypair()
        sig: bytes = sign_message(priv, b"correct")
        assert not verify_signature(pub, sig, b"wrong")

    def test_did_key_round_trip(self) -> None:
        _, pub = generate_ed25519_keypair()
        did: str = public_key_to_did_key(pub)
        assert did.startswith("did:key:z")
        restored = did_key_to_public_key(did)
        assert serialize_public_key(restored) == serialize_public_key(pub)

    def test_did_key_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid did:key"):
            did_key_to_public_key("not-a-did")

    def test_did_key_fragment(self) -> None:
        _, pub = generate_ed25519_keypair()
        did: str = public_key_to_did_key(pub)
        frag: str = did_key_fragment(did)
        assert frag.startswith("#z")

    def test_canonicalize_json_sorted_keys(self) -> None:
        payload: dict = {"b": 2, "a": 1}
        result: bytes = canonicalize_json(payload)
        assert json.loads(result) == {"a": 1, "b": 2}
        # Keys should be sorted in the byte representation.
        assert result.index(b'"a"') < result.index(b'"b"')

    def test_canonicalize_json_compact(self) -> None:
        payload: dict = {"key": "value"}
        result: bytes = canonicalize_json(payload)
        assert b" " not in result  # Compact separators.

    def test_sign_and_verify_json(self) -> None:
        priv, pub = generate_ed25519_keypair()
        payload: dict = {"agent": "test", "action": "sign"}
        sig: str = sign_json_payload(priv, payload)
        assert verify_json_signature(pub, payload, sig)

    def test_verify_json_tampered_fails(self) -> None:
        priv, pub = generate_ed25519_keypair()
        payload: dict = {"agent": "test"}
        sig: str = sign_json_payload(priv, payload)
        assert not verify_json_signature(pub, {"agent": "tampered"}, sig)

    def test_load_or_create_signing_key_creates_new(self, tmp_path: Path) -> None:
        key_file: Path = tmp_path / "signing_key.json"
        priv, did = load_or_create_signing_key(key_file)
        assert did.startswith("did:key:z")
        assert key_file.exists()

    def test_load_or_create_signing_key_loads_existing(self, tmp_path: Path) -> None:
        key_file: Path = tmp_path / "signing_key.json"
        priv1, did1 = load_or_create_signing_key(key_file)
        priv2, did2 = load_or_create_signing_key(key_file)
        assert did1 == did2
        assert serialize_private_key(priv1) == serialize_private_key(priv2)

    def test_load_or_create_signing_key_encrypted(self, tmp_path: Path) -> None:
        key_file: Path = tmp_path / "encrypted_key.json"
        priv1, did1 = load_or_create_signing_key(
            key_file, encryption_password="test-password",
        )
        priv2, did2 = load_or_create_signing_key(
            key_file, encryption_password="test-password",
        )
        assert did1 == did2
        # Verify the file contains encrypted data, not plaintext.
        data: dict = json.loads(key_file.read_text())
        assert "private_key_encrypted" in data
        assert "private_key_b64" not in data


jwt_mod = pytest.importorskip("jwt")

from orxhestra.security.token_parser import (  # noqa: E402
    TokenType,
    detect_token_type,
    extract_identity_from_token,
    parse_jwt_claims,
)


class TestTokenParser:
    """Tests for token type detection and JWT parsing."""

    def test_detect_did(self) -> None:
        assert detect_token_type("did:key:z6Mktest123") == TokenType.DID

    def test_detect_url(self) -> None:
        assert detect_token_type("https://example.com/agent") == TokenType.URL

    def test_detect_unknown(self) -> None:
        assert detect_token_type("short") == TokenType.UNKNOWN

    def test_detect_jwt(self) -> None:
        import jwt as pyjwt

        token: str = pyjwt.encode({"sub": "agent-1"}, "secret", algorithm="HS256")
        assert detect_token_type(token) == TokenType.JWT

    def test_parse_jwt_claims(self) -> None:
        import jwt as pyjwt

        token: str = pyjwt.encode(
            {"sub": "agent-1", "iss": "orxhestra", "scope": "read write"},
            "secret",
            algorithm="HS256",
        )
        parsed = parse_jwt_claims(token)
        assert parsed is not None
        assert parsed["subject"] == "agent-1"
        assert parsed["issuer"] == "orxhestra"
        assert parsed["scopes"] == ["read", "write"]

    def test_parse_jwt_invalid(self) -> None:
        assert parse_jwt_claims("not-a-jwt") is None

    def test_extract_identity_did(self) -> None:
        result = extract_identity_from_token("did:web:example.com")
        assert result["token_type"] == "did"
        assert result["did_method"] == "web"

    def test_extract_identity_url(self) -> None:
        result = extract_identity_from_token("https://agent.example.com")
        assert result["token_type"] == "url"
        assert "url" in result

    def test_extract_identity_jwt(self) -> None:
        import jwt as pyjwt

        token: str = pyjwt.encode(
            {"sub": "agent-1", "iss": "test"}, "secret", algorithm="HS256",
        )
        result = extract_identity_from_token(token)
        assert result["token_type"] == "jwt"
        assert result["subject"] == "agent-1"
