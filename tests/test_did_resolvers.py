"""Tests for ``orxhestra.security.did`` resolver implementations."""

from __future__ import annotations

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

import base58  # noqa: E402

from orxhestra.security.crypto import (  # noqa: E402
    ED25519_MULTICODEC_PREFIX,
    generate_ed25519_keypair,
    public_key_to_did_key,
    serialize_public_key,
)
from orxhestra.security.did import (  # noqa: E402
    CompositeResolver,
    DidKeyResolver,
    DidWebResolver,
    _did_web_to_url,
    _extract_ed25519_key,
)


class TestDidKeyResolver:
    """Offline did:key round-trip."""

    @pytest.mark.asyncio
    async def test_resolves_generated_key(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        resolver = DidKeyResolver()
        resolved = await resolver.resolve(did)
        assert serialize_public_key(resolved) == serialize_public_key(pub)

    @pytest.mark.asyncio
    async def test_rejects_non_did_key(self) -> None:
        resolver = DidKeyResolver()
        with pytest.raises(ValueError):
            await resolver.resolve("did:web:example.com")


class TestDidWebUrlMapping:
    """Pure helpers that map did:web identifiers to HTTPS URLs."""

    def test_host_only(self) -> None:
        assert (
            _did_web_to_url("did:web:example.com")
            == "https://example.com/.well-known/did.json"
        )

    def test_with_path_segments(self) -> None:
        assert (
            _did_web_to_url("did:web:example.com:agents:researcher")
            == "https://example.com/agents/researcher/did.json"
        )


class TestExtractEd25519Key:
    """Verification method extraction from a DID document."""

    def test_multibase_with_multicodec_prefix(self) -> None:
        _, pub = generate_ed25519_keypair()
        raw = serialize_public_key(pub)
        multibase = "z" + base58.b58encode(
            ED25519_MULTICODEC_PREFIX + raw,
        ).decode("ascii")
        doc = {
            "verificationMethod": [
                {
                    "type": "Ed25519VerificationKey2020",
                    "publicKeyMultibase": multibase,
                },
            ],
        }
        key = _extract_ed25519_key(doc, "did:web:x")
        assert serialize_public_key(key) == raw

    def test_base58_raw_key(self) -> None:
        _, pub = generate_ed25519_keypair()
        raw = serialize_public_key(pub)
        doc = {
            "verificationMethod": [
                {
                    "type": "Ed25519VerificationKey2018",
                    "publicKeyBase58": base58.b58encode(raw).decode("ascii"),
                },
            ],
        }
        key = _extract_ed25519_key(doc, "did:web:x")
        assert serialize_public_key(key) == raw

    def test_no_ed25519_method_raises(self) -> None:
        with pytest.raises(ValueError):
            _extract_ed25519_key({"verificationMethod": []}, "did:web:x")


class TestCompositeResolver:
    """Dispatch across multiple resolvers."""

    @pytest.mark.asyncio
    async def test_routes_to_first_accepting_resolver(self) -> None:
        priv, pub = generate_ed25519_keypair()
        did = public_key_to_did_key(pub)
        composite = CompositeResolver([DidKeyResolver(), DidWebResolver()])
        resolved = await composite.resolve(did)
        assert serialize_public_key(resolved) == serialize_public_key(pub)

    @pytest.mark.asyncio
    async def test_rejects_unknown_method(self) -> None:
        composite = CompositeResolver([DidKeyResolver()])
        with pytest.raises(ValueError):
            await composite.resolve("did:bogus:123")

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            CompositeResolver([])


class TestDidWebResolverSSRF:
    """``DidWebResolver`` blocks private hosts via the SSRF guard."""

    @pytest.mark.asyncio
    async def test_rejects_localhost(self) -> None:
        resolver = DidWebResolver()
        with pytest.raises(LookupError):
            await resolver.resolve("did:web:localhost")
