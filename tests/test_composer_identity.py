"""Tests for composer YAML wiring of identity / trust / attestation."""

from __future__ import annotations

from pathlib import Path

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")
yaml_mod = pytest.importorskip("yaml")

from orxhestra.composer.schema import (  # noqa: E402
    AttestationConfig,
    ComposeSpec,
    IdentityConfig,
    TrustConfig,
)
from orxhestra.security.crypto import load_or_create_signing_key  # noqa: E402


class TestSchemaValidation:
    """Field-level validation on the new YAML blocks."""

    def test_identity_defaults_to_did_key(self) -> None:
        cfg = IdentityConfig(signing_key="/tmp/k")
        assert cfg.did_method == "key"
        assert cfg.did is None

    def test_identity_web_requires_explicit_did(self) -> None:
        with pytest.raises(ValueError):
            IdentityConfig(signing_key="/tmp/k", did_method="web")

    def test_identity_web_accepts_did(self) -> None:
        cfg = IdentityConfig(
            signing_key="/tmp/k",
            did_method="web",
            did="did:web:example.com",
        )
        assert cfg.did == "did:web:example.com"

    def test_trust_defaults_to_permissive(self) -> None:
        cfg = TrustConfig()
        assert cfg.mode == "permissive"
        assert cfg.trusted_dids == []
        assert cfg.denied_dids == []
        assert cfg.require_chain is False
        assert cfg.allow_unsigned is True

    def test_attestation_local_requires_path(self) -> None:
        with pytest.raises(ValueError):
            AttestationConfig(provider="local")

    def test_attestation_local_with_path(self) -> None:
        cfg = AttestationConfig(provider="local", path="/tmp/audit")
        assert cfg.path == "/tmp/audit"

    def test_compose_spec_accepts_optional_blocks(self) -> None:
        spec = ComposeSpec.model_validate({
            "agents": {"root": {"type": "llm"}},
            "main_agent": "root",
            "identity": {"signing_key": "/tmp/k"},
            "trust": {"mode": "strict"},
            "attestation": {"provider": "noop"},
        })
        assert spec.identity is not None
        assert spec.trust is not None
        assert spec.attestation is not None


class TestComposerIdentityResolution:
    """End-to-end: _resolve_identity loads the key from YAML spec."""

    def test_resolve_identity_expands_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orxhestra.composer.composer import Composer

        key_path = tmp_path / "id.key"
        _, did = load_or_create_signing_key(key_path)

        monkeypatch.setenv("TEST_KEY_PATH", str(key_path))
        spec = ComposeSpec.model_validate({
            "agents": {"root": {"type": "llm"}},
            "main_agent": "root",
            "identity": {"signing_key": "${TEST_KEY_PATH}"},
        })
        composer = Composer(spec)
        signing_key, resolved_did = composer._resolve_identity()
        assert signing_key is not None
        assert resolved_did == did

    def test_resolve_identity_returns_none_when_absent(self) -> None:
        from orxhestra.composer.composer import Composer

        spec = ComposeSpec.model_validate({
            "agents": {"root": {"type": "llm"}},
            "main_agent": "root",
        })
        composer = Composer(spec)
        key, did = composer._resolve_identity()
        assert key is None
        assert did == ""

    def test_build_middleware_assembles_trust_and_attestation(
        self, tmp_path: Path,
    ) -> None:
        from orxhestra.composer.composer import Composer

        key_path = tmp_path / "id.key"
        _, did = load_or_create_signing_key(key_path)
        audit_dir = tmp_path / "audit"

        spec = ComposeSpec.model_validate({
            "agents": {"root": {"type": "llm"}},
            "main_agent": "root",
            "identity": {"signing_key": str(key_path)},
            "trust": {"mode": "strict"},
            "attestation": {"provider": "local", "path": str(audit_dir)},
        })
        composer = Composer(spec)
        signing_key, signer_did = composer._resolve_identity()
        middleware = composer._build_middleware(signing_key, signer_did)

        from orxhestra.middleware import AttestationMiddleware, TrustMiddleware

        assert len(middleware) == 2
        assert isinstance(middleware[0], TrustMiddleware)
        assert isinstance(middleware[1], AttestationMiddleware)
