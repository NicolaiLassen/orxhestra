"""Tests for the ``orx identity`` CLI subcommand."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

crypto = pytest.importorskip("cryptography")
base58_mod = pytest.importorskip("base58")

from orxhestra.cli.identity import (  # noqa: E402
    DEFAULT_KEY_PATH,
    run_parsed,
)


def _init_args(path: Path, encrypt: bool = False) -> argparse.Namespace:
    """Build an argparse namespace for ``orx identity init``."""
    return argparse.Namespace(
        action="init", path=str(path), encrypt=encrypt,
    )


def _show_args(path: Path) -> argparse.Namespace:
    """Build an argparse namespace for ``orx identity show``."""
    return argparse.Namespace(action="show", path=str(path))


def _did_web_args(
    path: Path, domain: str, sub: list[str] | None = None,
) -> argparse.Namespace:
    """Build an argparse namespace for ``orx identity did-web``."""
    return argparse.Namespace(
        action="did-web", path=str(path), domain=domain, sub_path=sub or [],
    )


class TestIdentityCLI:
    """End-to-end behaviour of :func:`orxhestra.cli.identity.run_parsed`."""

    def test_default_key_path_under_home(self) -> None:
        assert DEFAULT_KEY_PATH == Path.home() / ".orx" / "identity.key"

    def test_init_writes_did_key(self, tmp_path: Path) -> None:
        key_path = tmp_path / "id.key"
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_parsed(_init_args(key_path))
        assert rc == 0
        assert key_path.exists()
        out = buf.getvalue()
        assert "did: did:key:z" in out

    def test_init_encrypt_requires_password_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ORX_KEY_PASSWORD", raising=False)
        key_path = tmp_path / "id.key"
        err = io.StringIO()
        with redirect_stderr(err):
            rc = run_parsed(_init_args(key_path, encrypt=True))
        assert rc == 2
        assert "ORX_KEY_PASSWORD" in err.getvalue()

    def test_show_prints_did_and_multibase(self, tmp_path: Path) -> None:
        key_path = tmp_path / "id.key"
        run_parsed(_init_args(key_path))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_parsed(_show_args(key_path))
        assert rc == 0
        out = buf.getvalue()
        assert "did: did:key:z" in out
        assert "public_key_multibase: z" in out
        assert "encryption: none" in out

    def test_show_missing_file(self, tmp_path: Path) -> None:
        key_path = tmp_path / "nope.key"
        err = io.StringIO()
        with redirect_stderr(err):
            rc = run_parsed(_show_args(key_path))
        assert rc == 2
        assert "key file not found" in err.getvalue()

    def test_did_web_emits_document(self, tmp_path: Path) -> None:
        key_path = tmp_path / "id.key"
        run_parsed(_init_args(key_path))

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_parsed(_did_web_args(key_path, "example.com", ["agents"]))
        assert rc == 0
        doc = json.loads(buf.getvalue())
        assert doc["id"] == "did:web:example.com:agents"
        vm = doc["verificationMethod"][0]
        assert vm["type"] == "Ed25519VerificationKey2020"
        assert vm["publicKeyMultibase"].startswith("z")

    def test_unknown_action_reports_error(self, tmp_path: Path) -> None:
        args = argparse.Namespace(action="bogus")
        err = io.StringIO()
        with redirect_stderr(err):
            rc = run_parsed(args)
        assert rc == 1
