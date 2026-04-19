"""``orx identity`` CLI subcommand — manage Ed25519 signing identities.

Three actions:

- ``orx identity init [--path KEYFILE] [--encrypt]`` — generate a
  new Ed25519 keypair and write it to disk with its derived
  ``did:key``.
- ``orx identity show [--path KEYFILE]`` — print the DID and
  public-key multibase of an existing key file.
- ``orx identity did-web <domain> [path...]`` — render a
  ``did.json`` document suitable for hosting under
  ``https://<domain>/.well-known/did.json`` (or a sub-path).

All output goes to stdout; errors go to stderr with a non-zero exit
code.  The commands never print the private key.

See Also
--------
orxhestra.cli.app : Registers the parent ``identity`` subcommand and
    routes into :func:`run_parsed`.
orxhestra.security.crypto.load_or_create_signing_key : Underlying key
    persistence primitive.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

DEFAULT_KEY_PATH = Path.home() / ".orx" / "identity.key"
"""Default location for locally-generated identity keys."""


def run_parsed(args: argparse.Namespace) -> int:
    """Dispatch a pre-parsed ``identity`` argparse namespace.

    Called by :func:`orxhestra.cli.app.main` after the shared
    top-level parser has already resolved ``identity <action>``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.  Must carry ``action`` plus whatever
        action-specific fields the subparser added.

    Returns
    -------
    int
        Process exit code.  ``0`` on success, non-zero on failure.
    """
    try:
        if args.action == "init":
            return _cmd_init(args)
        if args.action == "show":
            return _cmd_show(args)
        if args.action == "did-web":
            return _cmd_did_web(args)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: install the auth extra: pip install 'orxhestra[auth]'",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"error: unknown action {args.action!r}", file=sys.stderr)
    return 1


def _cmd_init(args: argparse.Namespace) -> int:
    """Implement ``orx identity init``.

    Generates a new Ed25519 keypair (or loads the existing file when
    one is already present) and prints the derived DID.  Honors
    ``$ORX_KEY_PASSWORD`` when ``--encrypt`` is set.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``init`` arguments — ``path`` and ``encrypt``.

    Returns
    -------
    int
        Process exit code.
    """
    from orxhestra.security.crypto import load_or_create_signing_key

    path = Path(args.path).expanduser()
    password: str | None = None
    if args.encrypt:
        password = os.environ.get("ORX_KEY_PASSWORD")
        if not password:
            print(
                "error: --encrypt requires $ORX_KEY_PASSWORD to be set.",
                file=sys.stderr,
            )
            return 2

    _, did = load_or_create_signing_key(path, encryption_password=password)
    print(f"wrote identity to {path}")
    print(f"did: {did}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Implement ``orx identity show``.

    Reads the key file, derives the public key, and prints the DID
    plus the multibase-encoded public key.  Never prints the private
    material.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``show`` arguments — ``path``.

    Returns
    -------
    int
        Process exit code.
    """
    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"error: key file not found: {path}", file=sys.stderr)
        return 2

    data = json.loads(path.read_text())
    did = data.get("did_key", "")
    encryption = data.get("encryption", "none")

    public_multibase = ""
    if "private_key_b64" in data:
        import base58 as _base58

        from orxhestra.security.crypto import (
            deserialize_private_key,
            public_key_to_did_key,
            serialize_public_key,
        )

        priv = deserialize_private_key(base64.b64decode(data["private_key_b64"]))
        pub = priv.public_key()
        derived_did = public_key_to_did_key(pub)
        if derived_did != did:
            print(
                "warning: did_key field does not match derived DID "
                f"({did} vs {derived_did})",
                file=sys.stderr,
            )
        public_multibase = "z" + _base58.b58encode(
            bytes([0xED, 0x01]) + serialize_public_key(pub),
        ).decode("ascii")

    print(f"did: {did}")
    print(f"encryption: {encryption}")
    if public_multibase:
        print(f"public_key_multibase: {public_multibase}")
    return 0


def _cmd_did_web(args: argparse.Namespace) -> int:
    """Implement ``orx identity did-web``.

    Builds a W3C-compliant ``did.json`` document for a ``did:web``
    identity whose public key lives in the provided key file.  The
    document is printed to stdout so the caller can redirect it to
    ``<domain>/.well-known/did.json`` on their web host.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``did-web`` arguments — ``domain``, ``sub_path``, and
        ``path``.

    Returns
    -------
    int
        Process exit code.
    """
    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"error: key file not found: {path}", file=sys.stderr)
        return 2

    import base58 as _base58

    from orxhestra.security.crypto import (
        deserialize_private_key,
        serialize_public_key,
    )

    data = json.loads(path.read_text())
    if "private_key_b64" not in data:
        print(
            "error: did-web does not support encrypted key files yet.",
            file=sys.stderr,
        )
        return 2

    priv = deserialize_private_key(base64.b64decode(data["private_key_b64"]))
    pub = priv.public_key()
    multibase = "z" + _base58.b58encode(
        bytes([0xED, 0x01]) + serialize_public_key(pub),
    ).decode("ascii")

    segments = [args.domain] + list(args.sub_path or [])
    did = "did:web:" + ":".join(segments)

    document = {
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://w3id.org/security/suites/ed25519-2020/v1",
        ],
        "id": did,
        "verificationMethod": [
            {
                "id": f"{did}#key-1",
                "type": "Ed25519VerificationKey2020",
                "controller": did,
                "publicKeyMultibase": multibase,
            },
        ],
        "authentication": [f"{did}#key-1"],
        "assertionMethod": [f"{did}#key-1"],
    }

    print(json.dumps(document, indent=2))
    return 0
