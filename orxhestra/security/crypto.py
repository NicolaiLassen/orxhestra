"""Ed25519 cryptographic operations for orxhestra.

Handles key generation, serialisation, signing, verification, and
``did:key`` creation.  Also provides RFC 8785 JSON canonicalisation
for deterministic payload signing.

Requires the ``cryptography`` and ``base58`` packages.  Install via::

    pip install orxhestra[auth]
"""

from __future__ import annotations

import base64
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

try:
    import base58
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    _HAS_CRYPTO_DEPS: bool = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTO_DEPS = False

# Multicodec prefix for Ed25519 public key (0xed 0x01).
ED25519_MULTICODEC_PREFIX: bytes = bytes([0xED, 0x01])


def _check_crypto_deps() -> None:
    """Raise ``ImportError`` if crypto dependencies are missing."""
    if not _HAS_CRYPTO_DEPS:
        raise ImportError(
            "cryptography and base58 are required for Ed25519 operations. "
            "Install them with: pip install orxhestra[auth]"
        )


# ── Key generation & serialisation ──────────────────────────────


def generate_ed25519_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a new Ed25519 keypair.

    Returns
    -------
    tuple[Ed25519PrivateKey, Ed25519PublicKey]
        The generated private and public keys.

    Raises
    ------
    ImportError
        If ``cryptography`` is not installed.

    See Also
    --------
    load_or_create_signing_key : Persist keys across runs.
    public_key_to_did_key : Derive a ``did:key`` identity.
    sign_json_payload : Sign a payload with the private key.
    """
    _check_crypto_deps()
    private_key: Ed25519PrivateKey = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def serialize_private_key(private_key: Ed25519PrivateKey) -> bytes:
    """Serialise a private key to its raw 32-byte seed.

    Parameters
    ----------
    private_key : Ed25519PrivateKey
        The private key to serialise.

    Returns
    -------
    bytes
        Raw 32-byte seed.
    """
    return private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())


def deserialize_private_key(key_bytes: bytes) -> Ed25519PrivateKey:
    """Deserialise a private key from a raw 32-byte seed.

    Parameters
    ----------
    key_bytes : bytes
        Raw 32-byte seed.

    Returns
    -------
    Ed25519PrivateKey
    """
    _check_crypto_deps()
    return Ed25519PrivateKey.from_private_bytes(key_bytes)


def serialize_public_key(public_key: Ed25519PublicKey) -> bytes:
    """Serialise a public key to raw 32 bytes.

    Parameters
    ----------
    public_key : Ed25519PublicKey
        The public key to serialise.

    Returns
    -------
    bytes
        Raw 32-byte public key.
    """
    return public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)


def deserialize_public_key(key_bytes: bytes) -> Ed25519PublicKey:
    """Deserialise a public key from raw 32 bytes.

    Parameters
    ----------
    key_bytes : bytes
        Raw 32-byte public key.

    Returns
    -------
    Ed25519PublicKey
    """
    _check_crypto_deps()
    return Ed25519PublicKey.from_public_bytes(key_bytes)


# ── Signing & verification ──────────────────────────────────────


def sign_message(private_key: Ed25519PrivateKey, message: bytes) -> bytes:
    """Sign a message with an Ed25519 private key.

    Parameters
    ----------
    private_key : Ed25519PrivateKey
        The signing key.
    message : bytes
        The message to sign.

    Returns
    -------
    bytes
        The 64-byte Ed25519 signature.
    """
    return private_key.sign(message)


def verify_signature(
    public_key: Ed25519PublicKey, signature: bytes, message: bytes,
) -> bool:
    """Verify an Ed25519 signature.

    Parameters
    ----------
    public_key : Ed25519PublicKey
        The verification key.
    signature : bytes
        The 64-byte signature to verify.
    message : bytes
        The original message.

    Returns
    -------
    bool
        ``True`` if the signature is valid.
    """
    try:
        public_key.verify(signature, message)
        return True
    except Exception:
        return False


# ── DID:key encoding ────────────────────────────────────────────


def public_key_to_did_key(public_key: Ed25519PublicKey) -> str:
    """Convert an Ed25519 public key to a ``did:key`` identifier.

    Format: ``did:key:z<base58btc(multicodec_prefix + raw_public_key)>``

    Parameters
    ----------
    public_key : Ed25519PublicKey
        The public key to encode.

    Returns
    -------
    str
        The ``did:key`` identifier string.
    """
    raw_bytes: bytes = serialize_public_key(public_key)
    multicodec_bytes: bytes = ED25519_MULTICODEC_PREFIX + raw_bytes
    encoded: str = base58.b58encode(multicodec_bytes).decode("ascii")
    return f"did:key:z{encoded}"


def did_key_to_public_key(did: str) -> Ed25519PublicKey:
    """Extract an Ed25519 public key from a ``did:key`` identifier.

    Parameters
    ----------
    did : str
        A ``did:key:z...`` identifier string.

    Returns
    -------
    Ed25519PublicKey
        The decoded public key.

    Raises
    ------
    ValueError
        If the DID format is invalid or uses a non-Ed25519 multicodec.
    """
    _check_crypto_deps()
    if not did.startswith("did:key:z"):
        raise ValueError(f"Invalid did:key format: {did}")

    encoded: str = did[len("did:key:z"):]
    decoded: bytes = base58.b58decode(encoded)

    if decoded[:2] != ED25519_MULTICODEC_PREFIX:
        raise ValueError("Not an Ed25519 did:key (wrong multicodec prefix)")

    return deserialize_public_key(decoded[2:])


def did_key_fragment(did: str) -> str:
    """Return the ``did:key`` verification method fragment.

    Per the did:key spec, the fragment is the multibase-encoded public
    key portion (the ``z...`` part after ``did:key:``).

    Parameters
    ----------
    did : str
        A ``did:key:z...`` identifier string.

    Returns
    -------
    str
        The fragment, e.g. ``#z6Mk...``.

    Raises
    ------
    ValueError
        If the DID format is invalid.
    """
    if not did.startswith("did:key:z"):
        raise ValueError(f"Invalid did:key format: {did}")
    multibase: str = did[len("did:key:"):]
    return f"#{multibase}"


# ── Signing key persistence ─────────────────────────────────────


def load_or_create_signing_key(
    key_file: str | Path,
    *,
    encryption_password: str | None = None,
) -> tuple[Ed25519PrivateKey, str]:
    """Load a signing key from file or create a new one.

    When *encryption_password* is provided, the private key is stored
    encrypted using Fernet (PBKDF2-SHA256 derived key).  Otherwise
    falls back to base64 plaintext.

    Parameters
    ----------
    key_file : str or Path
        Path to the JSON key file.
    encryption_password : str, optional
        Password for Fernet-encrypting the key at rest.

    Returns
    -------
    tuple[Ed25519PrivateKey, str]
        The private key and its ``did:key`` identifier.
    """
    _check_crypto_deps()
    key_path = Path(key_file)
    fernet_key: bytes | None = _derive_fernet_key(encryption_password)

    if key_path.exists():
        try:
            data: dict[str, Any] = json.loads(key_path.read_text())

            if "private_key_encrypted" in data and fernet_key:
                from cryptography.fernet import Fernet

                cipher = Fernet(fernet_key)
                priv_bytes: bytes = cipher.decrypt(
                    data["private_key_encrypted"].encode("utf-8")
                )
            else:
                priv_bytes = base64.b64decode(data["private_key_b64"])

            private_key: Ed25519PrivateKey = deserialize_private_key(priv_bytes)
            did: str = data["did_key"]
            return private_key, did
        except Exception:
            logger.warning(
                "Corrupted signing key file %s — generating new key.", key_path,
            )

    # Generate new keypair.
    private_key, public_key = generate_ed25519_keypair()
    did = public_key_to_did_key(public_key)

    raw_bytes: bytes = serialize_private_key(private_key)
    key_data: dict[str, Any] = {
        "did_key": did,
        "algorithm": "Ed25519",
    }

    if fernet_key:
        from cryptography.fernet import Fernet

        cipher = Fernet(fernet_key)
        key_data["private_key_encrypted"] = cipher.encrypt(raw_bytes).decode("utf-8")
        key_data["encryption"] = "fernet-pbkdf2-sha256"
    else:
        key_data["private_key_b64"] = base64.b64encode(raw_bytes).decode("ascii")

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key_data, indent=2))
    return private_key, did


def _derive_fernet_key(password: str | None) -> bytes | None:
    """Derive a Fernet encryption key from a password.

    Parameters
    ----------
    password : str or None
        The password.  Returns ``None`` when no password is provided.

    Returns
    -------
    bytes or None
        The derived Fernet key, or ``None``.
    """
    if not password:
        return None
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        salt: bytes = b"orxhestra-signing-key-v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    except ImportError:
        return None


# ── JSON canonicalisation (RFC 8785 subset) ─────────────────────


def _normalize_for_signing(obj: Any) -> Any:
    """Recursively normalise values for deterministic JSON serialisation.

    Applies NFC Unicode normalisation for strings and ensures consistent
    number representation.
    """
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if obj == int(obj) and not (obj == 0.0 and str(obj).startswith("-")):
            return int(obj)
        return obj
    if isinstance(obj, dict):
        return {
            _normalize_for_signing(k): _normalize_for_signing(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_normalize_for_signing(x) for x in obj]
    return obj


def canonicalize_json(payload: dict[str, Any]) -> bytes:
    """Produce a canonical JSON byte string following RFC 8785 (JCS).

    Uses sorted keys, compact separators, NFC normalisation, and
    UTF-8 encoding.

    Parameters
    ----------
    payload : dict[str, Any]
        The JSON-serialisable payload.

    Returns
    -------
    bytes
        The canonical UTF-8 encoded JSON.
    """
    normalised: Any = _normalize_for_signing(payload)
    canonical: str = json.dumps(
        normalised, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )
    return canonical.encode("utf-8")


def sign_json_payload(
    private_key: Ed25519PrivateKey, payload: dict[str, Any],
) -> str:
    """Sign a JSON payload and return a base64url signature.

    The payload is first canonicalised using :func:`canonicalize_json`
    before signing.

    Parameters
    ----------
    private_key : Ed25519PrivateKey
        The signing key.
    payload : dict[str, Any]
        The JSON payload to sign.

    Returns
    -------
    str
        Base64url-encoded signature string.
    """
    canonical_bytes: bytes = canonicalize_json(payload)
    sig_bytes: bytes = sign_message(private_key, canonical_bytes)
    return base64.urlsafe_b64encode(sig_bytes).decode("ascii")


def verify_json_signature(
    public_key: Ed25519PublicKey,
    payload: dict[str, Any],
    signature_b64: str,
) -> bool:
    """Verify a JSON payload signature.

    Parameters
    ----------
    public_key : Ed25519PublicKey
        The verification key.
    payload : dict[str, Any]
        The JSON payload that was signed.
    signature_b64 : str
        The base64url-encoded signature to verify.

    Returns
    -------
    bool
        ``True`` if the signature is valid.
    """
    canonical_bytes: bytes = canonicalize_json(payload)
    sig_bytes: bytes = base64.urlsafe_b64decode(signature_b64)
    return verify_signature(public_key, sig_bytes, canonical_bytes)
