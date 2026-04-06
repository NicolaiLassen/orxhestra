"""Token type detection and JWT parsing.

Identifies whether an input string is a JWT, DID, URL, API key, or
unknown, and extracts claims from JWTs without signature verification
(useful for identity bridging and token classification).

Requires the ``PyJWT`` package for JWT decoding.  Install via::

    pip install orxhestra[auth]
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any

logger: logging.Logger = logging.getLogger(__name__)

try:
    import jwt

    _HAS_PYJWT: bool = True
except ImportError:  # pragma: no cover
    _HAS_PYJWT = False


def _check_jwt_deps() -> None:
    """Raise ``ImportError`` if PyJWT is not installed."""
    if not _HAS_PYJWT:
        raise ImportError(
            "PyJWT is required for JWT parsing. "
            "Install it with: pip install orxhestra[auth]"
        )


class TokenType(str, Enum):
    """Classification of an identity token string.

    Attributes
    ----------
    JWT : str
        JSON Web Token (three base64url segments).
    DID : str
        Decentralized Identifier (``did:<method>:<id>``).
    URL : str
        HTTP(S) URL (e.g. an A2A Agent Card endpoint).
    API_KEY : str
        Opaque API key (hex or mixed-case alphanumeric, >= 32 chars).
    UNKNOWN : str
        Unrecognised token format.
    """

    JWT = "jwt"
    DID = "did"
    URL = "url"
    API_KEY = "api_key"
    UNKNOWN = "unknown"


# Regex patterns for token type detection.
DID_PATTERN: re.Pattern[str] = re.compile(r"^did:[a-z0-9]+:.+$", re.IGNORECASE)
URL_PATTERN: re.Pattern[str] = re.compile(r"^https?://.+$", re.IGNORECASE)
JWT_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$"
)
API_KEY_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Fa-f0-9]{32,}$|^(?=.*[A-Z])(?=.*[a-z0-9])[A-Za-z0-9_-]{32,}$"
)


def detect_token_type(token: str) -> TokenType:
    """Detect the type of an identity token string.

    Uses regex matching and, for JWT candidates, attempts a decode
    (without signature verification) to confirm validity.

    Parameters
    ----------
    token : str
        The raw token string.

    Returns
    -------
    TokenType
        The detected token type.
    """
    token = token.strip()

    if DID_PATTERN.match(token):
        return TokenType.DID

    if JWT_PATTERN.match(token) and _HAS_PYJWT:
        try:
            jwt.decode(token, options={"verify_signature": False})
            return TokenType.JWT
        except Exception:
            pass

    if URL_PATTERN.match(token):
        return TokenType.URL

    if API_KEY_PATTERN.match(token) and not DID_PATTERN.match(token):
        return TokenType.API_KEY

    return TokenType.UNKNOWN


def parse_jwt_claims(token: str) -> dict[str, Any] | None:
    """Parse a JWT without verification and extract claims.

    Parameters
    ----------
    token : str
        The raw JWT string.

    Returns
    -------
    dict[str, Any] or None
        A dictionary with ``header``, ``claims``, ``subject``,
        ``issuer``, ``audience``, ``expiry``, ``issued_at``, and
        ``scopes`` keys, or ``None`` if the token cannot be decoded.

    Raises
    ------
    ImportError
        If ``PyJWT`` is not installed.
    """
    _check_jwt_deps()
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
        )
        header: dict[str, Any] = jwt.get_unverified_header(token)
        return {
            "header": header,
            "claims": claims,
            "subject": claims.get("sub"),
            "issuer": claims.get("iss"),
            "audience": claims.get("aud"),
            "expiry": claims.get("exp"),
            "issued_at": claims.get("iat"),
            "scopes": claims.get("scope", "").split() if claims.get("scope") else [],
        }
    except Exception:
        logger.warning("Failed to parse JWT token.")
        return None


def extract_identity_from_token(token: str) -> dict[str, Any]:
    """Extract identity information from any supported token type.

    Detects the token type and extracts the relevant fields.  For
    JWTs this includes subject, issuer, scopes, and expiry.  For DIDs
    it extracts the method and specific identifier.

    Parameters
    ----------
    token : str
        The raw token string.

    Returns
    -------
    dict[str, Any]
        A dictionary with ``token_type``, ``original_token``, and
        type-specific fields.
    """
    token_type: TokenType = detect_token_type(token)
    result: dict[str, Any] = {
        "token_type": token_type.value,
        "original_token": token,
    }

    if token_type == TokenType.JWT:
        parsed: dict[str, Any] | None = parse_jwt_claims(token)
        if parsed:
            result["subject"] = parsed["subject"]
            result["issuer"] = parsed["issuer"]
            result["scopes"] = parsed["scopes"]
            result["expiry"] = parsed["expiry"]
            result["jwt_header"] = parsed["header"]

    elif token_type == TokenType.DID:
        parts: list[str] = token.split(":")
        result["did_method"] = parts[1] if len(parts) >= 3 else "unknown"
        result["did_specific_id"] = ":".join(parts[2:]) if len(parts) >= 3 else token

    elif token_type == TokenType.URL:
        result["url"] = token
        result["note"] = "URL-based identity (e.g., A2A Agent Card endpoint)"

    elif token_type == TokenType.API_KEY:
        result["key_preview"] = (
            token[:6] + "..." + token[-4:] if len(token) > 12 else "***"
        )
        result["key_length"] = len(token)

    return result
