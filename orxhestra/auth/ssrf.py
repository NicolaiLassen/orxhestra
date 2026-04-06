"""SSRF protection utilities for orxhestra.

Validates hostnames and resolved IPs against private/reserved ranges
before making outbound HTTP requests.  Returns pinned IPs to prevent
DNS rebinding (TOCTOU) attacks.

All functions are pure-stdlib and have no external dependencies.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger: logging.Logger = logging.getLogger(__name__)

# Domains that are always blocked (case-insensitive).
_BLOCKED_DOMAINS: set[str] = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google.com",
    "169.254.169.254",
}

# Domain suffixes that are always blocked.
_BLOCKED_SUFFIXES: tuple[str, ...] = (
    ".local",
    ".internal",
    ".localhost",
)

# Maximum redirects to follow (0 = no redirects).
MAX_REDIRECTS: int = 0


def _is_private_ip(ip_str: str) -> bool:
    """Check whether an IP address is private, loopback, link-local, or reserved.

    Parameters
    ----------
    ip_str : str
        The IP address string to check.

    Returns
    -------
    bool
        ``True`` if the address is in a non-public range.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
    except ValueError:
        return False


def validate_url_host(hostname: str) -> str | None:
    """Validate that a hostname is safe for outbound requests.

    Blocks private IPs, reserved ranges, localhost variants, cloud
    metadata endpoints, and hostnames that resolve to private IPs.

    Parameters
    ----------
    hostname : str
        The hostname to validate.

    Returns
    -------
    str or None
        ``None`` if the hostname is safe, or an error string
        describing why it was blocked.
    """
    if not hostname:
        return "Empty hostname"

    clean: str = hostname.lower().strip("[]")

    if clean in _BLOCKED_DOMAINS:
        return f"Blocked: private hostname '{hostname}'"

    for suffix in _BLOCKED_SUFFIXES:
        if clean.endswith(suffix):
            return f"Blocked: private domain suffix '{hostname}'"

    # Try to parse as a raw IP address.
    try:
        ip = ipaddress.ip_address(clean)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return f"Blocked: private/reserved IP address '{hostname}'"
    except ValueError:
        pass  # Not a raw IP — continue to DNS resolution.

    # Resolve hostname and check ALL resolved IPs.
    try:
        resolved = socket.getaddrinfo(clean, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _, _, _, addr in resolved:
            ip_str: str = addr[0]
            if _is_private_ip(ip_str):
                return f"Blocked: '{hostname}' resolves to private IP {ip_str}"
    except socket.gaierror:
        pass  # DNS resolution failed — let the HTTP client handle it.

    return None


def validate_and_pin_url(url: str) -> tuple[str | None, list[str]]:
    """Validate a URL and return pinned resolved IPs.

    Resolves DNS once during validation and returns the IPs so the
    caller can connect directly to them, preventing DNS rebinding
    (TOCTOU) attacks.

    Parameters
    ----------
    url : str
        The full URL to validate.

    Returns
    -------
    tuple[str | None, list[str]]
        A two-element tuple of ``(error_or_none, pinned_ips)``.
        *error_or_none* is ``None`` when safe, or a descriptive
        error string when blocked.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return ("Invalid URL", [])

    hostname: str | None = parsed.hostname
    if not hostname:
        return ("No hostname in URL", [])

    clean: str = hostname.lower().strip("[]")

    if clean in _BLOCKED_DOMAINS:
        return (f"Blocked: private hostname '{hostname}'", [])

    for suffix in _BLOCKED_SUFFIXES:
        if clean.endswith(suffix):
            return (f"Blocked: private domain suffix '{hostname}'", [])

    # Try to parse as a raw IP address.
    try:
        ip = ipaddress.ip_address(clean)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return (f"Blocked: private/reserved IP address '{hostname}'", [])
        return (None, [str(ip)])
    except ValueError:
        pass  # Not a raw IP — resolve via DNS.

    # Resolve hostname and pin all safe IPs.
    safe_ips: list[str] = []
    try:
        resolved = socket.getaddrinfo(clean, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _, _, _, addr in resolved:
            ip_str: str = addr[0]
            if _is_private_ip(ip_str):
                return (f"Blocked: '{hostname}' resolves to private IP {ip_str}", [])
            if ip_str not in safe_ips:
                safe_ips.append(ip_str)
    except socket.gaierror:
        pass  # DNS resolution failed — let the HTTP client handle it.

    return (None, safe_ips)


def validate_redirect_target(redirect_url: str, original_hostname: str) -> str | None:
    """Validate a redirect target URL.

    Blocks redirects to private IPs or otherwise unsafe hosts.

    Parameters
    ----------
    redirect_url : str
        The redirect destination URL.
    original_hostname : str
        The hostname of the original request (currently unused but
        reserved for future same-origin checks).

    Returns
    -------
    str or None
        ``None`` if safe, or an error string if blocked.
    """
    try:
        parsed = urlparse(redirect_url)
    except Exception:
        return "Invalid redirect URL"

    redirect_host: str | None = parsed.hostname
    if not redirect_host:
        return "No hostname in redirect URL"

    error: str | None = validate_url_host(redirect_host)
    if error:
        return f"Redirect blocked: {error}"

    return None
