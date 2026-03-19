"""Composer error types."""

from __future__ import annotations


class ComposerError(Exception):
    """Base error for composer operations."""


class CircularReferenceError(ComposerError):
    """Raised when agents form a circular dependency."""
