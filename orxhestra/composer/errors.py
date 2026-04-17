"""Composer error types."""

from __future__ import annotations


class ComposerError(Exception):
    """Base error for composer operations.

    See Also
    --------
    Composer : Raises subclasses of this during YAML resolution.
    CircularReferenceError : Specific subclass for cyclic sub-agent
        references.
    """


class CircularReferenceError(ComposerError):
    """Raised when agents form a circular dependency.

    See Also
    --------
    Composer.from_yaml_async : Detects cycles while building the
        agent tree.
    """
