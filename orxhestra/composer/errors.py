"""Composer error types — the public exception surface.

Every failure path inside the composer — schema validation, YAML
parsing, agent resolution, tool wiring, identity loading — ends up
as one of these.  Callers can ``except ComposerError:`` to catch any
composer-level failure without reaching for Pydantic's
``ValidationError`` (which is wrapped at the boundary).

:class:`CircularReferenceError` is a subclass, raised specifically
when an agent's ``agents:`` / ``transfer:`` / ``tools: { agent: }``
chain points back at an ancestor mid-build.

See Also
--------
orxhestra.composer.composer.Composer : Primary raiser.
"""

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
