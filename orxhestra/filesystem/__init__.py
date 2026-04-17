"""Filesystem backend protocol — pluggable file access for agents.

Provides a protocol-based abstraction over filesystem operations so
agents can run against the real filesystem, an in-memory filesystem
(for tests), or a future sandbox backend, without code changes.

Usage::

    from orxhestra.filesystem import (
        FilesystemBackend,
        InMemoryFilesystemBackend,
        LocalFilesystemBackend,
    )

    fs: FilesystemBackend = InMemoryFilesystemBackend()
    await fs.write("notes.md", "hello")
    content = await fs.read("notes.md")

The existing :func:`orxhestra.tools.make_filesystem_tools` continues
to work unchanged — this package is purely additive.
"""

from __future__ import annotations

from orxhestra.filesystem.base import FilesystemBackend, GrepMatch
from orxhestra.filesystem.local import LocalFilesystemBackend
from orxhestra.filesystem.memory import InMemoryFilesystemBackend

__all__ = [
    "FilesystemBackend",
    "GrepMatch",
    "LocalFilesystemBackend",
    "InMemoryFilesystemBackend",
]
