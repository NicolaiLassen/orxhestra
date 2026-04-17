"""Filesystem backend protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class GrepMatch:
    """A single grep hit.

    Attributes
    ----------
    path : str
        File path relative to the backend root.
    line_no : int
        1-indexed line number.
    line : str
        The matching line (without trailing newline).
    """

    path: str
    line_no: int
    line: str


@runtime_checkable
class FilesystemBackend(Protocol):
    """Async filesystem operations.

    All implementations must be safe for concurrent use by a single
    agent invocation. Path semantics are backend-specific — the local
    backend enforces a workspace jail, the in-memory backend treats
    every path as a key.

    See Also
    --------
    LocalFilesystemBackend : Real-disk implementation.
    InMemoryFilesystemBackend : Dict-backed implementation for tests.
    GrepMatch : Return type of :meth:`grep`.

    Examples
    --------
    >>> from orxhestra.filesystem import InMemoryFilesystemBackend
    >>> fs: FilesystemBackend = InMemoryFilesystemBackend()
    >>> await fs.write("notes/todo.md", "- buy milk\\n- write docs\\n")
    >>> await fs.read("notes/todo.md")
    '- buy milk\\n- write docs\\n'
    >>> [m.line for m in await fs.grep("docs")]
    ['- write docs']
    """

    async def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read ``limit`` lines starting at ``offset``.

        Parameters
        ----------
        path : str
            File path.
        offset : int
            1-indexed line offset. ``0`` means start at line 1.
        limit : int, optional
            Maximum lines to return. ``None`` = read all.
        """
        ...

    async def write(self, path: str, content: str) -> None:
        """Replace file contents with ``content`` (creates parents)."""
        ...

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Replace occurrences of ``old`` with ``new``.

        Returns
        -------
        int
            Number of replacements made.
        """
        ...

    async def ls(self, path: str = ".") -> list[str]:
        """List immediate children of ``path``."""
        ...

    async def glob(
        self, pattern: str, *, path: str | None = None,
    ) -> list[str]:
        """Glob paths matching ``pattern`` under ``path`` (default root)."""
        ...

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch]:
        """Regex-search file contents. Optional ``glob`` filter."""
        ...

    async def mkdir(self, path: str, *, exist_ok: bool = True) -> None:
        """Create directory at ``path``."""
        ...

    async def delete(self, path: str) -> None:
        """Delete file or (empty) directory at ``path``."""
        ...

    async def exists(self, path: str) -> bool:
        """Return True if ``path`` exists."""
        ...
