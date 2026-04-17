"""In-memory filesystem backend — for tests, sandboxes, and ephemeral state."""

from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

from orxhestra.filesystem.base import FilesystemBackend, GrepMatch


def _normalize(path: str) -> str:
    """Normalize a path by collapsing ``.`` and ``..`` segments.

    Parameters
    ----------
    path : str
        Input path.

    Returns
    -------
    str
        Normalized POSIX-style path with no leading slash and no
        ``.``/``..`` segments. The root is rendered as ``.``.
    """
    p = PurePosixPath(path)
    parts: list[str] = []
    for part in p.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts) or "."


class InMemoryFilesystemBackend(FilesystemBackend):
    """Dict-backed filesystem for tests and sandboxing.

    Directories exist implicitly — any path with a file under it
    creates the directory structure in :meth:`ls`. Explicit
    :meth:`mkdir` is supported for empty directories. Paths are
    normalized via :func:`_normalize` so ``.`` and ``..`` segments
    collapse without ever touching the real disk.

    Parameters
    ----------
    initial : dict[str, str], optional
        Seed files. Keys are paths, values are UTF-8 string contents.

    See Also
    --------
    FilesystemBackend : The protocol this implements.
    LocalFilesystemBackend : Real-disk alternative.

    Examples
    --------
    >>> fs = InMemoryFilesystemBackend({"src/main.py": "print('hi')\\n"})
    >>> await fs.ls("src")
    ['main.py']
    >>> await fs.edit("src/main.py", "hi", "hello")
    1
    """

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = {}
        self._dirs: set[str] = set()
        if initial:
            for k, v in initial.items():
                self._files[_normalize(k)] = v

    async def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read lines. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        if key not in self._files:
            if key in self._dirs or self._is_implicit_dir(key):
                msg = f"Not a file: {path}"
                raise IsADirectoryError(msg)
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        content = self._files[key]
        lines = content.splitlines(keepends=True)
        start = max(offset, 0)
        end = len(lines) if limit is None else start + max(limit, 0)
        return "".join(lines[start:end])

    async def write(self, path: str, content: str) -> None:
        """Write a file. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        self._files[key] = content

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Edit a file. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        if key not in self._files:
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        content = self._files[key]
        if replace_all:
            count = content.count(old)
            new_content = content.replace(old, new)
        else:
            occurrences = content.count(old)
            if occurrences > 1:
                msg = (
                    f"String appears {occurrences} times in {path}. "
                    "Use replace_all=True or provide more context."
                )
                raise ValueError(msg)
            new_content = content.replace(old, new, 1)
            count = 1 if old in content else 0
        if count == 0:
            msg = f"String not found in {path}"
            raise ValueError(msg)
        self._files[key] = new_content
        return count

    async def ls(self, path: str = ".") -> list[str]:
        """List children. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        if key == "." or key == "":
            prefix = ""
        else:
            prefix = key + "/"
        names: set[str] = set()
        for p in self._files:
            if prefix and not p.startswith(prefix):
                continue
            rest = p[len(prefix):]
            if "/" in rest:
                names.add(rest.split("/", 1)[0])
            elif rest:
                names.add(rest)
        for d in self._dirs:
            if prefix and not d.startswith(prefix):
                continue
            rest = d[len(prefix):]
            if rest and "/" not in rest:
                names.add(rest)
        return sorted(names)

    async def glob(
        self, pattern: str, *, path: str | None = None,
    ) -> list[str]:
        """Glob files. See :class:`FilesystemBackend`."""
        prefix = _normalize(path) + "/" if path else ""
        results = [
            p for p in self._files
            if (not prefix or p.startswith(prefix))
            and (fnmatch.fnmatch(p, pattern) or fnmatch.fnmatch(p.rsplit("/", 1)[-1], pattern))
        ]
        return sorted(results)

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob: str | None = None,
    ) -> list[GrepMatch]:
        """Regex-search files. See :class:`FilesystemBackend`."""
        regex = re.compile(pattern)
        prefix = _normalize(path) + "/" if path else ""
        matches: list[GrepMatch] = []
        for p, content in self._files.items():
            if prefix and not p.startswith(prefix):
                continue
            if glob and not (
                fnmatch.fnmatch(p, glob)
                or fnmatch.fnmatch(p.rsplit("/", 1)[-1], glob)
            ):
                continue
            for i, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(GrepMatch(path=p, line_no=i, line=line))
        return matches

    async def mkdir(self, path: str, *, exist_ok: bool = True) -> None:
        """Create a directory. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        if key in self._files:
            msg = f"Path exists as a file: {path}"
            raise FileExistsError(msg)
        if key in self._dirs and not exist_ok:
            msg = f"Directory exists: {path}"
            raise FileExistsError(msg)
        self._dirs.add(key)

    async def delete(self, path: str) -> None:
        """Delete a file or empty directory. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        if key in self._files:
            del self._files[key]
            return
        if key in self._dirs:
            # Only remove if empty.
            if any(
                p.startswith(key + "/") for p in self._files
            ) or any(
                d.startswith(key + "/") for d in self._dirs if d != key
            ):
                msg = f"Directory not empty: {path}"
                raise OSError(msg)
            self._dirs.remove(key)

    async def exists(self, path: str) -> bool:
        """Return True if ``path`` exists. See :class:`FilesystemBackend`."""
        key = _normalize(path)
        if key == ".":
            # Root always exists, even when the backend is empty.
            return True
        return (
            key in self._files
            or key in self._dirs
            or self._is_implicit_dir(key)
        )

    def _is_implicit_dir(self, key: str) -> bool:
        """Return True if ``key`` is a directory implied by child files.

        Parameters
        ----------
        key : str
            Normalized path.

        Returns
        -------
        bool
            True if any stored file path starts with ``key + "/"``.
        """
        prefix = key + "/"
        return any(p.startswith(prefix) for p in self._files)
