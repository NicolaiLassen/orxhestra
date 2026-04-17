"""Local filesystem backend — workspace-jailed operations on disk."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from orxhestra.filesystem.base import FilesystemBackend, GrepMatch

_MAX_GLOB_RESULTS = 500
_MAX_GREP_RESULTS = 200


class LocalFilesystemBackend(FilesystemBackend):
    """Real-disk backend jailed to a workspace directory.

    Every operation resolves paths through :meth:`_resolve`, which
    rejects any target outside ``workspace`` (even via absolute paths
    or ``..`` segments).

    Parameters
    ----------
    workspace : str or Path
        Root directory. All operations are confined to this tree.
        The directory is created on first use.

    See Also
    --------
    FilesystemBackend : The protocol this implements.
    InMemoryFilesystemBackend : Dict-backed alternative for tests.

    Examples
    --------
    >>> fs = LocalFilesystemBackend("/tmp/agent-workspace")
    >>> await fs.write("report.md", "# Findings\\n")
    >>> files = await fs.glob("**/*.md")
    """

    def __init__(self, workspace: str | Path) -> None:
        self._workspace = Path(workspace).resolve()

    @property
    def workspace(self) -> Path:
        """The jailed workspace root."""
        return self._workspace

    def _resolve(self, path: str) -> Path:
        """Resolve ``path`` inside the workspace, rejecting escapes.

        Parameters
        ----------
        path : str
            Relative or absolute path.

        Returns
        -------
        Path
            Absolute path inside the workspace.

        Raises
        ------
        ValueError
            If the resolved path is outside the workspace.
        """
        ws = self._workspace
        ws.mkdir(parents=True, exist_ok=True)
        target = (ws / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if not str(target).startswith(str(ws)):
            msg = f"Path '{path}' is outside the workspace"
            raise ValueError(msg)
        return target

    async def read(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read lines from a file on disk. See :class:`FilesystemBackend`."""
        target = self._resolve(path)
        if not target.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        if not target.is_file():
            msg = f"Not a file: {path}"
            raise IsADirectoryError(msg)
        with target.open(encoding="utf-8") as f:
            lines = f.readlines()
        start = max(offset, 0)
        end = len(lines) if limit is None else start + max(limit, 0)
        return "".join(lines[start:end])

    async def write(self, path: str, content: str) -> None:
        """Write a file on disk. See :class:`FilesystemBackend`."""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        """Replace text in a file. See :class:`FilesystemBackend`."""
        target = self._resolve(path)
        if not target.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        content = target.read_text(encoding="utf-8")
        if replace_all:
            new_content = content.replace(old, new)
            count = content.count(old)
        else:
            if content.count(old) > 1:
                msg = (
                    f"String appears {content.count(old)} times in "
                    f"{path}. Use replace_all=True or provide more context."
                )
                raise ValueError(msg)
            new_content = content.replace(old, new, 1)
            count = 1 if old in content else 0
        if count == 0:
            msg = f"String not found in {path}"
            raise ValueError(msg)
        target.write_text(new_content, encoding="utf-8")
        return count

    async def ls(self, path: str = ".") -> list[str]:
        """List children of a path. See :class:`FilesystemBackend`."""
        target = self._resolve(path)
        if not target.exists():
            return []
        if target.is_file():
            return [target.name]
        return sorted(p.name for p in target.iterdir())

    async def glob(
        self, pattern: str, *, path: str | None = None,
    ) -> list[str]:
        """Glob files. See :class:`FilesystemBackend`."""
        root = self._resolve(path) if path else self._workspace
        results: list[str] = []
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(self._workspace).as_posix()
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                results.append(rel)
                if len(results) >= _MAX_GLOB_RESULTS:
                    break
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
        root = self._resolve(path) if path else self._workspace
        matches: list[GrepMatch] = []
        candidates = (
            [p for p in root.rglob("*") if p.is_file()]
            if root.is_dir() else [root] if root.is_file() else []
        )
        for p in candidates:
            rel = p.relative_to(self._workspace).as_posix()
            if glob and not (
                fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(p.name, glob)
            ):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(GrepMatch(path=rel, line_no=i, line=line))
                    if len(matches) >= _MAX_GREP_RESULTS:
                        return matches
        return matches

    async def mkdir(self, path: str, *, exist_ok: bool = True) -> None:
        """Create a directory. See :class:`FilesystemBackend`."""
        target = self._resolve(path)
        target.mkdir(parents=True, exist_ok=exist_ok)

    async def delete(self, path: str) -> None:
        """Delete a file or empty directory. See :class:`FilesystemBackend`."""
        target = self._resolve(path)
        if not target.exists():
            return
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()

    async def exists(self, path: str) -> bool:
        """Return True if ``path`` exists. See :class:`FilesystemBackend`."""
        try:
            return self._resolve(path).exists()
        except ValueError:
            return False
