"""Filesystem tools for agent file operations.

Provides sandboxed file system access: list, read, write, edit, mkdir,
glob, and grep. All operations route through a
:class:`FilesystemBackend` — the default is
:class:`LocalFilesystemBackend` (workspace-jailed pathlib) but any
backend can be swapped in for tests, sandboxes, or remote execution.

The tool factory keeps the rich LLM-friendly output formatting
(line-numbered reads, diff summaries, context-line grep) on top of
whatever backend is provided.

See Also
--------
FilesystemBackend : Protocol the backend must satisfy.
LocalFilesystemBackend : Default backend — real disk, workspace-jailed.
InMemoryFilesystemBackend : Dict-backed alternative for tests.
orxhestra.filesystem.make_tools : Lower-level factory that maps each
    backend method 1:1 to a tool. Use when the rich formatting here
    isn't needed.
"""

from __future__ import annotations

import fnmatch
import os
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool

from orxhestra.filesystem.local import LocalFilesystemBackend

if TYPE_CHECKING:
    from orxhestra.filesystem.base import FilesystemBackend

# Maximum file size (bytes) before requiring offset/limit.
_MAX_FILE_SIZE: int = 256 * 1024  # 256 KB

# Default number of lines to read per call.
_DEFAULT_LINE_LIMIT: int = 2000

# Maximum results returned by glob and grep.
_MAX_GLOB_RESULTS: int = 100
_MAX_GREP_RESULTS: int = 50


def _default_workspace() -> str:
    """Resolve workspace at call time, not import time."""
    return os.environ.get("AGENT_WORKSPACE", "/tmp/agent-workspace")


def _add_line_numbers(text: str, offset: int = 0) -> str:
    """Add line numbers to text (1-based, tab-separated)."""
    lines = text.splitlines()
    numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def make_filesystem_tools(
    workspace: str | None = None,
    *,
    backend: FilesystemBackend | None = None,
) -> list[BaseTool]:
    """Create filesystem tools backed by a :class:`FilesystemBackend`.

    Defaults to a :class:`LocalFilesystemBackend` rooted at ``workspace``
    (or ``$AGENT_WORKSPACE`` / ``/tmp/agent-workspace``), which matches
    the pathlib-on-disk behavior of earlier versions. Pass ``backend=``
    to swap in any other implementation — for example
    :class:`InMemoryFilesystemBackend` for tests.

    Parameters
    ----------
    workspace : str, optional
        Root directory for the default local backend. Ignored when
        ``backend`` is provided.
    backend : FilesystemBackend, optional
        Pre-built backend. Takes precedence over ``workspace``.

    Returns
    -------
    list[BaseTool]
        Seven tools: ``ls``, ``read_file``, ``write_file``,
        ``edit_file``, ``mkdir``, ``glob``, ``grep``.

    Raises
    ------
    ValueError
        If both ``workspace`` and ``backend`` are provided.

    See Also
    --------
    FilesystemBackend : Protocol the backend must satisfy.
    LocalFilesystemBackend : Default backend.
    InMemoryFilesystemBackend : Dict-backed alternative for tests.
    orxhestra.filesystem.make_tools : Lower-level factory without the
        rich formatting layer.

    Examples
    --------
    Default local backend:

    >>> tools = make_filesystem_tools(workspace="/tmp/agent-ws")

    In-memory backend for tests:

    >>> from orxhestra.filesystem import InMemoryFilesystemBackend
    >>> fs = InMemoryFilesystemBackend({"src/main.py": "print('hi')\\n"})
    >>> tools = make_filesystem_tools(backend=fs)
    """
    if backend is not None and workspace is not None:
        msg = "Pass either 'workspace' or 'backend', not both."
        raise ValueError(msg)

    if backend is None:
        ws: str = workspace or _default_workspace()
        backend = LocalFilesystemBackend(ws)

    fs: FilesystemBackend = backend

    async def ls(path: str = ".") -> str:
        """List files and directories at the given path."""
        if not await fs.exists(path):
            return f"Error: '{path}' does not exist"
        names = await fs.ls(path)
        if not names:
            return "(empty directory)"
        return "\n".join(names)

    async def read_file(
        path: str,
        offset: int = 0,
        limit: int = _DEFAULT_LINE_LIMIT,
    ) -> str:
        """Read a file, returning numbered lines.

        Parameters
        ----------
        path : str
            File path relative to workspace.
        offset : int
            Line number to start reading from (0 = beginning).
        limit : int
            Max number of lines to read. Default 2000.
        """
        if not await fs.exists(path):
            return f"Error: '{path}' does not exist"

        # Full raw content first (backends are expected to stream
        # large files efficiently via offset/limit, but we need the
        # total count for the pagination header).
        try:
            full = await fs.read(path)
        except IsADirectoryError:
            return f"Error: '{path}' is not a file"
        except FileNotFoundError:
            return f"Error: '{path}' does not exist"

        # Guard against huge files without explicit offset/limit.
        byte_size = len(full.encode("utf-8"))
        if (
            byte_size > _MAX_FILE_SIZE
            and offset == 0
            and limit == _DEFAULT_LINE_LIMIT
        ):
            size_kb = byte_size // 1024
            return (
                f"Error: File is {size_kb} KB ({byte_size:,} bytes). "
                f"Use 'offset' and 'limit' parameters to read specific "
                f"portions, or use 'grep' to search for specific content."
            )

        all_lines = full.splitlines()
        total = len(all_lines)
        start = min(offset, total)
        end = min(start + limit, total)
        selected = all_lines[start:end]

        content = _add_line_numbers("\n".join(selected), offset=start)
        header = f"Lines {start + 1}-{end} of {total}"
        if end < total:
            header += f" (use offset={end} to read more)"
        return f"{header}\n{content}"

    async def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed.

        Parameters
        ----------
        path : str
            Path relative to the workspace.
        content : str
            Full file contents. Overwrites any existing file.

        Returns
        -------
        str
            Confirmation message including byte count.
        """
        await fs.write(path, content)
        return f"Wrote {len(content)} characters to {path}"

    async def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of ``old`` with ``new`` in a file.

        Parameters
        ----------
        path : str
            Path relative to the workspace.
        old : str
            Exact substring to replace. Must occur in the file.
        new : str
            Replacement text.

        Returns
        -------
        str
            A compact diff summary showing the removed and added lines,
            or an error message if the file or substring is missing.
        """
        if not await fs.exists(path):
            return f"Error: '{path}' does not exist"
        try:
            text = await fs.read(path)
        except (FileNotFoundError, IsADirectoryError):
            return f"Error: '{path}' does not exist or is not a file"
        if old not in text:
            return f"Error: '{old}' not found in {path}"

        # Compute edit context before mutating.
        before_lines = text[: text.index(old)].count("\n")
        old_line_count = old.count("\n") + 1
        new_line_count = new.count("\n") + 1

        # First-occurrence replace (matches pre-refactor semantics).
        updated = text.replace(old, new, 1)
        await fs.write(path, updated)

        start_line = before_lines + 1
        diff_lines: list[str] = [f"Edited {path} (line {start_line}):"]
        for line in old.splitlines():
            diff_lines.append(f"- {line}")
        for line in new.splitlines():
            diff_lines.append(f"+ {line}")
        diff_lines.append(
            f"({old_line_count} lines removed, "
            f"{new_line_count} lines added)"
        )
        return "\n".join(diff_lines)

    async def mkdir(path: str) -> str:
        """Create a directory and any missing parents.

        Parameters
        ----------
        path : str
            Directory path relative to the workspace.

        Returns
        -------
        str
            Confirmation message. Idempotent — no error if the
            directory already exists.
        """
        await fs.mkdir(path, exist_ok=True)
        return f"Created directory {path}"

    async def glob(pattern: str, max_results: int = _MAX_GLOB_RESULTS) -> str:
        """Find files matching a glob pattern (e.g. '**/*.py').

        Parameters
        ----------
        pattern : str
            Glob pattern to match.
        max_results : int
            Maximum number of results. Default 100.
        """
        matches = await fs.glob(pattern)
        if not matches:
            return "(no matches)"
        if len(matches) > max_results:
            head = matches[:max_results]
            head.append(f"(truncated — {max_results} results shown)")
            return "\n".join(head)
        return "\n".join(matches)

    async def grep(
        pattern: str,
        path: str = ".",
        *,
        glob_filter: str = "*",
        context: int = 0,
        before: int = 0,
        after: int = 0,
        case_insensitive: bool = False,
        max_results: int = _MAX_GREP_RESULTS,
    ) -> str:
        """Search file contents for a text pattern.

        Parameters
        ----------
        pattern : str
            Text or regex pattern to search for.
        path : str
            Directory to search in, relative to workspace.
        glob_filter : str
            Only search files matching this glob (e.g. '*.py').
        context : int
            Lines of context before and after each match (-C).
        before : int
            Lines of context before each match (-B).
        after : int
            Lines of context after each match (-A).
        case_insensitive : bool
            Case-insensitive matching.
        max_results : int
            Maximum number of matching lines. Default 50.
        """
        ctx_before = before or context
        ctx_after = after or context
        search_pattern = pattern.lower() if case_insensitive else pattern

        # Enumerate candidate files via the backend. ``*`` matches all
        # paths in both backends because fnmatch's ``*`` crosses ``/``.
        search_path = None if path == "." else path
        candidates = await fs.glob("*", path=search_path)
        candidates = [
            c for c in candidates
            if fnmatch.fnmatch(c.rsplit("/", 1)[-1], glob_filter)
        ]

        results: list[str] = []
        match_count = 0

        for rel in candidates:
            try:
                text = await fs.read(rel)
            except (IsADirectoryError, FileNotFoundError, UnicodeDecodeError):
                continue

            lines = text.splitlines()
            for i, line in enumerate(lines):
                compare = line.lower() if case_insensitive else line
                if search_pattern in compare:
                    start = max(0, i - ctx_before)
                    end = min(len(lines), i + ctx_after + 1)

                    if ctx_before or ctx_after:
                        for j in range(start, end):
                            marker = ">" if j == i else " "
                            results.append(f"{rel}:{j + 1}:{marker} {lines[j]}")
                        results.append("--")
                    else:
                        results.append(f"{rel}:{i + 1}: {line}")

                    match_count += 1
                    if match_count >= max_results:
                        results.append(
                            f"(truncated — {max_results} matches shown)"
                        )
                        break

            if match_count >= max_results:
                break

        return "\n".join(results) if results else "(no matches)"

    return [
        StructuredTool.from_function(
            coroutine=ls,
            name="ls",
            description="List files and directories at a path within the workspace.",
        ),
        StructuredTool.from_function(
            coroutine=read_file,
            name="read_file",
            description=(
                "Read a file with line numbers. Use 'offset' and 'limit' to "
                "read specific portions of large files."
            ),
        ),
        StructuredTool.from_function(
            coroutine=write_file,
            name="write_file",
            description="Write content to a file in the workspace. Creates parent directories.",
        ),
        StructuredTool.from_function(
            coroutine=edit_file,
            name="edit_file",
            description="Replace the first occurrence of 'old' with 'new' in a file.",
        ),
        StructuredTool.from_function(
            coroutine=mkdir,
            name="mkdir",
            description="Create a directory (and parents) in the workspace.",
        ),
        StructuredTool.from_function(
            coroutine=glob,
            name="glob",
            description=(
                "Find files matching a glob pattern "
                "(e.g. '**/*.py', 'src/*.ts'). Max 100 results."
            ),
        ),
        StructuredTool.from_function(
            coroutine=grep,
            name="grep",
            description=(
                "Search file contents for a pattern. Supports context lines "
                "(before/after), case-insensitive matching, and glob filtering."
            ),
        ),
    ]


