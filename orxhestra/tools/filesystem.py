"""Filesystem tools for agent file operations.

Provides sandboxed file system access: list, read, write, edit, mkdir,
glob, and grep. All operations are restricted to a configurable workspace
root directory.

Large file reads are paginated via ``offset`` and ``limit`` parameters.
Output is line-numbered (``cat -n`` style) so the agent can reference
specific locations.
"""

from __future__ import annotations

import base64
import fnmatch
import mimetypes
import os
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

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


def _resolve_path(path: str, workspace: str) -> Path:
    """Resolve and validate a path within the workspace."""
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    target = (ws / path).resolve()
    if not str(target).startswith(str(ws)):
        msg = f"Path '{path}' is outside the workspace"
        raise ValueError(msg)
    return target


def _add_line_numbers(text: str, offset: int = 0) -> str:
    """Add line numbers to text (1-based, tab-separated)."""
    lines = text.splitlines()
    numbered = [f"{offset + i + 1}\t{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


def make_filesystem_tools(workspace: str | None = None) -> list[BaseTool]:
    """Create filesystem tools sandboxed to a workspace directory.

    Parameters
    ----------
    workspace : str, optional
        Root directory for file operations. Defaults to ``$AGENT_WORKSPACE``
        or ``/tmp/agent-workspace``.

    Returns
    -------
    list[BaseTool]
        Seven tools: ``ls``, ``read_file``, ``write_file``, ``edit_file``,
        ``mkdir``, ``glob``, ``grep``.
    """
    ws = workspace or _default_workspace()

    async def ls(path: str = ".") -> str:
        """List files and directories at the given path."""
        target = _resolve_path(path, ws)
        if not target.exists():
            return f"Error: '{path}' does not exist"
        if not target.is_dir():
            return f"Error: '{path}' is not a directory"
        entries = sorted(target.iterdir())
        lines: list[str] = []
        for entry in entries:
            suffix = "/" if entry.is_dir() else ""
            rel = entry.relative_to(Path(ws).resolve())
            lines.append(f"{rel}{suffix}")
        return "\n".join(lines) if lines else "(empty directory)"

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
        target = _resolve_path(path, ws)
        if not target.exists():
            return f"Error: '{path}' does not exist"
        if not target.is_file():
            return f"Error: '{path}' is not a file"

        # Images → base64 (no pagination).
        mime, _ = mimetypes.guess_type(str(target))
        if mime and mime.startswith("image/"):
            data = base64.b64encode(target.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"

        # Guard against huge files without explicit offset/limit.
        file_size = target.stat().st_size
        if file_size > _MAX_FILE_SIZE and offset == 0 and limit == _DEFAULT_LINE_LIMIT:
            size_kb = file_size // 1024
            return (
                f"Error: File is {size_kb} KB ({file_size:,} bytes). "
                f"Use 'offset' and 'limit' parameters to read specific "
                f"portions, or use 'grep' to search for specific content."
            )

        # Read and slice lines.
        text = target.read_text(encoding="utf-8")
        all_lines = text.splitlines()
        total = len(all_lines)

        # Apply offset and limit.
        start = min(offset, total)
        end = min(start + limit, total)
        selected = all_lines[start:end]

        content = _add_line_numbers("\n".join(selected), offset=start)

        # Metadata header.
        header = f"Lines {start + 1}-{end} of {total}"
        if end < total:
            header += f" (use offset={end} to read more)"
        return f"{header}\n{content}"

    async def write_file(path: str, content: str) -> str:
        """Write content to a file, creating parent directories as needed."""
        target = _resolve_path(path, ws)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {path}"

    async def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of 'old' with 'new' in a file."""
        target = _resolve_path(path, ws)
        if not target.exists():
            return f"Error: '{path}' does not exist"
        text = target.read_text(encoding="utf-8")
        if old not in text:
            return f"Error: '{old}' not found in {path}"

        # Find the line range of the edit for context.
        before_lines = text[: text.index(old)].count("\n")
        old_line_count = old.count("\n") + 1
        new_line_count = new.count("\n") + 1

        text = text.replace(old, new, 1)
        target.write_text(text, encoding="utf-8")

        # Build a compact diff summary.
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
        """Create a directory and any missing parents."""
        target = _resolve_path(path, ws)
        target.mkdir(parents=True, exist_ok=True)
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
        ws_path = Path(ws).resolve()
        ws_path.mkdir(parents=True, exist_ok=True)
        matches: list[str] = []
        for match in sorted(ws_path.glob(pattern)):
            if not str(match.resolve()).startswith(str(ws_path)):
                continue
            rel = match.relative_to(ws_path)
            suffix = "/" if match.is_dir() else ""
            matches.append(f"{rel}{suffix}")
            if len(matches) >= max_results:
                matches.append(f"(truncated — {max_results} results shown)")
                break
        return "\n".join(matches) if matches else "(no matches)"

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
            Text to search for (literal string).
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
        target = _resolve_path(path, ws)
        if not target.exists():
            return f"Error: '{path}' does not exist"

        ctx_before = before or context
        ctx_after = after or context
        search_pattern = pattern.lower() if case_insensitive else pattern

        results: list[str] = []
        match_count = 0
        files = sorted(target.rglob("*")) if target.is_dir() else [target]

        ws_resolved = Path(ws).resolve()
        for file in files:
            if not file.is_file():
                continue
            if not str(file.resolve()).startswith(str(ws_resolved)):
                continue
            if not fnmatch.fnmatch(file.name, glob_filter):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            lines = text.splitlines()
            rel = file.relative_to(Path(ws).resolve())

            for i, line in enumerate(lines):
                compare = line.lower() if case_insensitive else line
                if search_pattern in compare:
                    # Context lines.
                    start = max(0, i - ctx_before)
                    end = min(len(lines), i + ctx_after + 1)

                    if ctx_before or ctx_after:
                        for j in range(start, end):
                            marker = ">" if j == i else " "
                            results.append(
                                f"{rel}:{j + 1}:{marker} {lines[j]}"
                            )
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
