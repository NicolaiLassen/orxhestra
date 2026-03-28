"""Filesystem tools for agent file operations.

Provides sandboxed file system access: list, read, write, edit, mkdir,
glob, and grep. All operations are restricted to a configurable workspace
root directory.
"""

from __future__ import annotations

import base64
import fnmatch
import mimetypes
import os
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

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

    async def read_file(path: str) -> str:
        """Read the contents of a file. Returns base64 for images."""
        target = _resolve_path(path, ws)
        if not target.exists():
            return f"Error: '{path}' does not exist"
        if not target.is_file():
            return f"Error: '{path}' is not a file"
        mime, _ = mimetypes.guess_type(str(target))
        if mime and mime.startswith("image/"):
            data = base64.b64encode(target.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{data}"
        return target.read_text(encoding="utf-8")

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
        text = text.replace(old, new, 1)
        target.write_text(text, encoding="utf-8")
        return f"Edited {path}"

    async def mkdir(path: str) -> str:
        """Create a directory and any missing parents."""
        target = _resolve_path(path, ws)
        target.mkdir(parents=True, exist_ok=True)
        return f"Created directory {path}"

    async def glob(pattern: str) -> str:
        """Find files matching a glob pattern (e.g. '**/*.py')."""
        ws_path = Path(ws).resolve()
        ws_path.mkdir(parents=True, exist_ok=True)
        matches: list[str] = []
        for match in sorted(ws_path.glob(pattern)):
            # Ensure matches stay within workspace
            if not str(match.resolve()).startswith(str(ws_path)):
                continue
            rel = match.relative_to(ws_path)
            suffix = "/" if match.is_dir() else ""
            matches.append(f"{rel}{suffix}")
        return "\n".join(matches) if matches else "(no matches)"

    async def grep(
        pattern: str,
        path: str = ".",
        glob_filter: str = "*",
    ) -> str:
        """Search file contents for a text pattern.

        Args:
            pattern: Text to search for (literal string).
            path: Directory to search in, relative to workspace.
            glob_filter: Only search files matching this glob (e.g. '*.py').
        """
        target = _resolve_path(path, ws)
        if not target.exists():
            return f"Error: '{path}' does not exist"

        results: list[str] = []
        files = sorted(target.rglob("*")) if target.is_dir() else [target]

        for file in files:
            if not file.is_file():
                continue
            if not fnmatch.fnmatch(file.name, glob_filter):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    rel = file.relative_to(Path(ws).resolve())
                    results.append(f"{rel}:{i}: {line}")

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
            description="Read the contents of a file in the workspace.",
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
            description="Find files matching a glob pattern (e.g. '**/*.py', 'src/*.ts').",
        ),
        StructuredTool.from_function(
            coroutine=grep,
            name="grep",
            description="Search file contents for a pattern. Returns matching lines.",
        ),
    ]
