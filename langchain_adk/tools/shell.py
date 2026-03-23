"""Shell tools for agent command execution.

Provides sandboxed shell access: run commands within a configurable
workspace directory with timeout and output limits.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

_DEFAULT_WORKSPACE = os.environ.get("AGENT_WORKSPACE", "/tmp/agent-workspace")
_DEFAULT_TIMEOUT = int(os.environ.get("AGENT_SHELL_TIMEOUT", "30"))
_DEFAULT_MAX_OUTPUT = int(os.environ.get("AGENT_SHELL_MAX_OUTPUT", "102400"))


def make_shell_tools(
    workspace: str | None = None,
    timeout: int | None = None,
    max_output_bytes: int | None = None,
    allowed_commands: list[str] | None = None,
    denied_commands: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[BaseTool]:
    """Create shell tools sandboxed to a workspace directory.

    Parameters
    ----------
    workspace : str, optional
        Working directory for commands. Defaults to ``$AGENT_WORKSPACE``
        or ``/tmp/agent-workspace``.
    timeout : int, optional
        Max seconds per command. Defaults to ``$AGENT_SHELL_TIMEOUT`` or 30.
    max_output_bytes : int, optional
        Truncate stdout/stderr beyond this size. Defaults to
        ``$AGENT_SHELL_MAX_OUTPUT`` or 100 KB.
    allowed_commands : list[str], optional
        If set, only these base commands are permitted.
    denied_commands : list[str], optional
        Commands that are always blocked (e.g. ``["rm", "shutdown"]``).
    env : dict[str, str], optional
        Extra environment variables merged with the current environment.

    Returns
    -------
    list[BaseTool]
        Two tools: ``shell_exec``, ``shell_exec_background``.
    """
    ws: str = workspace or _DEFAULT_WORKSPACE
    max_timeout: int = timeout or _DEFAULT_TIMEOUT
    max_out: int = max_output_bytes or _DEFAULT_MAX_OUTPUT
    denied: set[str] = set(denied_commands or [])

    merged_env: dict[str, Any] | None = None
    if env:
        merged_env = {**os.environ, **env}

    def _validate_command(command: str) -> str | None:
        """Return error message if command is blocked, else None."""
        base: str = command.strip().split()[0] if command.strip() else ""
        if base in denied:
            return f"Error: command '{base}' is denied."
        if allowed_commands is not None and base not in allowed_commands:
            return (
                f"Error: command '{base}' is not allowed. "
                f"Allowed: {', '.join(allowed_commands)}"
            )
        return None

    def _truncate(data: bytes) -> str:
        """Decode and truncate output to max_output_bytes."""
        text: str = data.decode(errors="replace")
        if len(data) > max_out:
            return text[: max_out] + f"\n... (truncated, {len(data)} bytes total)"
        return text

    async def shell_exec(command: str, timeout_seconds: int | None = None) -> str:
        """Run a shell command and return stdout + stderr."""
        err: str | None = _validate_command(command)
        if err:
            return err

        t: int = min(timeout_seconds or max_timeout, max_timeout)
        os.makedirs(ws, exist_ok=True)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ws,
                env=merged_env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=t)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: command timed out after {t}s"

        output: str = ""
        if stdout:
            output += _truncate(stdout)
        if stderr:
            output += "\n[stderr]\n" + _truncate(stderr)
        if proc.returncode != 0:
            output += f"\n[exit code: {proc.returncode}]"
        return output.strip() or "(no output)"

    async def shell_exec_background(command: str) -> str:
        """Start a shell command in the background and return immediately."""
        err: str | None = _validate_command(command)
        if err:
            return err

        os.makedirs(ws, exist_ok=True)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=ws,
            env=merged_env,
        )
        return f"Started background process (pid={proc.pid})"

    return [
        StructuredTool.from_function(
            coroutine=shell_exec,
            name="shell_exec",
            description="Run a shell command in the workspace and return its output.",
        ),
        StructuredTool.from_function(
            coroutine=shell_exec_background,
            name="shell_exec_background",
            description="Start a shell command in the background. Returns immediately with PID.",
        ),
    ]
