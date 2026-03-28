"""Human-in-the-loop approval for destructive tool calls.

Wraps tools so that dangerous operations (write_file, edit_file, shell_exec, mkdir)
require explicit user confirmation before execution.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel

# Tools that require approval before execution
APPROVE_REQUIRED: frozenset[str] = frozenset({
    "write_file",
    "edit_file",
    "shell_exec",
    "mkdir",
})

# Patterns that indicate especially dangerous shell commands
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-rf?|--recursive)", re.IGNORECASE),
    re.compile(r"\bgit\s+(push\s+--force|reset\s+--hard|clean\s+-f)", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bcurl\s+.*\|\s*(bash|sh|zsh)\b"),
]

# Hidden Unicode categories that may indicate deception
_SUSPICIOUS_UNICODE: list[tuple[int, int]] = [
    (0x200B, 0x200F),  # zero-width chars
    (0x202A, 0x202E),  # bidi overrides
    (0x2066, 0x2069),  # bidi isolates
    (0xFEFF, 0xFEFF),  # BOM
]


def _has_suspicious_unicode(text: str) -> bool:
    """Check for hidden Unicode characters that may indicate deception."""
    for char in text:
        code: int = ord(char)
        for lo, hi in _SUSPICIOUS_UNICODE:
            if lo <= code <= hi:
                return True
    return False


def format_approval_prompt(tool_name: str, args: dict[str, Any]) -> str:
    """Format a Rich-styled approval prompt with a bordered panel."""
    # Build the primary value to display
    if tool_name == "shell_exec":
        primary: str = args.get("command", "")
    elif tool_name in ("write_file", "edit_file", "mkdir"):
        primary = args.get("path", "")
    else:
        primary = ""

    lines: list[str] = []
    if primary:
        display: str = primary if len(primary) <= 120 else primary[:120] + "..."
        lines.append(f"  [bold white]{display}[/bold white]")

    # Show other args (skip the primary one we already showed)
    skip_key: str = "command" if tool_name == "shell_exec" else "path"
    for key, value in args.items():
        if key == skip_key:
            continue
        val_str: str = str(value)
        if len(val_str) > 100:
            val_str = val_str[:100] + "..."
        lines.append(f"  [dim]{key}: {val_str}[/dim]")

    # Warn about dangerous patterns
    is_dangerous: bool = False
    if tool_name == "shell_exec":
        cmd: str = args.get("command", "")
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(cmd):
                is_dangerous = True
                break

    # Check for hidden Unicode
    has_hidden: bool = False
    for value in args.values():
        if isinstance(value, str) and _has_suspicious_unicode(value):
            has_hidden = True
            break

    if is_dangerous:
        lines.append("  [bold red]⚠ destructive command[/bold red]")
    if has_hidden:
        lines.append("  [bold red]⚠ hidden unicode detected[/bold red]")

    border: str = "[bold red]" if is_dangerous or has_hidden else "[yellow]"
    header: str = f"{border}{'━' * 50}[/]"
    label: str = f"  {border}approve[/] [bold]{tool_name}[/bold]"

    body: str = "\n".join(lines)
    return f"\n{header}\n{label}\n{body}\n{header}"


class ApprovalWrapper(BaseTool):
    """Wraps a tool to require human approval before execution."""

    name: str
    description: str
    args_schema: type[BaseModel] | None = None
    inner_tool: BaseTool
    auto_approve_session: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, tool: BaseTool) -> None:
        super().__init__(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            inner_tool=tool,
        )

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async ainvoke.")

    async def _arun(
        self,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute with approval check."""
        # Delegate to inner tool - approval is handled at the CLI level
        return await self.inner_tool.ainvoke(kwargs)


def wrap_tools_with_approval(tools: list[BaseTool]) -> list[BaseTool]:
    """Wrap destructive tools with approval wrappers."""
    result: list[BaseTool] = []
    for tool in tools:
        if tool.name in APPROVE_REQUIRED:
            result.append(ApprovalWrapper(tool))
        else:
            result.append(tool)
    return result
