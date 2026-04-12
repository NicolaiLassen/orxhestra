"""Rendering helpers for the orx CLI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from orxhestra.cli.theme import (
    SEP,
    TOOL_BOT,
    TOOL_MID,
    TOOL_TOP,
    TURN_DOT,
)

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.events.event import Event
    from orxhestra.tools.todo_tool import TodoList

_READ_TOOLS: frozenset[str] = frozenset({
    "read_file", "ls", "glob", "grep", "list_artifacts", "load_artifact",
    "list_skills", "load_skill",
})
_WRITE_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "mkdir", "save_artifact",
})
_SHELL_TOOLS: frozenset[str] = frozenset({
    "shell_exec", "shell_exec_background",
})


def _tool_style(tool_name: str) -> str:
    """Return theme style name for a tool category."""
    if tool_name in _READ_TOOLS:
        return "orx.tool.read"
    if tool_name in _WRITE_TOOLS:
        return "orx.tool.write"
    if tool_name in _SHELL_TOOLS:
        return "orx.tool.shell"
    return "orx.tool.default"


def _tool_arg_summary(tool_name: str, args: dict) -> str:
    """Build a concise one-line summary of tool call arguments."""
    if "path" in args:
        return args["path"]
    if "command" in args:
        cmd: str = args["command"]
        return cmd[:80] + ("..." if len(cmd) > 80 else "")
    if "pattern" in args:
        return args["pattern"]
    if "description" in args:
        desc: str = args["description"]
        return desc[:60] + ("..." if len(desc) > 60 else "")
    if "todos" in args:
        return "(task list update)"
    return ", ".join(
        f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3]
    )


def _timestamp() -> str:
    """Return current time as HH:MM for message timestamps."""
    return time.strftime("%H:%M")


def render_tool_call(event: Event, console: Console) -> None:
    """Render tool calls with boxed format and category coloring."""
    # Collapse consecutive read tools into one line.
    read_tools: list[str] = []
    other_tools: list[tuple[str, str, str]] = []

    for tc in event.tool_calls:
        if tc.metadata.get("interactive"):
            continue
        args: dict = tc.args or {}
        summary: str = _tool_arg_summary(tc.tool_name, args)
        style: str = _tool_style(tc.tool_name)

        if tc.tool_name in _READ_TOOLS and len(event.tool_calls) > 1:
            read_tools.append(f"{tc.tool_name}({summary})" if summary else tc.tool_name)
        else:
            other_tools.append((tc.tool_name, summary, style))

    # Render collapsed read group.
    if read_tools:
        collapsed: str = ", ".join(read_tools[:4])
        if len(read_tools) > 4:
            collapsed += f" +{len(read_tools) - 4} more"
        console.print(
            f"  [orx.tool.read]{TOOL_TOP} {collapsed}[/orx.tool.read]"
        )

    # Render other tools normally.
    for tool_name, summary, style in other_tools:
        console.print(f"  [{style}]{TOOL_TOP} {tool_name}[/{style}]")
        if summary:
            console.print(f"  [{style}]{TOOL_MID} {summary}[/{style}]")


def render_tool_response(
    event: Event,
    console: Console,
    *,
    elapsed: float | None = None,
) -> None:
    """Render a truncated tool response with optional timing."""
    text: str = (event.text or "")[:500]
    if elapsed is not None and elapsed < 0.1:
        elapsed_str = ""  # Don't show near-zero times — just noise.
    elif elapsed is not None:
        elapsed_str = f" ({elapsed:.1f}s)"
    else:
        elapsed_str = ""
    if text:
        lines: list[str] = text.splitlines()
        first_line: str = lines[0][:200]
        if len(lines) > 1:
            first_line += f"  ({len(lines)} lines)"
        console.print(
            f"  [orx.muted]{TOOL_BOT} {first_line}{elapsed_str}[/orx.muted]"
        )
    else:
        console.print(
            f"  [orx.muted]{TOOL_BOT} done{elapsed_str}[/orx.muted]"
        )


def render_todos(todo_list: TodoList, console: Console) -> None:
    """Render the todo list if it has pending items."""
    if todo_list is None or not todo_list.todos:
        return
    if not todo_list.has_pending():
        return
    rendered: str = todo_list.render()
    if rendered:
        console.print(f"\n[bold]Tasks:[/bold]\n{rendered}")


def render_turn_summary(
    elapsed: float,
    console: Console,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Print a concise summary line after each agent turn."""
    ts: str = _timestamp()
    parts: list[str] = [f"{elapsed:.1f}s"]
    total: int = prompt_tokens + completion_tokens
    if total > 0:
        parts.append(
            f"{total:,} tokens"
            f" ({prompt_tokens:,}\u2191 {completion_tokens:,}\u2193)"
        )
    summary: str = SEP.join(parts)
    console.print(
        f"  [orx.summary]{TURN_DOT} {summary}"
        f"  [orx.muted]{ts}[/orx.muted][/orx.summary]"
    )


def print_banner(
    orx_path: Path,
    model_name: str,
    workspace: str,
    console: Console,
) -> None:
    """Print a styled welcome banner."""
    import orxhestra

    try:
        from rich.panel import Panel
    except ImportError:
        console.print(
            f"\n  orx v{orxhestra.__version__}  model: {model_name}"
        )
        return

    try:
        import yaml

        with open(orx_path) as f:
            raw: dict = yaml.safe_load(f)
        agents: dict = raw.get("agents", {})
        agent_names: str = (
            ", ".join(agents.keys()) if agents else "default"
        )
    except (OSError, yaml.YAMLError):
        agent_names = "default"

    ws_display: str = str(workspace)
    home: str = str(Path.home())
    if ws_display.startswith(home):
        ws_display = "~" + ws_display[len(home):]

    ver: str = (
        f"[orx.banner.version]"
        f"v{orxhestra.__version__}"
        f"[/orx.banner.version]"
    )
    lbl: str = "orx.banner.label"
    content: str = (
        f"[orx.accent]orx[/orx.accent] {ver}\n"
        f"[{lbl}]model:[/{lbl}]     {model_name}\n"
        f"[{lbl}]workspace:[/{lbl}] {ws_display}\n"
        f"[{lbl}]agents:[/{lbl}]    {agent_names}"
    )

    console.print()
    console.print(
        Panel(
            content,
            border_style="orx.accent",
            padding=(0, 2),
        )
    )
