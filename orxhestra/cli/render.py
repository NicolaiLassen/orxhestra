"""Rendering helpers for the orx CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Tool categories for color-coding
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
    """Return a Rich style string based on tool category."""
    if tool_name in _READ_TOOLS:
        return "dim"
    if tool_name in _WRITE_TOOLS:
        return "yellow"
    if tool_name in _SHELL_TOOLS:
        return "bold"
    return "dim"


def _tool_arg_summary(tool_name: str, args: dict) -> str:
    """Build a concise summary of tool call arguments."""
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


def render_tool_call(event: Any, console: Any) -> None:
    """Print a tool call event with boxed format."""
    for tc in event.tool_calls:
        args: dict = tc.args or {}
        summary: str = _tool_arg_summary(tc.tool_name, args)
        style: str = _tool_style(tc.tool_name)
        console.print(f"  [{style}]\u250c {tc.tool_name}[/{style}]")
        if summary:
            console.print(f"  [{style}]\u2502 {summary}[/{style}]")


def render_tool_response(
    event: Any,
    console: Any,
    *,
    elapsed: float | None = None,
) -> None:
    """Print a truncated tool response with optional timing."""
    text: str = (event.text or "")[:300]
    elapsed_str: str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    if text:
        lines: list[str] = text.splitlines()
        first_line: str = lines[0][:120]
        if len(lines) > 1:
            first_line += f"  ({len(lines)} lines)"
        console.print(f"  [dim]\u2514 {first_line}{elapsed_str}[/dim]")
    elif elapsed_str:
        console.print(f"  [dim]\u2514 done{elapsed_str}[/dim]")


def render_todos(todo_list: Any, console: Any) -> None:
    """Render the todo list if it has items."""
    if todo_list is None or not todo_list.todos:
        return
    rendered: str = todo_list.render()
    if rendered:
        console.print(f"\n[bold]Tasks:[/bold]\n{rendered}")


def render_turn_summary(
    elapsed: float,
    console: Any,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Print a concise summary line after each agent turn."""
    parts: list[str] = [f"{elapsed:.1f}s"]
    total: int = prompt_tokens + completion_tokens
    if total > 0:
        parts.append(f"{total:,} tokens ({prompt_tokens:,}\u2191 {completion_tokens:,}\u2193)")
    sep: str = " \u00b7 "
    summary: str = sep.join(parts)
    console.print(f"  [dim]\u27e1 {summary}[/dim]")


def print_banner(
    orx_path: Path,
    model_name: str,
    workspace: str,
    console: Any,
) -> None:
    """Print a styled welcome banner."""
    import orxhestra

    try:
        from rich.panel import Panel
    except ImportError:
        console.print(f"\n  orx v{orxhestra.__version__}  model: {model_name}")
        return

    try:
        import yaml

        with open(orx_path) as f:
            raw: dict = yaml.safe_load(f)
        agents: dict = raw.get("agents", {})
        agent_names: str = ", ".join(agents.keys()) if agents else "default"
    except Exception:
        agent_names = "default"

    # Shorten workspace path
    ws_display: str = str(workspace)
    home: str = str(Path.home())
    if ws_display.startswith(home):
        ws_display = "~" + ws_display[len(home):]

    content: str = (
        f"[bold blue]orx[/bold blue] [dim]v{orxhestra.__version__}[/dim]\n"
        f"[dim]model:[/dim]     {model_name}\n"
        f"[dim]workspace:[/dim] {ws_display}\n"
        f"[dim]agents:[/dim]    {agent_names}"
    )

    console.print()
    console.print(Panel(content, border_style="dim", padding=(0, 2)))
