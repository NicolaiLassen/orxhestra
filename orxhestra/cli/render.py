"""Rendering helpers for the orx CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from orxhestra.cli.theme import SEP, TOOL_BOT, TOOL_MID, TOOL_TOP, TURN_DOT

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


def render_tool_call(event: Event, console: Console) -> None:
    """Render tool calls with boxed format and category coloring.

    Parameters
    ----------
    event : Event
        Event containing one or more tool calls.
    console : Console
        Rich console for output.
    """
    for tc in event.tool_calls:
        if tc.metadata.get("interactive"):
            continue
        args: dict = tc.args or {}
        summary: str = _tool_arg_summary(tc.tool_name, args)
        style: str = _tool_style(tc.tool_name)
        console.print(f"  [{style}]{TOOL_TOP} {tc.tool_name}[/{style}]")
        if summary:
            console.print(f"  [{style}]{TOOL_MID} {summary}[/{style}]")


def render_tool_response(
    event: Event,
    console: Console,
    *,
    elapsed: float | None = None,
) -> None:
    """Render a truncated tool response with optional timing.

    Parameters
    ----------
    event : Event
        Event containing the tool response text.
    console : Console
        Rich console for output.
    elapsed : float or None
        Wall-clock seconds the tool call took, or None if unavailable.
    """
    text: str = (event.text or "")[:300]
    elapsed_str: str = f" ({elapsed:.1f}s)" if elapsed is not None else ""
    if text:
        lines: list[str] = text.splitlines()
        first_line: str = lines[0][:120]
        if len(lines) > 1:
            first_line += f"  ({len(lines)} lines)"
        console.print(
            f"  [orx.muted]{TOOL_BOT} {first_line}{elapsed_str}[/orx.muted]"
        )
    elif elapsed_str:
        console.print(
            f"  [orx.muted]{TOOL_BOT} done{elapsed_str}[/orx.muted]"
        )


def render_todos(todo_list: TodoList, console: Console) -> None:
    """Render the todo list if it has items.

    Parameters
    ----------
    todo_list : TodoList
        The todo list instance to render.
    console : Console
        Rich console for output.
    """
    if todo_list is None or not todo_list.todos:
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
    """Print a concise summary line after each agent turn.

    Parameters
    ----------
    elapsed : float
        Wall-clock seconds for the turn.
    console : Console
        Rich console for output.
    prompt_tokens : int
        Number of prompt tokens used.
    completion_tokens : int
        Number of completion tokens used.
    """
    parts: list[str] = [f"{elapsed:.1f}s"]
    total: int = prompt_tokens + completion_tokens
    if total > 0:
        parts.append(
            f"{total:,} tokens ({prompt_tokens:,}\u2191 {completion_tokens:,}\u2193)"
        )
    summary: str = SEP.join(parts)
    console.print(f"  [orx.summary]{TURN_DOT} {summary}[/orx.summary]")


def print_banner(
    orx_path: Path,
    model_name: str,
    workspace: str,
    console: Console,
) -> None:
    """Print a styled welcome banner.

    Parameters
    ----------
    orx_path : Path
        Path to the orx YAML file.
    model_name : str
        Name of the active LLM model.
    workspace : str
        Workspace directory path.
    console : Console
        Rich console for output.
    """
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
        f"[orx.banner.version]v{orxhestra.__version__}[/orx.banner.version]"
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
        Panel(content, border_style="orx.subtle", padding=(0, 2))
    )
