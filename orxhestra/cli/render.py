"""Rendering helpers for the orx CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_tool_call(event: Any, console: Any) -> None:
    """Print a tool call event."""
    for tc in event.tool_calls:
        args: dict = tc.args or {}
        args_summary: str = ""

        if "path" in args:
            args_summary = args["path"]
        elif "command" in args:
            cmd: str = args["command"]
            args_summary = cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif "pattern" in args:
            args_summary = args["pattern"]
        elif "description" in args:
            desc: str = args["description"]
            args_summary = desc[:60] + ("..." if len(desc) > 60 else "")
        elif "todos" in args:
            args_summary = "(task list update)"
        else:
            args_summary = ", ".join(
                f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3]
            )

        console.print(f"  [dim]> {tc.tool_name}({args_summary})[/dim]")


def render_tool_response(event: Any, console: Any) -> None:
    """Print a truncated tool response."""
    text: str = (event.text or "")[:300]
    if text:
        lines: list[str] = text.splitlines()
        preview: str = "\n    ".join(lines[:5])
        if len(lines) > 5:
            preview += f"\n    ... ({len(lines)} lines total)"
        console.print(f"    [dim]{preview}[/dim]")


def render_todos(todo_list: Any, console: Any) -> None:
    """Render the todo list if it has items."""
    if todo_list is None or not todo_list.todos:
        return
    rendered: str = todo_list.render()
    if rendered:
        console.print(f"\n[bold]Tasks:[/bold]\n{rendered}")


def print_orx_config(orx_path: Path, console: Any) -> None:
    """Pretty-print the orx.yaml agent configuration on startup."""
    try:
        import yaml
    except ImportError:
        return

    try:
        with open(orx_path) as f:
            raw: dict = yaml.safe_load(f)
    except Exception:
        return

    agents: dict = raw.get("agents", {})
    main_agent: str = raw.get("main_agent", "")
    model_cfg: dict = raw.get("defaults", {}).get("model", {})
    model_str: str = model_cfg.get("name", "?")

    lines: list[str] = []
    for name, agent_def in agents.items():
        agent_type: str = agent_def.get("type", "llm")
        desc: str = agent_def.get("description", "")
        marker: str = "[bold cyan]*[/bold cyan] " if name == main_agent else "  "
        tools: list = agent_def.get("tools", [])
        tool_names: str = ", ".join(str(t) for t in tools) if tools else ""

        type_badge: str = f"[dim]({agent_type})[/dim]"
        line: str = f"  {marker}[bold]{name}[/bold] {type_badge}"
        if desc:
            line += f"  [dim]{desc}[/dim]"
        lines.append(line)
        if tool_names:
            lines.append(f"      [dim]tools: {tool_names}[/dim]")

        sub_agents: list | None = agent_def.get("agents")
        if sub_agents:
            lines.append(f"      [dim]agents: {' -> '.join(sub_agents)}[/dim]")

    from importlib.metadata import version as pkg_version

    try:
        pkg_ver: str = pkg_version("orxhestra")
    except Exception:
        pkg_ver = "?"

    console.print(f"\n  [bold blue]orx[/bold blue] [dim]v{pkg_ver}[/dim]  [dim]{orx_path.name}[/dim]")
    console.print(f"  [dim]model: {model_str}[/dim]")
    if lines:
        console.print()
        for line in lines:
            console.print(line)
