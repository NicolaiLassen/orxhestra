"""Streaming response handler for the orx CLI."""

from __future__ import annotations

import sys
import time
from typing import Any

from orxhestra.cli.approval import APPROVE_REQUIRED, format_approval_prompt
from orxhestra.cli.config import DEFAULT_USER_ID
from orxhestra.cli.render import (
    render_todos,
    render_tool_call,
    render_tool_response,
    render_turn_summary,
)
from orxhestra.events.event import EventType


async def prompt_approval(
    tool_name: str,
    args: dict[str, Any],
    console: Any,
    auto_approve: bool,
) -> bool:
    """Ask the user to approve a destructive tool call."""
    if auto_approve:
        return True
    if tool_name not in APPROVE_REQUIRED:
        return True

    console.print(format_approval_prompt(tool_name, args))

    try:
        response: str = input("  approve? [y/n/a(ll)]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if response in ("a", "all"):
        return True
    return response in ("y", "yes")


async def stream_response(
    runner: Any,
    session_id: str,
    message: str,
    console: Any,
    Markdown: type,
    *,
    todo_list: Any = None,
    auto_approve: bool = False,
) -> bool:
    """Stream a single agent turn, rendering events in real time.

    Returns updated auto_approve value (may change if user selects 'all').
    """
    buffer: str = ""
    in_stream: bool = False
    live: Any = None
    status: Any = None
    tool_start: float = 0.0
    turn_start: float = time.monotonic()
    prompt_tokens: int = 0
    completion_tokens: int = 0

    try:
        from rich.live import Live
    except ImportError:
        Live = None

    try:
        from rich.status import Status
    except ImportError:
        Status = None

    def _stop_status() -> None:
        nonlocal status
        if status is not None:
            status.stop()
            status = None

    try:
        async for event in runner.astream(
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
            new_message=message,
        ):
            # Extract token usage from event metadata
            usage: dict = event.metadata.get("usage", {})
            if usage:
                prompt_tokens += usage.get("prompt_tokens", 0) or 0
                completion_tokens += usage.get("completion_tokens", 0) or 0

            # Streaming partial tokens
            if event.partial and event.type == EventType.AGENT_MESSAGE and event.text:
                _stop_status()
                buffer += event.text
                if Live is not None:
                    if not in_stream:
                        in_stream = True
                        live = Live(
                            Markdown(buffer),
                            console=console,
                            refresh_per_second=12,
                            vertical_overflow="visible",
                        )
                        live.start()
                    else:
                        live.update(Markdown(buffer))
                else:
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                continue

            # Tool call — with approval for destructive tools
            if event.has_tool_calls:
                _stop_status()
                if in_stream and live:
                    live.stop()
                    in_stream = False
                    if buffer:
                        console.print(Markdown(buffer))
                        buffer = ""

                render_tool_call(event, console)

                for tc in event.tool_calls:
                    if tc.tool_name in APPROVE_REQUIRED and not auto_approve:
                        approved: bool = await prompt_approval(
                            tc.tool_name, tc.args or {}, console, auto_approve
                        )
                        if not approved:
                            console.print("  [dim]Denied.[/dim]")

                # Start spinner while waiting for tool response
                tool_start = time.monotonic()
                if Status is not None:
                    last_tool: str = event.tool_calls[-1].tool_name
                    status = Status(
                        f"  Running {last_tool}...",
                        console=console,
                        spinner="dots",
                    )
                    status.start()
                continue

            # Tool response
            if event.type == EventType.TOOL_RESPONSE:
                _stop_status()
                elapsed: float | None = None
                if tool_start > 0:
                    elapsed = time.monotonic() - tool_start
                    tool_start = 0.0
                render_tool_response(event, console, elapsed=elapsed)

                if event.tool_name == "write_todos" and todo_list is not None:
                    render_todos(todo_list, console)
                continue

            # Final response
            if event.is_final_response():
                _stop_status()
                was_streaming: bool = in_stream
                if in_stream and live:
                    live.stop()
                    in_stream = False
                    buffer = ""
                if not was_streaming and event.text:
                    agent_label: str = (
                        f"[{event.agent_name}] "
                        if event.agent_name and event.agent_name != "coder"
                        else ""
                    )
                    if agent_label:
                        console.print(f"\n[bold cyan]{agent_label}[/bold cyan]")
                    console.print(Markdown(event.text))
                continue

            # Error events
            if event.metadata.get("error") and event.text:
                _stop_status()
                if in_stream and live:
                    live.stop()
                    in_stream = False
                    buffer = ""
                console.print(f"[bold red]Error:[/bold red] {event.text}")
                continue

    except KeyboardInterrupt:
        _stop_status()
        if in_stream and live:
            live.stop()
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        _stop_status()
        if in_stream and live:
            live.stop()
            if buffer:
                console.print(Markdown(buffer))

    # Turn summary line
    turn_elapsed: float = time.monotonic() - turn_start
    render_turn_summary(
        turn_elapsed,
        console,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return auto_approve
