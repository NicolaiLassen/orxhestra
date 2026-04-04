"""Slash command handlers for the orx REPL."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from orxhestra.cli.config import DEFAULT_USER_ID

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.cli.state import ReplState

logger: logging.Logger = logging.getLogger(__name__)

_CMD: str = "orx.help.cmd"
_DESC: str = "orx.help.desc"
_HELP_TEXT: str = (
    f"[{_CMD}]Commands:[/{_CMD}]\n"
    f"  [{_CMD}]/model <name>[/{_CMD}]  [{_DESC}]Switch model[/{_DESC}]\n"
    f"  [{_CMD}]/clear[/{_CMD}]         [{_DESC}]Clear session[/{_DESC}]\n"
    f"  [{_CMD}]/compact[/{_CMD}]       [{_DESC}]Compact context[/{_DESC}]\n"
    f"  [{_CMD}]/todos[/{_CMD}]         [{_DESC}]Show tasks[/{_DESC}]\n"
    f"  [{_CMD}]/session[/{_CMD}]       [{_DESC}]Session info[/{_DESC}]\n"
    f"  [{_CMD}]/undo[/{_CMD}]          [{_DESC}]Remove last turn[/{_DESC}]\n"
    f"  [{_CMD}]/retry[/{_CMD}]         [{_DESC}]Re-run last message[/{_DESC}]\n"
    f"  [{_CMD}]/copy[/{_CMD}]          [{_DESC}]Copy last response[/{_DESC}]\n"
    f"  [{_CMD}]/memory[/{_CMD}]        [{_DESC}]List saved memories[/{_DESC}]\n"
    f"  [{_CMD}]/theme[/{_CMD}]         [{_DESC}]Switch theme (dark/light)[/{_DESC}]\n"
    f"  [{_CMD}]/exit[/{_CMD}]          [{_DESC}]Exit[/{_DESC}]\n"
    f"  [{_CMD}]/help[/{_CMD}]          [{_DESC}]Show this help[/{_DESC}]\n"
    f"\n[{_CMD}]Multi-line input:[/{_CMD}]\n"
    f'  [{_DESC}]Start with """ or \'\'\' and end with same.[/{_DESC}]'
)


async def _cmd_exit(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Exit the REPL."""
    console.print("[orx.status]Goodbye![/orx.status]")
    state.should_continue = False


async def _cmd_clear(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Clear the session and reset state."""
    state.session_id = str(uuid4())
    if state.todo_list is not None:
        state.todo_list.todos = []
    state.turn_count = 0
    console.print("[orx.status]Session cleared.[/orx.status]")


async def _cmd_compact(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Summarize old messages to free context."""
    if state.llm is None:
        console.print("[orx.status]Compact not available.[/orx.status]")
        return

    console.print("[orx.status]Compacting conversation...[/orx.status]")
    from orxhestra.cli.summarization import summarize_session

    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    result = await summarize_session(state.llm, session.events)
    if result is not None:
        session.events[:] = result
        console.print("[orx.status]Conversation compacted.[/orx.status]")
    else:
        console.print("[orx.status]Nothing to compact.[/orx.status]")


async def _cmd_model(
    state: ReplState,
    cmd_arg: str | None,
    *,
    console: Console,
    orx_path: object,
    workspace: str,
    **_kw: object,
) -> None:
    """Switch to a different model, preserving history."""
    from pathlib import Path

    if not cmd_arg:
        console.print(
            f"[orx.status]Current model: {state.model_name}[/orx.status]"
        )
        return

    try:
        from orxhestra.cli.builder import build_from_orx

        old_session = await state.runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=state.session_id
        )
        old_events = list(old_session.events)

        new_state = await build_from_orx(
            Path(str(orx_path)), cmd_arg, workspace
        )
        state.runner = new_state.runner
        state.todo_list = new_state.todo_list
        state.llm = new_state.llm
        state.model_name = cmd_arg
        state.turn_count = 0

        new_session = await state.runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=state.session_id
        )
        new_session.events.extend(old_events)

        msg: str = f"Switched to {state.model_name} (history preserved)"
        console.print(f"[orx.status]{msg}[/orx.status]")
    except Exception as e:
        console.print(f"[orx.error]Error: {e}[/orx.error]")


async def _cmd_todos(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Show the current task list."""
    from orxhestra.cli.render import render_todos

    if state.todo_list is not None:
        render_todos(state.todo_list, console)
        if not state.todo_list.todos:
            console.print("[orx.status]No tasks.[/orx.status]")
    else:
        console.print("[orx.status]No tasks.[/orx.status]")


async def _cmd_session(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Show session info (events, chars, turns)."""
    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    event_count: int = len(session.events)
    from orxhestra.sessions.compaction import _estimate_event_chars

    total_chars: int = sum(
        _estimate_event_chars(e) for e in session.events
    )
    chars_info: str = f"{total_chars:,} (~{total_chars // 4:,} tokens)"
    console.print(
        f"  [orx.status]session:  {state.session_id}[/orx.status]"
    )
    console.print(f"  [orx.status]events:   {event_count}[/orx.status]")
    console.print(f"  [orx.status]chars:    {chars_info}[/orx.status]")
    console.print(
        f"  [orx.status]turns:    {state.turn_count}[/orx.status]"
    )


async def _cmd_undo(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Remove the last conversation turn."""
    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    from orxhestra.events.event import EventType

    last_user_idx: int = -1
    for i in range(len(session.events) - 1, -1, -1):
        if session.events[i].type == EventType.USER_MESSAGE:
            last_user_idx = i
            break
    if last_user_idx >= 0:
        removed: int = len(session.events) - last_user_idx
        session.events[:] = session.events[:last_user_idx]
        state.turn_count = max(0, state.turn_count - 1)
        console.print(
            f"[orx.status]Removed last turn ({removed} events).[/orx.status]"
        )
    else:
        console.print("[orx.status]Nothing to undo.[/orx.status]")


async def _cmd_retry(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Re-run the last user message."""
    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    from orxhestra.events.event import EventType

    last_msg: str | None = None
    last_user_idx: int = -1
    for i in range(len(session.events) - 1, -1, -1):
        if session.events[i].type == EventType.USER_MESSAGE:
            last_msg = session.events[i].text
            last_user_idx = i
            break
    if last_msg and last_user_idx >= 0:
        session.events[:] = session.events[:last_user_idx]
        state.turn_count = max(0, state.turn_count - 1)
        console.print(
            f"[orx.status]Retrying: {last_msg[:60]}[/orx.status]"
        )
        state.retry_message = last_msg
    else:
        console.print("[orx.status]Nothing to retry.[/orx.status]")


async def _cmd_copy(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Copy the last agent response to the clipboard."""
    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    from orxhestra.events.event import EventType

    last_response: str | None = None
    for i in range(len(session.events) - 1, -1, -1):
        e = session.events[i]
        if (
            e.type == EventType.AGENT_MESSAGE
            and e.text
            and not e.partial
        ):
            last_response = e.text
            break
    if last_response:
        try:
            import subprocess

            subprocess.run(
                ["pbcopy"],
                input=last_response.encode(),
                check=True,
            )
            console.print(
                "[orx.status]Copied to clipboard.[/orx.status]"
            )
        except Exception:
            logger.debug("Clipboard copy failed", exc_info=True)
            console.print(
                "[orx.status]Clipboard not available.[/orx.status]"
            )
    else:
        console.print("[orx.status]No response to copy.[/orx.status]")


async def _cmd_memory(
    _state: ReplState,
    cmd_arg: str | None,
    *,
    console: Console,
    workspace: str,
    **_kw: object,
) -> None:
    """List or clear saved memories."""
    from orxhestra.memory.file_memory_service import (
        get_memory_dir,
        scan_memory_files,
    )

    mem_dir = get_memory_dir(workspace)

    if cmd_arg and cmd_arg.strip().lower() == "clear":
        headers = scan_memory_files(mem_dir)
        if not headers:
            console.print("[orx.status]No memories to clear.[/orx.status]")
            return
        try:
            confirm = input(
                f"  Delete {len(headers)} memories? [y/n]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if confirm not in ("y", "yes"):
            console.print("[orx.status]Cancelled.[/orx.status]")
            return
        for h in headers:
            h.filepath.unlink(missing_ok=True)
        index_path = mem_dir / "MEMORY.md"
        index_path.unlink(missing_ok=True)
        console.print(
            f"[orx.success]Deleted {len(headers)} memories.[/orx.success]"
        )
        return

    headers = scan_memory_files(mem_dir)
    if not headers:
        console.print(
            "[orx.status]No memories saved. The agent can use "
            "save_memory to persist learnings.[/orx.status]"
        )
        return

    console.print(f"[orx.accent]{len(headers)} memories:[/orx.accent]")
    for h in headers:
        type_tag = f"[{h.memory_type}]" if h.memory_type else "[?]"
        desc = f" — {h.description}" if h.description else ""
        console.print(f"  [orx.muted]{type_tag}[/orx.muted] {h.name}{desc}")


async def _cmd_theme(
    _state: ReplState,
    cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Switch between dark and light themes."""
    if cmd_arg and cmd_arg.lower() in ("dark", "light"):
        new_theme = cmd_arg.lower()
    else:
        # Toggle current theme.
        import os

        current = os.environ.get("ORX_THEME", "dark")
        new_theme = "light" if current == "dark" else "dark"

    import os

    os.environ["ORX_THEME"] = new_theme
    console.print(
        f"[orx.status]Theme set to {new_theme}. "
        f"Restart orx to apply fully.[/orx.status]"
    )


async def _cmd_help(
    _state: ReplState,
    _cmd_arg: str | None,
    *,
    console: Console,
    **_kw: object,
) -> None:
    """Show the help text."""
    console.print(_HELP_TEXT)


_DISPATCH: dict[str, Callable[..., object]] = {
    "/exit": _cmd_exit,
    "/quit": _cmd_exit,
    "/clear": _cmd_clear,
    "/compact": _cmd_compact,
    "/model": _cmd_model,
    "/todos": _cmd_todos,
    "/session": _cmd_session,
    "/undo": _cmd_undo,
    "/retry": _cmd_retry,
    "/copy": _cmd_copy,
    "/memory": _cmd_memory,
    "/theme": _cmd_theme,
    "/help": _cmd_help,
}


async def handle_slash_command(
    cmd: str,
    cmd_arg: str | None,
    state: ReplState,
    *,
    console: Console,
    orx_path: object,
    workspace: str,
) -> None:
    """Dispatch a slash command, mutating *state* in place.

    Parameters
    ----------
    cmd : str
        The slash command string (e.g. ``"/clear"``).
    cmd_arg : str or None
        Optional argument following the command.
    state : ReplState
        Shared mutable REPL state.
    console : Console
        Rich console for output.
    orx_path : object
        Path to the orx YAML file.
    workspace : str
        Workspace directory path.
    """
    handler = _DISPATCH.get(cmd)
    if handler is None:
        console.print(
            f"[orx.status]Unknown command: {cmd}. Type /help[/orx.status]"
        )
        return
    await handler(
        state,
        cmd_arg,
        console=console,
        orx_path=orx_path,
        workspace=workspace,
    )
