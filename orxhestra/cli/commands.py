"""Slash command handlers for the orx REPL.

Every handler takes the shared :class:`ReplState`, an optional
``cmd_arg`` (whatever follows the slash command on the input line),
and a :class:`Writer` plus any extra keyword context the dispatcher
injects.  Registration happens through :func:`register_command` so
plugin authors can extend the REPL vocabulary.

See Also
--------
orxhestra.cli.app : Entry point that launches the REPL.
orxhestra.cli.state.ReplState : Shared mutable REPL state.
orxhestra.cli.writer.Writer : Output sink shared across commands.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from orxhestra.cli.config import DEFAULT_USER_ID

if TYPE_CHECKING:
    from orxhestra.cli.state import ReplState
    from orxhestra.cli.writer import Writer

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
    writer: Writer,
    **_kw: object,
) -> None:
    """Exit the REPL by flipping ``state.should_continue`` to ``False``.

    Parameters
    ----------
    state : ReplState
        REPL state flipped to terminate the outer loop.
    _cmd_arg : str or None
        Ignored.
    writer : Writer
        Output sink used to print the goodbye message.
    """
    writer.print_rich("[orx.status]Goodbye![/orx.status]")
    state.should_continue = False


async def _cmd_clear(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Start a fresh session — new session id, zeroed todos and turn count.

    Parameters
    ----------
    state : ReplState
        State mutated in place.
    _cmd_arg : str or None
        Ignored.
    writer : Writer
        Output sink.
    """
    state.session_id = str(uuid4())
    if state.todo_list is not None:
        state.todo_list.todos = []
    state.turn_count = 0
    writer.print_rich("[orx.status]Session cleared.[/orx.status]")


async def _cmd_compact(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Summarize old messages to free up LLM context window space.

    Runs :func:`orxhestra.cli.summarization.summarize_session` on the
    current session's events.  No-ops when no model is configured or
    the session contains nothing worth compacting.

    Parameters
    ----------
    state : ReplState
    _cmd_arg : str or None
        Ignored.
    writer : Writer
    """
    if state.model is None:
        writer.print_rich("[orx.status]Compact not available.[/orx.status]")
        return

    writer.print_rich("[orx.status]Compacting conversation...[/orx.status]")
    from orxhestra.cli.summarization import summarize_session

    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    result = await summarize_session(state.model, session.events)
    if result is not None:
        session.events[:] = result
        writer.print_rich("[orx.status]Conversation compacted.[/orx.status]")
    else:
        writer.print_rich("[orx.status]Nothing to compact.[/orx.status]")


async def _cmd_model(
    state: ReplState,
    cmd_arg: str | None,
    *,
    writer: Writer,
    orx_path: object,
    workspace: str,
    **_kw: object,
) -> None:
    """Switch to a different LLM model, preserving conversation history.

    Rebuilds the Runner using :func:`orxhestra.cli.builder.build_from_orx`
    and rehydrates the new session with the prior events so context
    is not lost.  When ``cmd_arg`` is empty, prints the active model
    name instead of switching.

    Parameters
    ----------
    state : ReplState
    cmd_arg : str or None
        New model identifier (e.g. ``"claude-sonnet-4-6"``).  Empty
        / ``None`` triggers a read-only print of the current model.
    writer : Writer
    orx_path : object
        Path to the current orx YAML file (for rebuilding the tree).
    workspace : str
        Workspace directory used by the builder.
    """
    from pathlib import Path

    if not cmd_arg:
        writer.print_rich(
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
        state.model = new_state.model
        state.model_name = cmd_arg
        state.turn_count = 0

        new_session = await state.runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=state.session_id
        )
        new_session.events.extend(old_events)

        msg: str = f"Switched to {state.model_name} (history preserved)"
        writer.print_rich(f"[orx.status]{msg}[/orx.status]")
    except Exception as e:
        writer.print_rich(f"[orx.error]Error: {e}[/orx.error]")


async def _cmd_todos(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Render the agent's current todo list to the writer.

    Parameters
    ----------
    state : ReplState
        REPL state; ``state.todo_list`` is read, not mutated.
    _cmd_arg : str or None
        Ignored.
    writer : Writer
        Output sink.
    """
    from orxhestra.cli.render import render_todos

    if state.todo_list is not None:
        render_todos(state.todo_list, writer)
        if not state.todo_list.todos:
            writer.print_rich("[orx.status]No tasks.[/orx.status]")
    else:
        writer.print_rich("[orx.status]No tasks.[/orx.status]")


async def _cmd_session(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Print session diagnostics — event count, bytes used, turn count.

    Uses :func:`orxhestra.sessions.compaction._estimate_event_chars`
    for a coarse character budget so the user can eyeball how close
    they are to the context window limit.

    Parameters
    ----------
    state : ReplState
    _cmd_arg : str or None
        Ignored.
    writer : Writer
    """
    session = await state.runner.get_or_create_session(
        user_id=DEFAULT_USER_ID, session_id=state.session_id
    )
    event_count: int = len(session.events)
    from orxhestra.sessions.compaction import _estimate_event_chars

    total_chars: int = sum(
        _estimate_event_chars(e) for e in session.events
    )
    chars_info: str = f"{total_chars:,} (~{total_chars // 4:,} tokens)"
    writer.print_rich(
        f"  [orx.status]session:  {state.session_id}[/orx.status]"
    )
    writer.print_rich(f"  [orx.status]events:   {event_count}[/orx.status]")
    writer.print_rich(f"  [orx.status]chars:    {chars_info}[/orx.status]")
    writer.print_rich(
        f"  [orx.status]turns:    {state.turn_count}[/orx.status]"
    )
    if state.signer_did:
        writer.print_rich(
            f"  [orx.status]identity: {state.signer_did}[/orx.status]"
        )
        if state.identity_key_path:
            writer.print_rich(
                f"  [orx.status]key file: {state.identity_key_path}[/orx.status]"
            )
    else:
        writer.print_rich(
            "  [orx.status]identity: disabled "
            "(pass --identity PATH to enable signing)[/orx.status]"
        )


async def _cmd_undo(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Remove the last conversation turn (user + response) from history.

    Walks backwards to find the last :attr:`EventType.USER_MESSAGE`
    and truncates the session event list there.  Decrements the turn
    counter.  No-ops when no user message is found.

    Parameters
    ----------
    state : ReplState
    _cmd_arg : str or None
        Ignored.
    writer : Writer
    """
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
        writer.print_rich(
            f"[orx.status]Removed last turn ({removed} events).[/orx.status]"
        )
    else:
        writer.print_rich("[orx.status]Nothing to undo.[/orx.status]")


async def _cmd_retry(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Re-run the most recent user message.

    Truncates history at the last user message, stashes the prompt
    in ``state.retry_message``, and lets the main REPL loop replay
    it against the (possibly updated) agent.

    Parameters
    ----------
    state : ReplState
    _cmd_arg : str or None
        Ignored.
    writer : Writer
    """
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
        writer.print_rich(
            f"[orx.status]Retrying: {last_msg[:60]}[/orx.status]"
        )
        state.retry_message = last_msg
    else:
        writer.print_rich("[orx.status]Nothing to retry.[/orx.status]")


async def _cmd_copy(
    state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Copy the last agent response to the macOS clipboard via ``pbcopy``.

    Non-macOS platforms fall through to a gentle failure message.

    Parameters
    ----------
    state : ReplState
    _cmd_arg : str or None
        Ignored.
    writer : Writer
    """
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
            writer.print_rich(
                "[orx.status]Copied to clipboard.[/orx.status]"
            )
        except Exception:
            logger.debug("Clipboard copy failed", exc_info=True)
            writer.print_rich(
                "[orx.status]Clipboard not available.[/orx.status]"
            )
    else:
        writer.print_rich("[orx.status]No response to copy.[/orx.status]")


async def _cmd_memory(
    _state: ReplState,
    cmd_arg: str | None,
    *,
    writer: Writer,
    workspace: str,
    **_kw: object,
) -> None:
    """List or clear the agent's saved memories.

    Memories live on disk under
    :func:`orxhestra.memory.file_memory_service.get_memory_dir`.
    The ``clear`` subcommand prompts for confirmation before
    deleting every memory file plus the ``MEMORY.md`` index.

    Parameters
    ----------
    _state : ReplState
        Ignored; present to satisfy the dispatcher signature.
    cmd_arg : str or None
        ``"clear"`` to delete all memories; anything else prints the
        list.
    writer : Writer
    workspace : str
        Workspace root used to locate the memory directory.
    """
    from orxhestra.memory.file_memory_service import (
        get_memory_dir,
        scan_memory_files,
    )

    mem_dir = get_memory_dir(workspace)

    if cmd_arg and cmd_arg.strip().lower() == "clear":
        headers = scan_memory_files(mem_dir)
        if not headers:
            writer.print_rich("[orx.status]No memories to clear.[/orx.status]")
            return
        try:
            confirm = (await writer.prompt_input(
                f"  Delete {len(headers)} memories? [y/n]: "
            )).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if confirm not in ("y", "yes"):
            writer.print_rich("[orx.status]Cancelled.[/orx.status]")
            return
        for h in headers:
            h.filepath.unlink(missing_ok=True)
        index_path = mem_dir / "MEMORY.md"
        index_path.unlink(missing_ok=True)
        writer.print_rich(
            f"[orx.success]Deleted {len(headers)} memories.[/orx.success]"
        )
        return

    headers = scan_memory_files(mem_dir)
    if not headers:
        writer.print_rich(
            "[orx.status]No memories saved. The agent can use "
            "save_memory to persist learnings.[/orx.status]"
        )
        return

    writer.print_rich(f"[orx.accent]{len(headers)} memories:[/orx.accent]")
    for h in headers:
        type_tag = f"[{h.memory_type}]" if h.memory_type else "[?]"
        desc = f" — {h.description}" if h.description else ""
        writer.print_rich(f"  [orx.muted]{type_tag}[/orx.muted] {h.name}{desc}")


async def _cmd_theme(
    _state: ReplState,
    cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Switch between the dark and light REPL themes.

    With no argument, toggles the current value of
    ``$ORX_THEME``.  With ``dark`` or ``light``, sets it explicitly.
    Changes persist via
    :func:`orxhestra.cli.theme.save_theme` but only take effect on
    next launch.

    Parameters
    ----------
    _state : ReplState
        Ignored.
    cmd_arg : str or None
        ``"dark"`` / ``"light"``, or empty to toggle.
    writer : Writer
    """
    import os

    from orxhestra.cli.theme import save_theme

    if cmd_arg and cmd_arg.lower() in ("dark", "light"):
        new_theme = cmd_arg.lower()
    else:
        # Toggle current theme.
        current = os.environ.get("ORX_THEME", "dark")
        new_theme = "light" if current == "dark" else "dark"

    os.environ["ORX_THEME"] = new_theme
    save_theme(new_theme)
    writer.print_rich(
        f"[orx.status]Theme set to {new_theme}. "
        f"Restart orx to apply.[/orx.status]"
    )


async def _cmd_help(
    _state: ReplState,
    _cmd_arg: str | None,
    *,
    writer: Writer,
    **_kw: object,
) -> None:
    """Print the REPL help banner.

    Parameters
    ----------
    _state : ReplState
        Ignored.
    _cmd_arg : str or None
        Ignored.
    writer : Writer
    """
    writer.print_rich(_HELP_TEXT)


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


def register_command(
    name: str,
    handler: Callable[..., object] | None = None,
) -> Callable[..., object]:
    """Register a slash-command handler in the global dispatcher.

    Dual-mode: works as a decorator when ``handler`` is omitted,
    otherwise registers the supplied handler immediately.  Existing
    commands under the same name are overwritten.

    Parameters
    ----------
    name : str
        Command name including the leading slash (e.g. ``"/greet"``).
    handler : callable, optional
        Async handler accepting ``(state, cmd_arg, *, writer, **kw)``.
        When ``None``, the returned function itself acts as a
        decorator.

    Returns
    -------
    callable
        The registered handler (for decorator chaining), or a
        decorator that registers its argument when called.

    Examples
    --------
    >>> @register_command("/greet")
    ... async def greet(state, cmd_arg, *, writer, **kw):
    ...     writer.print_rich("Hello!")

    >>> register_command("/greet", my_handler)
    """
    if handler is not None:
        _DISPATCH[name] = handler
        return handler

    def decorator(fn: Callable[..., object]) -> Callable[..., object]:
        _DISPATCH[name] = fn
        return fn

    return decorator


def get_command_names() -> list[str]:
    """Return all registered slash-command names in sorted order.

    Used by the input autocomplete to surface available commands.

    Returns
    -------
    list[str]
        Slash-command names including the leading slash.
    """
    return sorted(_DISPATCH.keys())


async def handle_slash_command(
    cmd: str,
    cmd_arg: str | None,
    state: ReplState,
    *,
    writer: Writer,
    orx_path: object,
    workspace: str,
) -> None:
    """Dispatch a slash command, mutating ``state`` in place.

    Unknown commands produce a friendly pointer to ``/help`` rather
    than raising, so a typo never takes the REPL down.

    Parameters
    ----------
    cmd : str
        The slash-command string (e.g. ``"/clear"``).
    cmd_arg : str or None
        Optional argument following the command, with leading
        whitespace stripped by the caller.
    state : ReplState
        Shared mutable REPL state.
    writer : Writer
        Output writer.
    orx_path : object
        Path to the orx YAML file for commands that rebuild the
        Runner.
    workspace : str
        Workspace directory path.
    """
    handler = _DISPATCH.get(cmd)
    if handler is None:
        writer.print_rich(
            f"[orx.status]Unknown command: {cmd}. Type /help[/orx.status]"
        )
        return
    await handler(
        state,
        cmd_arg,
        writer=writer,
        orx_path=orx_path,
        workspace=workspace,
    )
