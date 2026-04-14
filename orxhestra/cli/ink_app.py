"""pyink-based REPL for the orx CLI."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

from pyink import Box, Spacer, Static, Text, component, render
from pyink.hooks import use_animation, use_app, use_input, use_ref, use_state, use_window_size

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.cli.state import ReplState

import shutil as _shutil

_ACCENT = "#6C8EBF"
_MUTED = "#6c6c6c"
_SEPARATOR = "─" * (_shutil.get_terminal_size().columns - 1)

APPROVAL_OPTIONS = ["Yes", "Yes, allow all edits during this session (shift+tab)", "No"]


def _history_item(item, _index=0):
    """Render a single history item as a VNode."""
    t = item.get("type", "")
    ansi = item.get("ansi", "")
    if t == "user":
        return Box(
            Text("\u276f ", color=_ACCENT),
            Text(item["text"], bold=True),
            flex_direction="row",
            margin_top=1,
            margin_bottom=1,
        )
    if t == "response":
        return Box(
            Text("\u25cf ", color=_ACCENT),
            Text(ansi),
            flex_direction="row",
        )
    if t == "response":
        return Box(
            Text("\u25cf ", color=_ACCENT),
            Text(ansi),
            flex_direction="row",
        )
    if t == "rich":
        return Text(ansi)
    if t == "tool_done":
        return Box(Text(ansi), margin_bottom=1)
    if t == "plain":
        return Text(ansi, color=item.get("color"), dim=item.get("dim", False))
    if t == "separator":
        return Text(_SEPARATOR, color=_MUTED, dim=True)
    return Text(str(item))


@component
def _selector(prompt_text, options, selected_idx):
    """Approval selector (like Claude Code's permission prompt)."""
    rows = [
        Text(prompt_text, bold=True, color="#E5C07B"),
    ]
    rows.append(Text(""))
    for i, opt in enumerate(options):
        is_sel = i == selected_idx
        prefix = "\u276f" if is_sel else " "
        label = f"  {prefix} {i + 1}. {opt}"
        rows.append(Text(label, color="white" if is_sel else _MUTED, bold=is_sel))
    rows.append(Text(""))
    rows.append(Text("  Esc to cancel", color=_MUTED, dim=True))
    return Box(*rows, flex_direction="column", margin_top=1)


@component
def _autocomplete_menu(suggestions, selected_idx):
    """Autocomplete dropdown."""
    if not suggestions:
        return Text("")
    items = []
    for i, cmd in enumerate(suggestions):
        is_sel = i == selected_idx
        items.append(
            Text(f"  {cmd}", color="white" if is_sel else _MUTED, bold=is_sel, inverse=is_sel)
        )
    return Box(*items, flex_direction="column")


@component
def orx_repl(
    initial_history,
    command_names,
    spinner_frames,
    state_ref,
    console_ref,
    orx_path_ref,
    workspace_ref,
):
    win = use_window_size()

    history, set_history = use_state(initial_history)
    buf, set_buf = use_state("")
    cursor, set_cursor = use_state(0)
    phase, set_phase = use_state("idle")
    spinner_text, set_spinner_text = use_state("")
    stream_buf, set_stream = use_state("")
    # Approval selector state.
    approval_active, set_approval_active = use_state(False)
    approval_prompt, set_approval_prompt = use_state("")
    approval_sel, set_approval_sel = use_state(0)
    approval_event_ref = use_ref(None)
    approval_result_ref = use_ref(None)

    cmd_hist = use_ref([])
    hist_idx = use_ref(-1)
    running = use_ref(False)
    writer_ref = use_ref(None)
    ac_idx = use_ref(0)

    app = use_app()

    # Build writer once.
    if writer_ref.current is None:
        from orxhestra.cli.writer import InkWriter

        writer_ref.current = InkWriter(
            set_history=set_history,
            set_spinner_text=set_spinner_text,
            set_stream=set_stream,
            set_phase=set_phase,
            console=console_ref.current,
            approval_callback=_make_approval_callback(
                set_approval_active, set_approval_prompt, set_approval_sel,
                approval_event_ref, approval_result_ref,
            ),
        )

    # Spinner animation.
    anim = use_animation(interval=200, is_active=(phase == "spinning"))
    frame_idx = anim.frame % len(spinner_frames) if spinner_frames else 0
    frame_char = spinner_frames[frame_idx] if spinner_frames else ""

    # Autocomplete suggestions.
    suggestions = []
    if buf.lstrip().startswith("/") and phase == "idle" and not approval_active:
        prefix = buf.lstrip()
        suggestions = [c for c in command_names if c.startswith(prefix) and c != prefix]
    sel_idx = min(ac_idx.current, max(0, len(suggestions) - 1))

    def on_key(ch, key):
        # ── Approval mode ──
        if approval_active:
            if key.up_arrow:
                set_approval_sel(lambda i: max(0, i - 1))
                return
            if key.down_arrow:
                set_approval_sel(lambda i: min(len(APPROVAL_OPTIONS) - 1, i + 1))
                return
            if key.return_key:
                result = ["y", "a", "n"][approval_sel]
                approval_result_ref.current = result
                set_approval_active(False)
                if approval_event_ref.current:
                    approval_event_ref.current.set()
                return
            if key.escape:
                approval_result_ref.current = "n"
                set_approval_active(False)
                if approval_event_ref.current:
                    approval_event_ref.current.set()
                return
            return

        # ── Normal mode ──
        if key.ctrl and ch == "d":
            app.exit()
            return

        if key.ctrl and ch == "c":
            if running.current:
                running.current = False
                set_phase("idle")
                set_spinner_text("")
                set_stream("")
                set_history(lambda h: [
                    *h, {"type": "plain", "ansi": "  Interrupted.", "color": _MUTED},
                ])
            return

        # Tab — accept autocomplete.
        if key.tab:
            if suggestions:
                set_buf(suggestions[sel_idx] + " ")
                ac_idx.current = 0
            return

        # Enter.
        if key.return_key:
            if suggestions:
                set_buf(suggestions[sel_idx] + " ")
                ac_idx.current = 0
                return

            msg = buf.strip()
            if not msg:
                return
            set_buf("")
            set_cursor(0)
            cmd_hist.current = [*cmd_hist.current, msg]
            hist_idx.current = -1

            if msg.startswith("/"):
                _dispatch_slash(
                    msg, state_ref.current, writer_ref.current,
                    orx_path_ref.current, workspace_ref.current,
                    set_history, app,
                )
            elif not running.current:
                _dispatch_agent(
                    msg, state_ref.current, writer_ref.current,
                    set_history, set_phase, running, console_ref.current,
                )
            else:
                set_history(lambda h: [
                    *h, {"type": "plain", "ansi": "  Agent is still running.", "color": _MUTED},
                ])
            return

        # Arrow keys for autocomplete.
        if key.up_arrow and suggestions:
            ac_idx.current = max(0, ac_idx.current - 1)
            return
        if key.down_arrow and suggestions:
            ac_idx.current = min(len(suggestions) - 1, ac_idx.current + 1)
            return

        # History.
        if key.up_arrow:
            h = cmd_hist.current
            if h:
                i = hist_idx.current
                i = len(h) - 1 if i == -1 else max(0, i - 1)
                hist_idx.current = i
                set_buf(h[i])
            return
        if key.down_arrow:
            i = hist_idx.current
            h = cmd_hist.current
            if i >= 0:
                if i < len(h) - 1:
                    hist_idx.current = i + 1
                    set_buf(h[i + 1])
                else:
                    hist_idx.current = -1
                    set_buf("")
                    set_cursor(0)
            return

        # Left/right cursor movement.
        if key.left_arrow:
            set_cursor(lambda c: max(0, c - 1))
            return
        if key.right_arrow:
            set_cursor(lambda c: min(len(buf), c + 1))
            return

        # Home/End.
        if key.home:
            set_cursor(0)
            return
        if key.end:
            set_cursor(len(buf))
            return

        if key.backspace or key.delete:
            ac_idx.current = 0
            if cursor > 0:
                pos = cursor
                set_buf(lambda t: t[:pos - 1] + t[pos:])
                set_cursor(lambda c: max(0, c - 1))
            return

        if ch and not key.ctrl and not key.meta and not key.escape:
            ac_idx.current = 0
            pos = cursor
            set_buf(lambda t: t[:pos] + ch + t[pos:])
            set_cursor(lambda c: c + 1)

    use_input(on_key)

    # ── Build component tree ──
    children = [Static(items=history, render_item=_history_item)]

    # Spinner.
    if phase == "spinning" and spinner_text:
        children.append(Text(f"{frame_char} {spinner_text}", color=_ACCENT))

    # Streaming response.
    if phase == "streaming" and stream_buf:
        children.append(Box(
            Text("\u25cf ", color=_ACCENT),
            Text(stream_buf),
            flex_direction="row",
        ))

    # Approval selector.
    if approval_active:
        children.append(_selector(
            prompt_text=approval_prompt,
            options=APPROVAL_OPTIONS,
            selected_idx=approval_sel,
        ))

    # Push input to the bottom.
    children.append(Spacer())

    # Input area with border above and below.
    # Render input with cursor overlaying the character at position.
    before = buf[:cursor]
    char_at = buf[cursor] if cursor < len(buf) else " "
    after = buf[cursor + 1:] if cursor < len(buf) else ""
    input_children = [
        Text(_SEPARATOR, color=_MUTED, dim=True),
        Box(
            Text("  \u276f ", color=_ACCENT, bold=True),
            Text(before, bold=True),
            Text(char_at, bold=True, inverse=True),
            Text(after, bold=True),
            flex_direction="row",
        ),
    ]
    if suggestions and not approval_active:
        input_children.append(_autocomplete_menu(
            suggestions=suggestions[:8],
            selected_idx=sel_idx,
        ))
    input_children.append(Text(_SEPARATOR, color=_MUTED, dim=True))
    children.append(Box(*input_children, flex_direction="column"))

    return Box(*children, flex_direction="column", min_height=win.rows)


def _make_approval_callback(
    set_approval_active, set_approval_prompt, set_approval_sel,
    approval_event_ref, approval_result_ref,
):
    """Create an approval callback for the InkWriter."""
    def request_approval(label: str) -> str:
        evt = threading.Event()
        approval_event_ref.current = evt
        approval_result_ref.current = "n"
        set_approval_sel(0)
        set_approval_prompt(label)
        set_approval_active(True)
        evt.wait()
        return approval_result_ref.current or "n"
    return request_approval


def run_ink_app(
    state: ReplState,
    console: Console,
    orx_path: Any,
    workspace: str,
) -> None:
    """Start the pyink REPL (blocking — manages its own event loop)."""
    from pyink.fiber import Ref

    from orxhestra.cli.commands import get_command_names
    from orxhestra.cli.render import render_banner
    from orxhestra.cli.theme import ORX_SPINNER
    from orxhestra.cli.writer import rich_to_ansi

    banner_ansi = rich_to_ansi(
        console, render_banner(orx_path, state.model_name, workspace)
    )

    initial_history = [
        {"type": "rich", "ansi": banner_ansi},
        {"type": "plain", "ansi": "  type /help for commands, Ctrl+D to exit",
         "color": _MUTED, "dim": True},
        {"type": "separator"},
    ]

    vnode = orx_repl(
        initial_history=initial_history,
        command_names=get_command_names(),
        spinner_frames=ORX_SPINNER["frames"],
        state_ref=Ref(state),
        console_ref=Ref(console),
        orx_path_ref=Ref(orx_path),
        workspace_ref=Ref(workspace),
    )

    render(vnode)


def _dispatch_slash(text, state, writer, orx_path, workspace, set_history, app):
    from orxhestra.cli.commands import handle_slash_command

    parts = text.split(maxsplit=1)
    cmd_arg = parts[1].strip() if len(parts) > 1 else None
    set_history(lambda h: [*h, {"type": "plain", "ansi": f"  {text}", "color": _MUTED}])

    def run():
        loop = asyncio.new_event_loop()
        loop.run_until_complete(handle_slash_command(
            parts[0].lower(), cmd_arg, state,
            writer=writer, orx_path=orx_path, workspace=workspace,
        ))
        loop.close()
        if not state.should_continue:
            app.exit()

    threading.Thread(target=run, daemon=True).start()


def _dispatch_agent(message, state, writer, set_history, set_phase, running_ref, console):
    set_history(lambda h: [*h, {"type": "user", "text": message}])
    running_ref.current = True

    def run():
        try:
            from rich.markdown import Markdown
        except ImportError:
            set_history(lambda h: [*h, {"type": "plain", "ansi": "Error: rich required."}])
            running_ref.current = False
            return

        from orxhestra.cli.stream import stream_response

        loop = asyncio.new_event_loop()
        try:
            state.auto_approve = loop.run_until_complete(stream_response(
                state.runner, state.session_id, message,
                writer, Markdown,
                todo_list=state.todo_list,
                auto_approve=state.auto_approve,
            ))
            state.turn_count += 1
        except Exception as exc:
            err_msg = str(exc)
            set_history(lambda h: [*h, {"type": "plain", "ansi": f"  Error: {err_msg}"}])
        finally:
            running_ref.current = False
            set_phase("idle")
        loop.close()

    threading.Thread(target=run, daemon=True).start()
