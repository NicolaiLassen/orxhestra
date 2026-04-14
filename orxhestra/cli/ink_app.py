"""pyink-based REPL for the orx CLI."""

from __future__ import annotations

import asyncio
import re as _re
import shutil as _shutil
import threading
from typing import TYPE_CHECKING, Any

from pyink import Box, Spacer, Static, Text, component, render
from pyink.hooks import (
    use_animation,
    use_app,
    use_input,
    use_ref,
    use_state,
    use_window_size,
)

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.cli.state import ReplState

_ACCENT = "#6C8EBF"
_MUTED = "#6c6c6c"
_SEPARATOR = "\u2500" * (_shutil.get_terminal_size().columns - 1)

APPROVAL_OPTIONS = [
    "Yes",
    "Yes, allow all edits during this session",
    "No",
]
APPROVAL_RESULTS = ["y", "a", "n"]

_OPTION_RE = _re.compile(r"^\s*(\d+)\.\s+(.+)$", _re.MULTILINE)


def _parse_options(text: str) -> tuple[str, list[str]]:
    """Parse numbered options from a question string.

    Returns (question_text, options_list). If no options found,
    options_list is empty.
    """
    matches = _OPTION_RE.findall(text)
    if len(matches) < 2:
        return text, []
    # Extract the question (text before the first option).
    first_match = _OPTION_RE.search(text)
    question = text[:first_match.start()].strip() if first_match else text
    options = [m[1].strip() for m in matches]
    return question, options


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
    if t == "rich":
        return Text(ansi)
    if t == "tool_done":
        return Box(Text(ansi), margin_bottom=1)
    if t == "plain":
        return Text(
            ansi, color=item.get("color"), dim=item.get("dim", False),
        )
    if t == "separator":
        return Text(_SEPARATOR, color=_MUTED, dim=True)
    return Text(str(item))


@component
def _selector_view(prompt_text, options, selected_idx, show_type_option):
    """Numbered selector for approval prompts and human_input questions."""
    rows = [Text(prompt_text, bold=True, color="#E5C07B")]
    rows.append(Text(""))
    for i, opt in enumerate(options):
        is_sel = i == selected_idx
        prefix = "\u276f" if is_sel else " "
        rows.append(Text(
            f"  {prefix} {i + 1}. {opt}",
            color="white" if is_sel else _MUTED,
            bold=is_sel,
        ))
    if show_type_option:
        is_sel = selected_idx == len(options)
        prefix = "\u276f" if is_sel else " "
        rows.append(Text(
            f"  {prefix} {len(options) + 1}. Type something...",
            color="white" if is_sel else _MUTED,
            bold=is_sel,
        ))
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
        items.append(Text(
            f"  {cmd}",
            color="white" if is_sel else _MUTED,
            bold=is_sel,
            inverse=is_sel,
        ))
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
    selector_state_ref=None,
):
    win = use_window_size()

    history, set_history = use_state(initial_history)
    buf, set_buf = use_state("")
    cursor, set_cursor = use_state(0)
    phase, set_phase = use_state("idle")
    spinner_text, set_spinner_text = use_state("")
    stream_buf, set_stream = use_state("")

    # Selector state (approval + human_input questions).
    sel_active, set_sel_active = use_state(False)
    sel_prompt, set_sel_prompt = use_state("")
    sel_options, set_sel_options = use_state([])
    sel_idx, set_sel_idx = use_state(0)
    sel_mode = use_ref("approval")  # "approval" | "human_input"
    sel_show_type = use_ref(False)
    sel_event = use_ref(None)
    sel_result = use_ref(None)
    # Free-text input mode (when user picks "Type something").
    freetext, set_freetext = use_state(False)

    cmd_hist = use_ref([])
    hist_idx = use_ref(-1)
    running = use_ref(False)
    writer_ref = use_ref(None)
    ac_idx = use_ref(0)

    app = use_app()

    if writer_ref.current is None:
        from orxhestra.cli.writer import InkWriter

        selector_cb = _make_selector_callback(
            set_sel_active, set_sel_prompt, set_sel_options,
            set_sel_idx, sel_mode, sel_show_type,
            sel_event, sel_result,
        )

        writer_ref.current = InkWriter(
            set_history=set_history,
            set_spinner_text=set_spinner_text,
            set_stream=set_stream,
            set_phase=set_phase,
            console=console_ref.current,
            approval_callback=selector_cb,
        )

        # Register the callback so human_input tool uses the selector too.
        if selector_state_ref and selector_state_ref.current is not None:
            selector_state_ref.current["callback"] = selector_cb

        # Wire into permission system approval holder if present on state.
        _state = state_ref.current
        if hasattr(_state, "_approval_holder") and _state._approval_holder:
            _state._approval_holder["fn"] = selector_cb

    # Spinner animation.
    anim = use_animation(interval=200, is_active=(phase == "spinning"))
    fi = anim.frame % len(spinner_frames) if spinner_frames else 0
    frame_char = spinner_frames[fi] if spinner_frames else ""

    # Autocomplete suggestions.
    suggestions = []
    if (buf.lstrip().startswith("/")
            and phase == "idle"
            and not sel_active
            and not freetext):
        prefix = buf.lstrip()
        suggestions = [
            c for c in command_names
            if c.startswith(prefix) and c != prefix
        ]
    ac_sel = min(ac_idx.current, max(0, len(suggestions) - 1))

    # Total options count (including "Type something" if applicable).
    total_opts = len(sel_options) + (1 if sel_show_type.current else 0)

    def on_key(ch, key):
        # ── Free-text input mode (answering human_input) ──
        if freetext:
            if key.return_key:
                answer = buf.strip()
                if answer:
                    sel_result.current = answer
                    set_freetext(False)
                    set_buf("")
                    set_cursor(0)
                    if sel_event.current:
                        sel_event.current.set()
                return
            if key.escape:
                set_freetext(False)
                return
            # Fall through to normal text editing below.
            pass
        # ── Selector mode ──
        elif sel_active:
            if key.up_arrow:
                set_sel_idx(lambda i: max(0, i - 1))
                return
            if key.down_arrow:
                set_sel_idx(lambda i: min(total_opts - 1, i + 1))
                return
            if key.return_key:
                idx = sel_idx
                if sel_mode.current == "approval":
                    sel_result.current = APPROVAL_RESULTS[
                        min(idx, len(APPROVAL_RESULTS) - 1)
                    ]
                elif idx < len(sel_options):
                    sel_result.current = sel_options[idx]
                else:
                    # "Type something" selected.
                    set_sel_active(False)
                    set_freetext(True)
                    set_buf("")
                    set_cursor(0)
                    return
                set_sel_active(False)
                if sel_event.current:
                    sel_event.current.set()
                return
            if key.escape:
                sel_result.current = (
                    "n" if sel_mode.current == "approval" else ""
                )
                set_sel_active(False)
                if sel_event.current:
                    sel_event.current.set()
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
                    *h,
                    {"type": "plain", "ansi": "  Interrupted.",
                     "color": _MUTED},
                ])
            return

        # Tab — accept autocomplete.
        if key.tab:
            if suggestions:
                completed = suggestions[ac_sel] + " "
                set_buf(completed)
                set_cursor(len(completed))
                ac_idx.current = 0
            return

        # Enter.
        if key.return_key:
            if suggestions:
                completed = suggestions[ac_sel] + " "
                set_buf(completed)
                set_cursor(len(completed))
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
                    set_history, set_phase, running,
                )
            else:
                set_history(lambda h: [
                    *h,
                    {"type": "plain",
                     "ansi": "  Agent is still running.",
                     "color": _MUTED},
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
                set_cursor(len(h[i]))
            return
        if key.down_arrow:
            i = hist_idx.current
            h = cmd_hist.current
            if i >= 0:
                if i < len(h) - 1:
                    hist_idx.current = i + 1
                    set_buf(h[i + 1])
                    set_cursor(len(h[i + 1]))
                else:
                    hist_idx.current = -1
                    set_buf("")
                    set_cursor(0)
            return

        # Cursor movement.
        if key.left_arrow:
            set_cursor(lambda c: max(0, c - 1))
            return
        if key.right_arrow:
            set_cursor(lambda c: min(len(buf), c + 1))
            return
        if key.home:
            set_cursor(0)
            return
        if key.end:
            set_cursor(len(buf))
            return

        # Backspace.
        if key.backspace or key.delete:
            ac_idx.current = 0
            if cursor > 0:
                pos = cursor
                set_buf(lambda t: t[:pos - 1] + t[pos:])
                set_cursor(lambda c: max(0, c - 1))
            return

        # Regular character (or pasted text — ch may be multiple chars).
        if ch and not key.ctrl and not key.meta and not key.escape:
            ac_idx.current = 0
            text = ch
            set_cursor(lambda c: c + len(text))
            set_buf(lambda t: t + text if cursor >= len(t) else t[:cursor] + text + t[cursor:])

    use_input(on_key)

    # ── Build component tree ──
    children = [Static(items=history, render_item=_history_item)]

    # Spinner.
    if phase == "spinning" and spinner_text:
        children.append(
            Text(f"{frame_char} {spinner_text}", color=_ACCENT),
        )

    # Streaming response.
    if phase == "streaming" and stream_buf:
        children.append(Box(
            Text("\u25cf ", color=_ACCENT),
            Text(stream_buf),
            flex_direction="row",
        ))

    # Selector (approval or human_input).
    if sel_active:
        children.append(_selector_view(
            prompt_text=sel_prompt,
            options=sel_options,
            selected_idx=sel_idx,
            show_type_option=sel_show_type.current,
        ))

    # Push input to the bottom.
    children.append(Spacer())

    # Input area with border + cursor.
    before = buf[:cursor]
    char_at = buf[cursor] if cursor < len(buf) else " "
    after = buf[cursor + 1:] if cursor < len(buf) else ""

    prompt_label = "?" if freetext else "\u276f"

    input_children = [
        Text(_SEPARATOR, color=_MUTED, dim=True),
        Box(
            Text(f"  {prompt_label} ", color=_ACCENT, bold=True),
            Text(before, bold=True),
            Text(char_at, bold=True, inverse=True),
            Text(after, bold=True),
            flex_direction="row",
        ),
    ]
    if suggestions and not sel_active and not freetext:
        input_children.append(_autocomplete_menu(
            suggestions=suggestions[:8],
            selected_idx=ac_sel,
        ))
    input_children.append(Text(_SEPARATOR, color=_MUTED, dim=True))
    children.append(Box(*input_children, flex_direction="column"))

    return Box(*children, flex_direction="column", min_height=win.rows)


def _make_selector_callback(
    set_sel_active, set_sel_prompt, set_sel_options,
    set_sel_idx, sel_mode, sel_show_type,
    sel_event, sel_result,
):
    """Create a callback for the InkWriter's prompt_input."""
    def request_input(label: str) -> str:
        # Parse numbered options from the label.
        question, options = _parse_options(label)

        evt = threading.Event()
        sel_event.current = evt
        sel_result.current = ""
        set_sel_idx(0)

        if options:
            # Human input with parsed options.
            sel_mode.current = "human_input"
            sel_show_type.current = True
            set_sel_prompt(question)
            set_sel_options(options)
        else:
            # Tool approval prompt.
            sel_mode.current = "approval"
            sel_show_type.current = False
            set_sel_prompt(label)
            set_sel_options(APPROVAL_OPTIONS)

        set_sel_active(True)
        evt.wait()
        return sel_result.current or ""
    return request_input


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
        console, render_banner(orx_path, state.model_name, workspace),
    )

    initial_history = [
        {"type": "rich", "ansi": banner_ansi},
        {"type": "plain",
         "ansi": "  type /help for commands, Ctrl+D to exit",
         "color": _MUTED, "dim": True},
        {"type": "separator"},
    ]

    # Create the selector callback before render so we can rewire human_input.
    sel_state = {
        "set_sel_active": None,
        "set_sel_prompt": None,
        "set_sel_options": None,
        "set_sel_idx": None,
        "sel_mode": None,
        "sel_show_type": None,
        "sel_event": None,
        "sel_result": None,
    }

    def _human_input_via_selector(question: str) -> str:
        """Blocking callback for the human_input tool."""
        cb = sel_state.get("callback")
        if cb:
            return cb(question)
        return input(f"\n  ? {question}\n  > ")

    # Rewire the human_input tool callbacks to use our selector.
    from orxhestra.cli.builder import _set_human_input_callbacks

    async def _async_human_input(question: str) -> str:
        return _human_input_via_selector(question)

    _set_human_input_callbacks(state.runner.agent, _async_human_input)

    vnode = orx_repl(
        initial_history=initial_history,
        command_names=get_command_names(),
        spinner_frames=ORX_SPINNER["frames"],
        state_ref=Ref(state),
        console_ref=Ref(console),
        orx_path_ref=Ref(orx_path),
        workspace_ref=Ref(workspace),
        selector_state_ref=Ref(sel_state),
    )

    render(vnode, max_fps=15)


def _dispatch_slash(text, state, writer, orx_path, workspace,
                    set_history, app):
    from orxhestra.cli.commands import handle_slash_command

    parts = text.split(maxsplit=1)
    cmd_arg = parts[1].strip() if len(parts) > 1 else None
    set_history(lambda h: [
        *h, {"type": "plain", "ansi": f"  {text}", "color": _MUTED},
    ])

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


def _dispatch_agent(message, state, writer, set_history,
                    set_phase, running_ref):
    set_history(lambda h: [*h, {"type": "user", "text": message}])
    running_ref.current = True

    def run():
        try:
            from rich.markdown import Markdown
        except ImportError:
            set_history(lambda h: [
                *h, {"type": "plain", "ansi": "Error: rich required."},
            ])
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
            set_history(lambda h: [
                *h, {"type": "plain", "ansi": f"  Error: {err_msg}"},
            ])
        finally:
            running_ref.current = False
            set_phase("idle")
        loop.close()

    threading.Thread(target=run, daemon=True).start()
