"""pyink-based REPL for the orx CLI."""

from __future__ import annotations

import asyncio
import re as _re
import shutil as _shutil
import threading
from collections import deque
from typing import TYPE_CHECKING, Any

from pyink import Box, Static, Text, component, render
from pyink.hooks import (
    use_animation,
    use_app,
    use_input,
    use_ref,
    use_state,
)

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.cli.state import ReplState

from orxhestra.cli.theme import (
    BRAND_AMBER,
    BRAND_PAPER,
    BRAND_SIGNAL,
    BRAND_WHISPER,
)

_ACCENT = BRAND_SIGNAL
_MUTED = BRAND_WHISPER
_HIGHLIGHT = BRAND_PAPER
_PROMPT = BRAND_AMBER

# Pre-baked ANSI escape for the response bullet — pyink Text doesn't
# pass ANSI through cleanly when colour is set on a flex child, so we
# embed the escape directly. Derived once from BRAND_SIGNAL (#3FE0A8 →
# rgb(63, 224, 168)).
_SIGNAL_RGB = tuple(int(BRAND_SIGNAL[i : i + 2], 16) for i in (1, 3, 5))
_SIGNAL_ANSI_ON = f"\x1b[38;2;{_SIGNAL_RGB[0]};{_SIGNAL_RGB[1]};{_SIGNAL_RGB[2]}m"
_SIGNAL_ANSI_OFF = "\x1b[0m"

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
        # Prepend bullet directly to avoid flex-row spacing issues
        # where ANSI codes can consume the space between ● and text.
        return Text(f"{_SIGNAL_ANSI_ON}\u25cf{_SIGNAL_ANSI_OFF} {ansi}")
    if t == "rich":
        return Text(ansi)
    if t in ("tool_done", "tool_done_last"):
        return Text(ansi)
    if t == "tasks":
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
    rows = [Text(prompt_text, bold=True, color=_PROMPT)]
    rows.append(Text(""))
    for i, opt in enumerate(options):
        is_sel = i == selected_idx
        prefix = "\u276f" if is_sel else " "
        rows.append(Text(
            f"  {prefix} {i + 1}. {opt}",
            color=_HIGHLIGHT if is_sel else _MUTED,
            bold=is_sel,
        ))
    if show_type_option:
        is_sel = selected_idx == len(options)
        prefix = "\u276f" if is_sel else " "
        rows.append(Text(
            f"  {prefix} {len(options) + 1}. Type something...",
            color=_HIGHLIGHT if is_sel else _MUTED,
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
            color=_HIGHLIGHT if is_sel else _MUTED,
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

    # Pending message queue: messages typed while the agent is running
    # are appended here and drained in FIFO order by _dispatch_agent
    # once the current turn completes. Queued messages render as
    # floating user lines in the dynamic area (between stream and
    # input) — NOT in scrollback — so the in-flight response's final
    # chunk is never committed after a queued user line. When the
    # worker dequeues a message it pushes the user line into history
    # at the correct moment.
    pending_msgs = use_ref(deque())
    queue_lock = use_ref(threading.Lock())
    # Snapshot of pending_msgs for rendering. Lives in state (not ref)
    # so mutations trigger a re-render. Updated under queue_lock so
    # the snapshot and the deque can never disagree on order.
    pending_view, set_pending_view = use_state(())

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
        # Debug helper: ORX_KEY_DEBUG=1 surfaces every keypress as a
        # muted history line so we can see exactly what bytes the
        # terminal sends for whatever shortcut we're trying to bind.
        import os as _os
        if _os.environ.get("ORX_KEY_DEBUG"):
            flags = [
                name for name in (
                    "shift", "ctrl", "meta", "super_key", "tab",
                    "return_key", "escape", "backspace", "delete",
                    "up_arrow", "down_arrow", "left_arrow", "right_arrow",
                    "home", "end",
                ) if getattr(key, name, False)
            ]
            ch_repr = repr(ch) if ch else "''"
            line = f"  key: ch={ch_repr} flags={'|'.join(flags) or '-'}"
            set_history(lambda h, ln=line: [
                *h, {"type": "plain", "ansi": ln, "color": _MUTED, "dim": True},
            ])

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
                with queue_lock.current:
                    pending_msgs.current.clear()
                    set_pending_view(())
                    running.current = False
                w = writer_ref.current
                if w is not None and hasattr(w, "cancel_live"):
                    w.cancel_live()
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

        # Multi-line newline insert. Different terminals send wildly
        # different bytes for "modified Enter":
        #   - kitty/ghostty (with kitty keyboard protocol): key.shift
        #     gets set on the parsed Enter.
        #   - iTerm2 with "Send Esc+" or Terminal.app with "Use Option
        #     as Meta key": Alt/Option+Enter sends "\x1b\r".
        #   - Some emit "\x1bOM" (xterm SS3 enter).
        #   - Kitty CSI u format: "\x1b[13;<mod>u" for modified Enter.
        # Universal fallback: type "\" then press Enter — the trailing
        # backslash is replaced with a newline (popular in shells).
        is_modified_enter = (
            (key.return_key and (key.shift or key.meta or key.ctrl))
            or ch in ("\x1b\r", "\x1b\n", "\x1bOM")
            or (ch and ch.startswith("\x1b[13;") and ch.endswith("u"))
        )
        backslash_enter = (
            key.return_key
            and not (key.shift or key.meta or key.ctrl)
            and cursor > 0
            and cursor <= len(buf)
            and buf[cursor - 1] == "\\"
            and not suggestions
        )
        if is_modified_enter:
            pos = cursor
            set_buf(lambda t, p=pos: t[:p] + "\n" + t[p:])
            set_cursor(lambda c: c + 1)
            ac_idx.current = 0
            return
        if backslash_enter:
            # Replace the trailing "\" with a newline. Cursor stays at
            # the same offset (was after "\", now after "\n").
            pos = cursor
            set_buf(lambda t, p=pos: t[:p - 1] + "\n" + t[p:])
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
            else:
                # Atomically check running state and either dispatch
                # immediately or enqueue. Same lock the worker uses, so
                # the producer never races with the consumer's drain
                # path. Queued messages render in the floating dynamic
                # area (via pending_view) — they get pushed into
                # scrollback only when the worker actually dequeues
                # them, so the previous turn's response is always fully
                # committed before the next user line lands in history.
                with queue_lock.current:
                    if running.current:
                        pending_msgs.current.append(msg)
                        snapshot = tuple(pending_msgs.current)
                        set_pending_view(snapshot)
                        return
                    running.current = True
                set_history(lambda h, m=msg: [
                    *h, {"type": "user", "text": m},
                ])
                _dispatch_agent(
                    msg, state_ref.current, writer_ref.current,
                    set_history, set_phase, running,
                    pending_msgs, queue_lock, set_pending_view,
                )
            return

        # Arrow keys for autocomplete.
        if key.up_arrow and suggestions:
            ac_idx.current = max(0, ac_idx.current - 1)
            return
        if key.down_arrow and suggestions:
            ac_idx.current = min(len(suggestions) - 1, ac_idx.current + 1)
            return

        # Multi-line cursor navigation. When the buffer spans multiple
        # lines, up/down move between lines first; only fall through to
        # history nav when the cursor is on the top/bottom line.
        if key.up_arrow and "\n" in buf:
            buf_lines = buf.split("\n")
            remaining = cursor
            cur_line, cur_col = 0, 0
            for i, line in enumerate(buf_lines):
                if remaining <= len(line):
                    cur_line, cur_col = i, remaining
                    break
                remaining -= len(line) + 1
            else:
                cur_line, cur_col = len(buf_lines) - 1, len(buf_lines[-1])
            if cur_line > 0:
                target_col = min(cur_col, len(buf_lines[cur_line - 1]))
                offset = sum(len(l) + 1 for l in buf_lines[:cur_line - 1]) + target_col
                set_cursor(offset)
                return
        if key.down_arrow and "\n" in buf:
            buf_lines = buf.split("\n")
            remaining = cursor
            cur_line, cur_col = 0, 0
            for i, line in enumerate(buf_lines):
                if remaining <= len(line):
                    cur_line, cur_col = i, remaining
                    break
                remaining -= len(line) + 1
            else:
                cur_line, cur_col = len(buf_lines) - 1, len(buf_lines[-1])
            if cur_line < len(buf_lines) - 1:
                target_col = min(cur_col, len(buf_lines[cur_line + 1]))
                offset = sum(len(l) + 1 for l in buf_lines[:cur_line + 1]) + target_col
                set_cursor(offset)
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
            # Jump to start of current line (or buffer if single-line).
            if "\n" in buf:
                line_start = buf.rfind("\n", 0, cursor) + 1
                set_cursor(line_start)
            else:
                set_cursor(0)
            return
        if key.end:
            # Jump to end of current line (or buffer if single-line).
            if "\n" in buf:
                next_nl = buf.find("\n", cursor)
                set_cursor(len(buf) if next_nl == -1 else next_nl)
            else:
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
    # Claude Code pattern: Static for completed history (terminal scrollback),
    # dynamic Box below for ephemeral content (redrawn each frame).
    # Terminal's native scrollbar handles scrolling automatically.

    # Dynamic content (redrawn each frame)
    dynamic = []

    # Spinner.
    if phase == "spinning" and spinner_text:
        dynamic.append(
            Text(f"{frame_char} {spinner_text}", color=_ACCENT),
        )

    # Streaming response.
    if phase == "streaming" and stream_buf:
        dynamic.append(Box(
            Text("\u25cf ", color=_ACCENT),
            Text(stream_buf),
            flex_direction="row",
        ))

    # Floating queued user messages — typed while the agent is
    # running. They live here (not in scrollback) until the worker
    # drains them, so the in-flight response is always committed to
    # history before any queued user-line lands there.
    if pending_view:
        for queued in pending_view:
            dynamic.append(Box(
                Text("❯ ", color=_ACCENT),
                Text(queued, bold=True),
                flex_direction="row",
                margin_top=1,
            ))

    # Selector (approval or human_input).
    if sel_active:
        dynamic.append(_selector_view(
            prompt_text=sel_prompt,
            options=sel_options,
            selected_idx=sel_idx,
            show_type_option=sel_show_type.current,
        ))

    # Input area with border + cursor. For multi-line buffers, render
    # one row per line; the first row carries the \u276f prompt, the rest
    # use a blank-aligned indent so the columns line up.
    prompt_label = "?" if freetext else "\u276f"

    lines = buf.split("\n") if buf else [""]
    # Map cursor offset \u2192 (line_idx, col_idx).
    remaining = cursor
    cursor_line = 0
    cursor_col = 0
    for i, line in enumerate(lines):
        if remaining <= len(line):
            cursor_line, cursor_col = i, remaining
            break
        remaining -= len(line) + 1  # +1 for the \n
    else:
        cursor_line = len(lines) - 1
        cursor_col = len(lines[-1])

    rows = []
    for i, line in enumerate(lines):
        prompt = f"  {prompt_label} " if i == 0 else "    "
        if i == cursor_line:
            before = line[:cursor_col]
            char_at = line[cursor_col] if cursor_col < len(line) else " "
            after = line[cursor_col + 1:] if cursor_col < len(line) else ""
            rows.append(Box(
                Text(prompt, color=_ACCENT, bold=True),
                Text(before, bold=True),
                Text(char_at, bold=True, inverse=True),
                Text(after, bold=True),
                flex_direction="row",
            ))
        else:
            rows.append(Box(
                Text(prompt, color=_ACCENT, bold=True),
                Text(line, bold=True),
                flex_direction="row",
            ))

    input_children = [
        Text(_SEPARATOR, color=_MUTED, dim=True),
        Box(*rows, flex_direction="column"),
    ]
    if suggestions and not sel_active and not freetext:
        input_children.append(_autocomplete_menu(
            suggestions=suggestions[:8],
            selected_idx=ac_sel,
        ))
    input_children.append(Text(_SEPARATOR, color=_MUTED, dim=True))
    dynamic.append(Box(*input_children, flex_direction="column"))

    return Box(
        Static(items=history, render_item=_history_item),
        Box(*dynamic, flex_direction="column"),
        flex_direction="column",
    )


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
        console,
        render_banner(
            orx_path,
            state.model_name,
            workspace,
            signer_did=state.signer_did,
        ),
    )

    initial_history: list = [
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

    render(vnode, max_fps=30)


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
                    set_phase, running_ref,
                    pending_msgs_ref, queue_lock_ref, set_pending_view):
    """Spawn the worker thread that drains the user-message queue.

    The caller has already pushed the user-line for ``message`` to
    history and flipped ``running_ref.current`` to True under
    ``queue_lock_ref.current``. The worker runs the turn, then loops:
    pop the next pending message under the lock, push *its* user-line
    to history (so it lands AFTER the prior turn's response has been
    committed by the live handle's ``stop``), and run it. When the
    queue is empty the worker flips ``running_ref.current`` to False
    under the same lock — so the Enter handler's check-and-enqueue is
    race-free. The lock is only held for O(1) work, never across
    ``stream_response``.
    """

    def run():
        try:
            from rich.markdown import Markdown
        except ImportError:
            set_history(lambda h: [
                *h, {"type": "plain", "ansi": "Error: rich required."},
            ])
            with queue_lock_ref.current:
                pending_msgs_ref.current.clear()
                set_pending_view(())
                running_ref.current = False
            return

        from orxhestra.cli.stream import stream_response

        loop = asyncio.new_event_loop()
        current = message
        try:
            while current is not None:
                try:
                    state.auto_approve = loop.run_until_complete(stream_response(
                        state.runner, state.session_id, current,
                        writer, Markdown,
                        todo_list=state.todo_list,
                        auto_approve=state.auto_approve,
                    ))
                    state.turn_count += 1
                except Exception as exc:
                    err_msg = str(exc)
                    if len(err_msg) > 200:
                        err_msg = err_msg[:200] + "..."
                    set_history(lambda h, e=err_msg: [
                        *h, {"type": "plain",
                             "ansi": f"Error: {e}", "color": "red"},
                    ])
                    # Drop the queue on error so we don't loop on
                    # whatever caused the failure.
                    with queue_lock_ref.current:
                        pending_msgs_ref.current.clear()
                        set_pending_view(())
                        running_ref.current = False
                    current = None
                    break

                # Atomically: either drain the queue and stay running,
                # or flip running=False under the same lock so the
                # Enter handler's check-and-enqueue is race-free.
                # Drain semantics: messages typed while the previous
                # turn was streaming all show in history as user-lines,
                # but only the LAST one triggers a new completion. The
                # intermediate ones are treated as superseded thoughts.
                with queue_lock_ref.current:
                    if pending_msgs_ref.current:
                        drained = list(pending_msgs_ref.current)
                        pending_msgs_ref.current.clear()
                        set_pending_view(())
                    else:
                        running_ref.current = False
                        drained = None
                if drained:
                    for line in drained:
                        set_history(lambda h, m=line: [
                            *h, {"type": "user", "text": m},
                        ])
                    current = drained[-1]
                else:
                    current = None
        finally:
            set_phase("idle")
            # Cancel pending tasks before closing to avoid
            # "Task was destroyed but it is pending" warnings.
            for task in asyncio.all_tasks(loop):
                task.cancel()
            try:
                leftover = asyncio.all_tasks(loop)
                loop.run_until_complete(
                    asyncio.gather(*leftover, return_exceptions=True),
                )
            except Exception:
                pass
            loop.close()

    threading.Thread(target=run, daemon=True).start()
