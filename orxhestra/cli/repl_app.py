"""Interactive REPL using prompt_toolkit Application.

Provides a split-screen layout with scrollable output above and
an always-active input field at the bottom.  The user can type
while the agent streams output.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from orxhestra.cli.config import HISTORY_DIR, HISTORY_FILE

if TYPE_CHECKING:
    from io import StringIO

    from rich.console import Console

    from orxhestra.cli.state import ReplState


def _rich_to_formatted(console: Console, *args: Any, **kwargs: Any) -> list[tuple[str, str]]:
    """Render Rich output to prompt_toolkit formatted text tuples."""
    from io import StringIO

    buf = StringIO()
    from rich.console import Console as _Console

    tmp = _Console(file=buf, force_terminal=True, width=console.width or 120, no_color=False)
    tmp.print(*args, **kwargs)
    return list(to_formatted_text(ANSI(buf.getvalue())))


class ReplApp:
    """prompt_toolkit-based REPL with concurrent input and output."""

    def __init__(
        self,
        state: ReplState,
        console: Console,
        orx_path: Any,
        workspace: str,
        *,
        auto_approve: bool = False,
        prompt_label: str = "orx",
    ) -> None:
        self.state = state
        self.console = console
        self.orx_path = orx_path
        self.workspace = workspace
        self.auto_approve = auto_approve
        self.prompt_label = prompt_label
        self._agent_running = False

        # Output buffer — list of (style, text) tuples.
        self._output: list[tuple[str, str]] = []
        self._app: Application | None = None

    # ── Output management ────────────────────────────────────────

    def _get_output(self) -> list[tuple[str, str]]:
        return self._output or [("", "")]

    def _get_cursor_pos(self) -> Point:
        text = "".join(t for _, t in self._output)
        return Point(0, text.count("\n"))

    def write(self, text: str, style: str = "") -> None:
        """Append plain text to the output pane."""
        self._output.append((style, text))
        if self._app:
            self._app.invalidate()

    def write_rich(self, *args: Any, **kwargs: Any) -> None:
        """Render Rich content to the output pane."""
        frags = _rich_to_formatted(self.console, *args, **kwargs)
        self._output.extend(frags)
        if self._app:
            self._app.invalidate()

    def write_line(self, text: str = "", style: str = "") -> None:
        """Append a line to the output pane."""
        self.write(text + "\n", style)

    # ── Build the application ────────────────────────────────────

    def build(self) -> Application:
        from orxhestra.cli.commands import get_command_names

        output_control = FormattedTextControl(
            text=self._get_output,
            focusable=False,
            show_cursor=False,
            get_cursor_position=self._get_cursor_pos,
        )
        output_window = Window(
            content=output_control,
            wrap_lines=True,
        )

        command_names = get_command_names()
        completer = WordCompleter(command_names, sentence=True)

        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        input_field = TextArea(
            height=1,
            prompt=f"{self.prompt_label}> ",
            multiline=False,
            wrap_lines=False,
            completer=completer,
            history=FileHistory(str(HISTORY_FILE)),
        )
        input_field.accept_handler = self._on_submit

        kb = KeyBindings()

        @kb.add("c-c", eager=True)
        def _interrupt(event: Any) -> None:
            if self._agent_running:
                self.write_line("  Interrupted.", "class:muted")
                self._agent_running = False
            else:
                event.app.exit()

        @kb.add("c-d", eager=True)
        def _quit(event: Any) -> None:
            event.app.exit()

        container = HSplit([
            output_window,
            Window(height=1, char="\u2500", style="class:separator"),
            input_field,
        ])

        style = Style([
            ("separator", "#4a4a4a"),
            ("class:muted", "#6c6c6c"),
        ])

        self._app = Application(
            layout=Layout(container, focused_element=input_field),
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
        )
        return self._app

    # ── Input handling ───────────────────────────────────────────

    def _on_submit(self, buff: Any) -> None:
        text = buff.text.strip()
        if not text:
            return

        if text.startswith("/"):
            if self._app:
                self._app.create_background_task(self._handle_slash(text))
            return

        if self._agent_running:
            self.write_line("  Agent is still running.", "class:muted")
            return

        if self._app:
            self._app.create_background_task(self._run_agent(text))

    async def _handle_slash(self, text: str) -> None:
        from orxhestra.cli.commands import handle_slash_command

        parts = text.split(maxsplit=1)
        cmd_arg = parts[1].strip() if len(parts) > 1 else None

        self.write_line(f"  {text}", "class:muted")

        await handle_slash_command(
            parts[0].lower(),
            cmd_arg,
            self.state,
            console=self.console,
            orx_path=self.orx_path,
            workspace=self.workspace,
        )

        if not self.state.should_continue and self._app:
            self._app.exit()

    async def _run_agent(self, message: str) -> None:
        from orxhestra.cli.stream import stream_response

        try:
            from rich.markdown import Markdown
        except ImportError:
            self.write_line("Error: rich is required.", "class:muted")
            return

        self.write_line(f"\n> {message}\n", "bold")
        self._agent_running = True

        try:
            self.auto_approve = await stream_response(
                self.state.runner,
                self.state.session_id,
                message,
                self.console,
                Markdown,
                todo_list=self.state.todo_list,
                auto_approve=self.auto_approve,
            )
            self.state.turn_count += 1
        except Exception as exc:
            self.write_line(f"  Error: {exc}", "class:muted")
        finally:
            self._agent_running = False
            if self._app:
                self._app.invalidate()
