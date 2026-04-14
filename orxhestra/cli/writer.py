"""Writer abstraction for CLI output.

All CLI output flows through the ``Writer`` protocol. Two
implementations:

- ``ConsoleWriter`` — Rich console for ``-c`` single-shot mode.
- ``InkWriter`` — pushes events into pyink component state
  for the React-style TUI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console


class SpinnerHandle(Protocol):
    def update_text(self, text: str) -> None: ...
    def stop(self) -> None: ...


class LiveHandle(Protocol):
    def update(self, renderable: Any) -> None: ...
    def stop(self, *, keep: bool = True) -> None: ...


@runtime_checkable
class Writer(Protocol):
    def print_rich(self, *args: Any, **kwargs: Any) -> None: ...
    def start_spinner(self, text: str) -> SpinnerHandle: ...
    def start_live(self) -> LiveHandle: ...
    async def prompt_input(self, label: str) -> str: ...


def rich_to_ansi(console: Console, *args: Any, **kwargs: Any) -> str:
    """Render Rich content to a raw ANSI string."""
    from io import StringIO

    from rich.console import Console as _Console

    from orxhestra.cli.theme import DARK_THEME, LIGHT_THEME, detect_terminal_theme

    buf = StringIO()
    theme = LIGHT_THEME if detect_terminal_theme() == "light" else DARK_THEME
    tmp = _Console(
        file=buf,
        force_terminal=True,
        width=console.width or 120,
        no_color=False,
        theme=theme,
    )
    tmp.print(*args, **kwargs)
    return buf.getvalue().rstrip("\n")


# ── ConsoleWriter (for -c single-shot mode) ─────────────────────


class _ConsoleSpinner:
    def __init__(self, status: Any) -> None:
        self._status = status

    def update_text(self, text: str) -> None:
        if self._status:
            from orxhestra.cli.theme import ACCENT
            self._status.update(f"  [{ACCENT}]{text}[/{ACCENT}]")

    def stop(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None


class _ConsoleLive:
    def __init__(self, live: Any) -> None:
        self._live = live

    def update(self, renderable: Any) -> None:
        if self._live:
            self._live.update(renderable)

    def stop(self, *, keep: bool = True) -> None:
        if self._live:
            self._live.stop()
            self._live = None


class ConsoleWriter:
    """Writer that delegates to a Rich Console (for -c mode)."""

    def __init__(self, console: Console) -> None:
        self.console = console

    def print_rich(self, *args: Any, **kwargs: Any) -> None:
        self.console.print(*args, **kwargs)

    def start_spinner(self, text: str) -> _ConsoleSpinner:
        from orxhestra.cli.theme import ACCENT
        try:
            from rich.spinner import SPINNERS
            from rich.status import Status
        except ImportError:
            return _ConsoleSpinner(None)
        from orxhestra.cli.theme import ORX_SPINNER
        SPINNERS["orx_music"] = ORX_SPINNER
        status = Status(
            f"  [{ACCENT}]{text}[/{ACCENT}]",
            console=self.console, spinner="orx_music", spinner_style=ACCENT,
        )
        status.start()
        return _ConsoleSpinner(status)

    def stop_spinner(self, handle: _ConsoleSpinner) -> None:
        handle.stop()

    def start_live(self) -> _ConsoleLive:
        try:
            from rich.live import Live
        except ImportError:
            return _ConsoleLive(None)
        live = Live(console=self.console, refresh_per_second=8, transient=False)
        live.start()
        return _ConsoleLive(live)

    def stop_live(self, handle: _ConsoleLive, *, keep: bool = True) -> None:
        handle.stop(keep=keep)

    async def prompt_input(self, label: str) -> str:
        return input(label)


# ── InkWriter (for pyink TUI mode) ──────────────────────────────


class _InkSpinnerHandle:
    def __init__(self, set_spinner_text: Callable, set_phase: Callable) -> None:
        self._set_spinner_text = set_spinner_text
        self._set_phase = set_phase

    def update_text(self, text: str) -> None:
        self._set_spinner_text(text)

    def stop(self) -> None:
        self._set_phase("idle")
        self._set_spinner_text("")


class _InkLiveHandle:
    _MIN_INTERVAL = 1.0 / 12  # throttle to ~12 FPS

    def __init__(
        self,
        set_stream: Callable,
        set_phase: Callable,
        set_history: Callable,
        console: Console,
    ) -> None:
        self._set_stream = set_stream
        self._set_phase = set_phase
        self._set_history = set_history
        self._console = console
        self._last_renderable: Any = None
        self._last_update = 0.0

    def update(self, renderable: Any) -> None:
        import time
        self._last_renderable = renderable
        now = time.monotonic()
        if now - self._last_update < self._MIN_INTERVAL:
            return
        self._last_update = now
        ansi = rich_to_ansi(self._console, renderable)
        self._set_stream(ansi)

    def stop(self, *, keep: bool = True) -> None:
        if keep and self._last_renderable is not None:
            ansi = rich_to_ansi(self._console, self._last_renderable)
            if ansi:
                self._set_history(lambda h: [*h, {"type": "response", "ansi": ansi}])
        self._set_phase("idle")
        self._set_stream("")


class InkWriter:
    """Writer that pushes events into pyink component state."""

    def __init__(
        self,
        set_history: Callable,
        set_spinner_text: Callable,
        set_stream: Callable,
        set_phase: Callable,
        console: Console,
        approval_callback: Callable | None = None,
    ) -> None:
        self._set_history = set_history
        self._set_spinner_text = set_spinner_text
        self._set_stream = set_stream
        self._set_phase = set_phase
        self._console = console
        self._approval_callback = approval_callback

    def print_rich(self, *args: Any, **kwargs: Any) -> None:
        ansi = rich_to_ansi(self._console, *args, **kwargs)
        if ansi:
            # Tag tool response lines (└) so the UI can add margin after them.
            item_type = "tool_done" if "\u2514" in ansi else "rich"
            self._set_history(lambda h: [*h, {"type": item_type, "ansi": ansi}])

    def start_spinner(self, text: str) -> _InkSpinnerHandle:
        self._set_phase("spinning")
        self._set_spinner_text(text)
        return _InkSpinnerHandle(self._set_spinner_text, self._set_phase)

    def stop_spinner(self, handle: _InkSpinnerHandle) -> None:
        handle.stop()

    def start_live(self) -> _InkLiveHandle:
        self._set_phase("streaming")
        self._set_stream("")
        return _InkLiveHandle(
            self._set_stream, self._set_phase, self._set_history, self._console,
        )

    def stop_live(self, handle: _InkLiveHandle, *, keep: bool = True) -> None:
        handle.stop(keep=keep)

    async def prompt_input(self, label: str) -> str:
        if self._approval_callback:
            return self._approval_callback(label)
        return input(label)
