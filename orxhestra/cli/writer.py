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
    """Handle returned by ``Writer.start_spinner``.

    Attributes
    ----------
    update_text : callable
        Change the spinner label text.
    stop : callable
        Cancel the spinner and remove it from the display.
    """

    def update_text(self, text: str) -> None: ...
    def stop(self) -> None: ...


class LiveHandle(Protocol):
    """Handle returned by ``Writer.start_live``.

    Attributes
    ----------
    update : callable
        Replace the live region content with a new renderable.
    stop : callable
        End the live region, optionally freezing content to history.
    """

    def update(self, renderable: Any) -> None: ...
    def stop(self, *, keep: bool = True) -> None: ...


@runtime_checkable
class Writer(Protocol):
    """Unified output interface for the CLI.

    All rendering code (``stream.py``, ``render.py``, ``commands.py``)
    writes through this protocol so the same logic works for both the
    pyink TUI and the Rich console fallback.
    """

    def print_rich(self, *args: Any, **kwargs: Any) -> None: ...
    def start_spinner(self, text: str) -> SpinnerHandle: ...
    def start_live(self) -> LiveHandle: ...
    async def prompt_input(self, label: str) -> str: ...


def rich_to_ansi(console: Console, *args: Any, **kwargs: Any) -> str:
    """Render Rich content to a raw ANSI string.

    Parameters
    ----------
    console : Console
        Rich console used for width and theme detection.
    *args : Any
        Positional arguments forwarded to ``Console.print``.
    **kwargs : Any
        Keyword arguments forwarded to ``Console.print``.

    Returns
    -------
    str
        ANSI-escaped string with trailing newline stripped.
    """
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
    # Strip trailing whitespace per line — Rich pads each line to
    # console width with spaces, which breaks yoga flex-row layout
    # (the padded text is measured as full-width, squeezing siblings).
    raw = buf.getvalue().rstrip("\n")
    return "\n".join(line.rstrip() for line in raw.split("\n"))


class _ConsoleSpinner:
    """Spinner wrapping Rich ``Status``."""

    def __init__(self, status: Any) -> None:
        self._status = status

    def update_text(self, text: str) -> None:
        """Change the spinner label."""
        if self._status:
            from orxhestra.cli.theme import ACCENT

            self._status.update(f"  [{ACCENT}]{text}[/{ACCENT}]")

    def stop(self) -> None:
        """Stop and remove the spinner."""
        if self._status:
            self._status.stop()
            self._status = None


class _ConsoleLive:
    """Live region wrapping Rich ``Live``."""

    def __init__(self, live: Any) -> None:
        self._live = live

    def update(self, renderable: Any) -> None:
        """Replace the live display content."""
        if self._live:
            self._live.update(renderable)

    def stop(self, *, keep: bool = True) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None


class ConsoleWriter:
    """Writer that delegates directly to a Rich ``Console``.

    Used for ``-c`` single-shot mode.

    Parameters
    ----------
    console : Console
        Rich console to write to.
    """

    def __init__(self, console: Console) -> None:
        self.console = console

    def print_rich(self, *args: Any, **kwargs: Any) -> None:
        """Print Rich content to the console."""
        kwargs.pop("item_type", None)
        self.console.print(*args, **kwargs)

    def start_spinner(self, text: str) -> _ConsoleSpinner:
        """Start a Rich ``Status`` spinner.

        Parameters
        ----------
        text : str
            Status text shown next to the spinner.

        Returns
        -------
        _ConsoleSpinner
            Handle to update or stop the spinner.
        """
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
            console=self.console,
            spinner="orx_music",
            spinner_style=ACCENT,
        )
        status.start()
        return _ConsoleSpinner(status)

    def stop_spinner(self, handle: _ConsoleSpinner) -> None:
        """Stop a spinner."""
        handle.stop()

    def start_live(self) -> _ConsoleLive:
        """Start a Rich ``Live`` display.

        Returns
        -------
        _ConsoleLive
            Handle to update or stop the live display.
        """
        try:
            from rich.live import Live
        except ImportError:
            return _ConsoleLive(None)
        live = Live(console=self.console, refresh_per_second=8, transient=False)
        live.start()
        return _ConsoleLive(live)

    def stop_live(self, handle: _ConsoleLive, *, keep: bool = True) -> None:
        """Stop a live display."""
        handle.stop(keep=keep)

    async def prompt_input(self, label: str) -> str:
        """Prompt user via ``input()``."""
        return input(label)


class _InkSpinnerHandle:
    """Spinner handle that updates pyink component state."""

    def __init__(self, set_spinner_text: Callable, set_phase: Callable) -> None:
        self._set_spinner_text = set_spinner_text
        self._set_phase = set_phase

    def update_text(self, text: str) -> None:
        """Change the spinner label text."""
        self._set_spinner_text(text)

    def stop(self) -> None:
        """Clear the spinner and reset phase to idle."""
        self._set_phase("idle")
        self._set_spinner_text("")


class _InkLiveHandle:
    """Live-updating region backed by pyink state.

    Renders Rich content to ANSI and pushes it into the stream
    buffer. Throttled to ``_MIN_INTERVAL`` to avoid excessive
    re-renders during fast token streaming.
    """

    _MIN_INTERVAL = 1.0 / 12

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
        """Replace the stream content with a re-rendered *renderable*.

        Throttled to ~12 FPS so pyink can keep up with fast tokens.
        """
        import time

        self._last_renderable = renderable
        now = time.monotonic()
        if now - self._last_update < self._MIN_INTERVAL:
            return
        self._last_update = now
        ansi = rich_to_ansi(self._console, renderable)
        self._set_stream(ansi)

    def stop(self, *, keep: bool = True) -> None:
        """End the live region.

        Parameters
        ----------
        keep : bool
            If True, render the final content through Rich and push
            it to history as a ``"response"`` item with a bullet.
        """
        if keep and self._last_renderable is not None:
            ansi = rich_to_ansi(self._console, self._last_renderable)
            if ansi:
                self._set_history(lambda h: [*h, {"type": "response", "ansi": ansi}])
        self._set_phase("idle")
        self._set_stream("")


class InkWriter:
    """Writer that pushes events into pyink component state.

    Each method translates output operations into ``set_state``
    calls that trigger pyink re-renders.

    Parameters
    ----------
    set_history : callable
        State setter for the ``Static`` history list.
    set_spinner_text : callable
        State setter for the spinner label.
    set_stream : callable
        State setter for the streaming buffer.
    set_phase : callable
        State setter for the phase (``"idle"``/``"spinning"``/``"streaming"``).
    console : Console
        Rich console for rendering content to ANSI.
    approval_callback : callable, optional
        Blocking callback for approval/human-input prompts.
    """

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

    def print_rich(self, *args: Any, item_type: str | None = None, **kwargs: Any) -> None:
        """Render Rich content and append to history.

        Tool response lines (containing ``U+2514``) are tagged as
        ``"tool_done"`` so the UI can distinguish them.
        """
        ansi = rich_to_ansi(self._console, *args, **kwargs)
        if ansi:
            if item_type is None:
                item_type = "tool_done" if "\u2514" in ansi else "rich"
            self._set_history(lambda h: [*h, {"type": item_type, "ansi": ansi}])

    def start_spinner(self, text: str) -> _InkSpinnerHandle:
        """Start a spinner by setting phase and text.

        Parameters
        ----------
        text : str
            Label shown next to the spinner animation.

        Returns
        -------
        _InkSpinnerHandle
            Handle to update or stop the spinner.
        """
        self._set_phase("spinning")
        self._set_spinner_text(text)
        return _InkSpinnerHandle(self._set_spinner_text, self._set_phase)

    def stop_spinner(self, handle: _InkSpinnerHandle) -> None:
        """Stop the spinner."""
        handle.stop()

    def start_live(self) -> _InkLiveHandle:
        """Start a live streaming region.

        Returns
        -------
        _InkLiveHandle
            Handle to update or stop the live region.
        """
        self._set_phase("streaming")
        self._set_stream("")
        return _InkLiveHandle(
            self._set_stream,
            self._set_phase,
            self._set_history,
            self._console,
        )

    def stop_live(self, handle: _InkLiveHandle, *, keep: bool = True) -> None:
        """Stop the live region, optionally freezing content to history."""
        handle.stop(keep=keep)

    async def prompt_input(self, label: str) -> str:
        """Prompt user for input via the selector UI.

        Runs the blocking selector callback in a thread executor so
        it doesn't freeze the agent's asyncio event loop.

        Parameters
        ----------
        label : str
            Prompt text, possibly containing numbered options.

        Returns
        -------
        str
            The user's response.
        """
        import asyncio

        if self._approval_callback:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._approval_callback, label,
            )
        return input(label)
