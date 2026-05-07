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
        Replace the live region content with a new Rich renderable.
    update_text : callable
        Submit the cumulative buffer text + markdown class. Preferred
        for streaming so the implementation can window/throttle/spill
        without the caller knowing.
    stop : callable
        End the live region, optionally freezing content to history.
    """

    def update(self, renderable: Any) -> None: ...
    def update_text(self, text: str, markdown_cls: type) -> None: ...
    def stop(self, *, keep: bool = True) -> None: ...


@runtime_checkable
class Writer(Protocol):
    """Unified output interface for the CLI.

    All rendering code (``stream.py``, ``render.py``, ``commands.py``)
    writes through this protocol so the same logic works for both the
    pyink TUI and the Rich console fallback.

    See Also
    --------
    ConsoleWriter : Rich-based implementation for ``-c`` single-shot mode.
    InkWriter : pyink-based implementation for the interactive REPL.
    SpinnerHandle : Returned by :meth:`start_spinner`.
    LiveHandle : Returned by :meth:`start_live`.
    """

    def print_rich(self, *args: Any, **kwargs: Any) -> None:
        """Write Rich-formatted content to the output stream.

        Parameters
        ----------
        *args : Any
            Rich renderables or strings forwarded to the underlying
            ``Console.print``.
        **kwargs : Any
            Keyword arguments forwarded to ``Console.print``
            (e.g. ``style``, ``end``). The pyink implementation also
            accepts ``item_type`` to tag the history entry.
        """
        ...

    def start_spinner(self, text: str) -> SpinnerHandle:
        """Start a spinner with the given label.

        Parameters
        ----------
        text : str
            Text shown next to the animated spinner.

        Returns
        -------
        SpinnerHandle
            Handle used to update the label or stop the spinner.
        """
        ...

    def start_live(self) -> LiveHandle:
        """Start a live-updating region for streaming output.

        Returns
        -------
        LiveHandle
            Handle used to replace the region's content or end it.
        """
        ...

    async def prompt_input(self, label: str) -> str:
        """Prompt the user for a line of input.

        Parameters
        ----------
        label : str
            Prompt shown to the user. May contain numbered options.

        Returns
        -------
        str
            The line entered by the user.
        """
        ...


# Bounds for the live streaming region. Keep per-frame layout work
# constant regardless of total response length. See _StreamRenderer.
_STREAM_TAIL_LINES = 80
_SPILL_THRESHOLD = 400
_MAX_LINE_CHARS = 2000
_RENDER_INTERVAL = 1.0 / 12


def _soft_wrap_long_lines(text: str, max_chars: int = _MAX_LINE_CHARS) -> str:
    """Break pathologically long single lines (e.g. minified JSON).

    Yoga layout cost grows with the longest line; uncapped lines can
    starve the pyink render loop. Splits at the last whitespace inside
    the chunk when possible, falls back to a hard cut otherwise.
    """
    out: list[str] = []
    for line in text.split("\n"):
        if len(line) <= max_chars:
            out.append(line)
            continue
        i = 0
        while i < len(line):
            end = i + max_chars
            if end >= len(line):
                out.append(line[i:])
                break
            chunk = line[i:end]
            ws = chunk.rfind(" ")
            if ws > max_chars // 2:
                out.append(chunk[:ws])
                i += ws + 1
            else:
                out.append(chunk)
                i = end
    return "\n".join(out)


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

    def update_text(self, text: str, markdown_cls: type) -> None:
        """Replace the live content with ``markdown_cls(text)``."""
        if self._live:
            self._live.update(markdown_cls(text))

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

    See Also
    --------
    Writer : Protocol this class implements.
    InkWriter : pyink-based alternative for the interactive REPL.
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


class _StreamRenderer:
    """Background thread converting markdown to ANSI off the worker.

    Keeps the agent worker thread free of Rich rendering cost. Holds a
    single-slot mailbox: the producer overwrites the latest
    ``(text, markdown_cls)`` and notifies; the scheduler wakes up to
    every ``_RENDER_INTERVAL`` seconds, takes the slot, converts, and
    pushes to pyink state.

    Spill-to-history: when accumulated text exceeds ``_SPILL_THRESHOLD``
    lines, older content is frozen as immutable history items and
    dropped from the live tail. This bounds per-frame layout work so the
    pyink input handler (including Ctrl+C) is never starved.

    Lock discipline: the mailbox lock is only held for O(1) slot
    read/write — never around Rich conversion or state setters — so
    the producer never blocks behind the renderer.
    """

    def __init__(
        self,
        set_stream: Callable,
        set_history: Callable,
        console: Console,
    ) -> None:
        import threading

        self._set_stream = set_stream
        self._set_history = set_history
        self._console = console
        self._cond = threading.Condition()
        self._slot: tuple[str, type] | None = None
        self._stopped = False
        self._spilled_lines = 0
        self._last_text = ""
        self._last_markdown_cls: type | None = None
        self._bullet_emitted = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, text: str, markdown_cls: type) -> None:
        """Producer: store latest text + markdown class; wake scheduler."""
        with self._cond:
            self._slot = (text, markdown_cls)
            self._cond.notify()

    def _run(self) -> None:
        import time

        while True:
            with self._cond:
                while self._slot is None and not self._stopped:
                    self._cond.wait(timeout=_RENDER_INTERVAL)
                if self._stopped and self._slot is None:
                    return
                slot = self._slot
                self._slot = None
                stopped_now = self._stopped

            if slot is None:
                continue
            text, markdown_cls = slot
            self._last_text = text
            self._last_markdown_cls = markdown_cls

            # Always run spill (updates _spilled_lines, may push to
            # history) so stop()'s final flush stays bounded; but skip
            # set_stream when we're already stopping — the worker is
            # about to push the final tail to scrollback and clear the
            # dynamic region in one batch, and any late set_stream
            # would cause a flicker between the two frames.
            visible = self._maybe_spill(text)
            if not stopped_now:
                try:
                    wrapped = _soft_wrap_long_lines(visible)
                    ansi = rich_to_ansi(self._console, markdown_cls(wrapped))
                    self._set_stream(ansi)
                except Exception:
                    pass

            if stopped_now:
                return
            time.sleep(_RENDER_INTERVAL)

    def _maybe_spill(self, text: str) -> str:
        """Freeze older content to history once the buffer is too large."""
        all_lines = text.split("\n")
        unspilled = all_lines[self._spilled_lines:]
        if len(unspilled) <= _SPILL_THRESHOLD:
            return "\n".join(unspilled)

        keep = unspilled[-_STREAM_TAIL_LINES:]
        spill = unspilled[:-_STREAM_TAIL_LINES]
        spill_text = "\n".join(spill)
        if self._last_markdown_cls is not None and spill_text.strip():
            try:
                wrapped = _soft_wrap_long_lines(spill_text)
                ansi = rich_to_ansi(
                    self._console, self._last_markdown_cls(wrapped),
                )
                if ansi:
                    item_type = "response" if not self._bullet_emitted else "rich"
                    self._set_history(
                        lambda h, a=ansi, t=item_type: [
                            *h, {"type": t, "ansi": a},
                        ],
                    )
                    self._bullet_emitted = True
            except Exception:
                pass
        self._spilled_lines += len(spill)
        return "\n".join(keep)

    def stop(self, *, keep: bool = True) -> tuple[str | None, str]:
        """Stop the scheduler and return the final tail ANSI + item type."""
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=0.5)

        item_type = "response" if not self._bullet_emitted else "rich"
        if not keep or self._last_markdown_cls is None:
            return None, item_type
        all_lines = self._last_text.split("\n")
        tail_text = "\n".join(all_lines[self._spilled_lines:])
        if not tail_text.strip():
            return None, item_type
        try:
            wrapped = _soft_wrap_long_lines(tail_text)
            return (
                rich_to_ansi(self._console, self._last_markdown_cls(wrapped)),
                item_type,
            )
        except Exception:
            return None, item_type


class _InkLiveHandle:
    """Live-updating region backed by pyink state.

    Delegates Rich→ANSI conversion to a background ``_StreamRenderer``
    thread so the agent worker is never blocked by rendering, and
    bounds per-frame layout work via spill-to-history.
    """

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
        self._renderer = _StreamRenderer(set_stream, set_history, console)
        self._stopped = False

    def update(self, renderable: Any) -> None:
        """Convert and push immediately. Used by non-streaming callers.

        Streaming callers should prefer :meth:`update_text`.
        """
        if self._stopped:
            return
        try:
            ansi = rich_to_ansi(self._console, renderable)
            self._set_stream(ansi)
        except Exception:
            pass

    def update_text(self, text: str, markdown_cls: type) -> None:
        """Submit raw text + markdown class to the background renderer."""
        if self._stopped:
            return
        self._renderer.submit(text, markdown_cls)

    def stop(self, *, keep: bool = True) -> None:
        """End the live region. Idempotent.

        Parameters
        ----------
        keep : bool
            If True, push the final un-spilled tail into history as a
            ``"response"`` (with bullet) or ``"rich"`` item depending on
            whether earlier spill chunks already received the bullet.

        Order matters here: the history append must precede the stream
        clear so pyink's batched render shows the response transitioning
        from the dynamic area into scrollback in a single frame, with
        no blank-frame flicker.
        """
        if self._stopped:
            return
        self._stopped = True
        final_ansi, item_type = self._renderer.stop(keep=keep)
        if keep and final_ansi:
            self._set_history(
                lambda h, a=final_ansi, t=item_type: [
                    *h, {"type": t, "ansi": a},
                ],
            )
        self._set_stream("")
        self._set_phase("idle")


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

    See Also
    --------
    Writer : Protocol this class implements.
    ConsoleWriter : Rich-based alternative for ``-c`` mode.
    rich_to_ansi : Helper used to serialize Rich content for pyink.
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
        self._active_live: _InkLiveHandle | None = None

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
        handle = _InkLiveHandle(
            self._set_stream,
            self._set_phase,
            self._set_history,
            self._console,
        )
        self._active_live = handle
        return handle

    def stop_live(self, handle: _InkLiveHandle, *, keep: bool = True) -> None:
        """Stop the live region, optionally freezing content to history."""
        handle.stop(keep=keep)
        if self._active_live is handle:
            self._active_live = None

    def cancel_live(self) -> None:
        """Pre-empt any active live region without freezing it to history.

        Called from the input handler on Ctrl+C so the renderer thread
        stops pushing tokens immediately and the live tail clears.
        """
        handle = self._active_live
        if handle is not None:
            handle.stop(keep=False)
            self._active_live = None

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
