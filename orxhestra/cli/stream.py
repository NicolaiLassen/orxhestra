"""Streaming response handler for the orx CLI."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orxhestra.cli.approval import APPROVE_REQUIRED, format_approval_prompt
from orxhestra.cli.config import DEFAULT_USER_ID
from orxhestra.cli.render import (
    render_todos,
    render_tool_call,
    render_tool_response,
    render_turn_summary,
)
from orxhestra.cli.theme import ACCENT, RESPONSE_CONNECTOR
from orxhestra.events.event import EventType

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.events.event import Event
    from orxhestra.runner import Runner
    from orxhestra.tools.todo_tool import TodoList


def _spinner_text(todo_list: TodoList | None) -> str:
    """Pick spinner text: active task name or random phrase."""
    if todo_list is not None:
        active: str | None = todo_list.get_active_task()
        if active:
            # Truncate long task names for the spinner.
            return active[:60] + "..." if len(active) > 60 else active
    return random.choice(_THINKING_PHRASES)


_THINKING_PHRASES: list[str] = [
    "Thinking",
    "Reasoning",
    "Pondering",
    "Analyzing",
    "Considering",
    "Processing",
    "Reflecting",
    "Evaluating",
    "Examining",
    "Contemplating",
    "Synthesizing",
    "Interpreting",
    "Formulating",
    "Deliberating",
    "Connecting dots",
    "Piecing together",
    "Working through it",
    "Mapping it out",
]


# Interval (seconds) between rotating the thinking phrase text.
_PHRASE_ROTATE_INTERVAL: float = 4.0


@dataclass
class _StreamState:
    """Mutable state tracked during a single streaming turn."""

    buffer: str = ""
    in_stream: bool = False
    thinking_active: bool = False
    live: Any = None
    status: Any = None
    tool_start: float = 0.0
    turn_start: float = field(default_factory=time.monotonic)
    _phrase_task: Any = field(default=None, repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    interactive_tool_ids: set[str] = field(default_factory=set)
    confirmation_tool_ids: set[str] = field(default_factory=set)

    def stop_status(self) -> None:
        """Stop and clear the spinner and phrase rotation."""
        if self._phrase_task is not None:
            self._phrase_task.cancel()
            self._phrase_task = None
        if self.status is not None:
            self.status.stop()
            self.status = None

    def end_stream(self, console: Console, markdown_cls: type) -> None:
        """Stop live preview, render final Markdown, clear buffer."""
        if self.in_stream:
            if self.live is not None:
                self.live.stop()
                self.live = None
            if self.buffer:
                console.print(
                    f"[orx.muted]{RESPONSE_CONNECTOR}[/orx.muted] ",
                    end="",
                )
                console.print(markdown_cls(self.buffer))
            self.in_stream = False
            self.buffer = ""

    def accumulate_usage(self, event: Event) -> None:
        """Extract token usage from event metadata."""
        usage: dict = event.metadata.get("usage", {})
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0


async def prompt_approval(
    tool_name: str,
    args: dict[str, Any],
    console: Console,
    auto_approve: bool,
) -> bool:
    """Ask the user to approve a destructive tool call.

    Parameters
    ----------
    tool_name : str
        Name of the tool requiring approval.
    args : dict[str, Any]
        Arguments passed to the tool call.
    console : Console
        Rich console for rendering the prompt.
    auto_approve : bool
        If True, skip the prompt and approve automatically.

    Returns
    -------
    bool
        True if the user approved (or auto-approved), False otherwise.
    """
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
    runner: Runner,
    session_id: str,
    message: str,
    console: Console,
    markdown_cls: type,
    *,
    todo_list: TodoList | None = None,
    auto_approve: bool = False,
) -> bool:
    """Stream a single agent turn, rendering events in real time.

    Uses Rich Live with ``transient=True``: the live display shows a
    Markdown preview during streaming, then disappears and is replaced
    by a final static Markdown render.

    Parameters
    ----------
    runner : Runner
        The agent runner to stream from.
    session_id : str
        Session identifier for the conversation.
    message : str
        User message to send to the agent.
    console : Console
        Rich console for rendering output.
    markdown_cls : type
        Rich Markdown class used to render streamed text.
    todo_list : TodoList or None
        Shared todo list for task tracking display.
    auto_approve : bool
        If True, skip approval prompts for destructive tools.

    Returns
    -------
    bool
        Updated auto_approve value.
    """
    s = _StreamState()

    try:
        from rich.live import Live
    except ImportError:
        Live = None

    try:
        from rich.spinner import SPINNERS
        from rich.status import Status
    except ImportError:
        SPINNERS = None
        Status = None

    # Register the custom orxhestra spinner.
    if SPINNERS is not None:
        from orxhestra.cli.theme import ORX_SPINNER

        SPINNERS["orx_music"] = ORX_SPINNER

    async def _start_spinner(state: _StreamState) -> None:
        """Start the spinner with rotating phrase text."""
        if Status is None:
            return
        phrase = _spinner_text(todo_list)
        state.status = Status(
            f"  [orx.accent]{phrase}...[/orx.accent]",
            console=console,
            spinner="orx_music",
            spinner_style=ACCENT,
        )
        state.status.start()

        async def _rotate() -> None:
            """Rotate the phrase text on an interval."""
            try:
                while True:
                    await asyncio.sleep(_PHRASE_ROTATE_INTERVAL)
                    if state.status is None:
                        break
                    new_phrase = _spinner_text(todo_list)
                    state.status.update(f"  [orx.accent]{new_phrase}...[/orx.accent]")
            except asyncio.CancelledError:
                pass

        state._phrase_task = asyncio.create_task(_rotate())

    await _start_spinner(s)

    try:
        async for event in runner.astream(
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
            new_message=message,
        ):
            s.accumulate_usage(event)

            if event.partial and event.type == EventType.AGENT_MESSAGE and event.thinking:
                s.stop_status()
                if not s.thinking_active:
                    s.thinking_active = True
                    console.print(
                        "[orx.thinking]  thinking ...[/orx.thinking]",
                    )
                console.print(
                    f"[orx.thinking]{event.thinking}[/orx.thinking]",
                    end="",
                )
                continue

            if event.partial and event.type == EventType.AGENT_MESSAGE and event.text:
                s.stop_status()
                if s.thinking_active:
                    console.print()  # newline after thinking
                    s.thinking_active = False
                s.buffer += event.text
                if Live is not None:
                    if not s.in_stream:
                        s.in_stream = True
                        s.live = Live(
                            markdown_cls(s.buffer),
                            console=console,
                            refresh_per_second=8,
                            transient=True,
                            vertical_overflow="visible",
                        )
                        s.live.start()
                    else:
                        s.live.update(markdown_cls(s.buffer))
                else:
                    if not s.in_stream:
                        s.in_stream = True
                    import sys

                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                continue

            if event.has_tool_calls:
                s.stop_status()
                s.end_stream(console, markdown_cls)

                has_interactive: bool = False
                has_confirmation: bool = False
                for tc in event.tool_calls:
                    if tc.metadata.get("interactive"):
                        s.interactive_tool_ids.add(tc.tool_call_id)
                        has_interactive = True
                    if tc.metadata.get("require_confirmation"):
                        s.confirmation_tool_ids.add(tc.tool_call_id)
                        has_confirmation = True

                render_tool_call(event, console)

                for tc in event.tool_calls:
                    if tc.tool_name in APPROVE_REQUIRED and not auto_approve:
                        approved: bool = await prompt_approval(
                            tc.tool_name, tc.args or {}, console, auto_approve
                        )
                        if not approved:
                            console.print("  [orx.denied]Denied.[/orx.denied]")

                s.tool_start = time.monotonic()
                if Status is not None and not has_interactive and not has_confirmation:
                    last_tool: str = event.tool_calls[-1].tool_name
                    s.status = Status(
                        f"  [orx.accent]Running {last_tool}...[/orx.accent]",
                        console=console,
                        spinner="orx_music",
                        spinner_style=ACCENT,
                    )
                    s.status.start()
                    # No phrase rotation for tool running — fixed text.
                continue

            if event.type == EventType.TOOL_RESPONSE:
                s.stop_status()
                tool_call_id: str = ""
                if event.content.tool_responses:
                    tool_call_id = event.content.tool_responses[0].tool_call_id
                if tool_call_id in s.interactive_tool_ids:
                    s.interactive_tool_ids.discard(tool_call_id)
                    s.tool_start = 0.0
                    continue
                # Confirmation tools render normally (unlike interactive).
                if tool_call_id in s.confirmation_tool_ids:
                    s.confirmation_tool_ids.discard(tool_call_id)
                elapsed: float | None = None
                if s.tool_start > 0:
                    elapsed = time.monotonic() - s.tool_start
                render_tool_response(event, console, elapsed=elapsed)

                if event.tool_name == "write_todos" and todo_list is not None:
                    render_todos(todo_list, console)

                # Restart spinner — shows active task name if available.
                if Status is not None and s.status is None:
                    await _start_spinner(s)
                continue

            if event.is_final_response():
                s.stop_status()
                was_streaming: bool = s.in_stream
                s.end_stream(console, markdown_cls)
                if not was_streaming and event.text:
                    agent_label: str = (
                        f"[{event.agent_name}] "
                        if event.agent_name and event.agent_name != "coder"
                        else ""
                    )
                    if agent_label:
                        console.print(f"\n[orx.agent]{agent_label}[/orx.agent]")
                    console.print(
                        f"[orx.muted]{RESPONSE_CONNECTOR}[/orx.muted] ",
                        end="",
                    )
                    console.print(markdown_cls(event.text))
                continue

            if event.metadata.get("error") and event.text:
                s.stop_status()
                s.end_stream(console, markdown_cls)
                console.print(f"[orx.error]Error:[/orx.error] {event.text}")
                continue

    except KeyboardInterrupt:
        s.stop_status()
        s.end_stream(console, markdown_cls)
        console.print("\n[orx.interrupted]Interrupted.[/orx.interrupted]")
    finally:
        s.stop_status()
        if s.in_stream:
            s.end_stream(console, markdown_cls)

    turn_elapsed: float = time.monotonic() - s.turn_start
    render_turn_summary(
        turn_elapsed,
        console,
        prompt_tokens=s.prompt_tokens,
        completion_tokens=s.completion_tokens,
    )

    return auto_approve
