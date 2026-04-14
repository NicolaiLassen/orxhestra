"""Streaming response handler for the orx CLI."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orxhestra.cli.approval import APPROVE_REQUIRED
from orxhestra.cli.config import DEFAULT_USER_ID
from orxhestra.cli.render import (
    render_todos,
    render_tool_call,
    render_tool_response,
    render_turn_summary,
)
from orxhestra.events.event import EventType

if TYPE_CHECKING:
    from orxhestra.cli.writer import Writer
    from orxhestra.events.event import Event
    from orxhestra.runner import Runner
    from orxhestra.tools.todo_tool import TodoList


def _spinner_text(todo_list: TodoList | None) -> str:
    """Pick spinner text: active task name or random phrase."""
    if todo_list is not None:
        active: str | None = todo_list.get_active_task()
        if active:
            return active[:60] + "..." if len(active) > 60 else active
    return random.choice(_THINKING_PHRASES)


_THINKING_PHRASES: list[str] = [
    "Thinking",
    "Pondering",
    "Considering",
    "Reflecting",
    "Reasoning",
    "Analyzing",
    "Processing",
    "Evaluating",
    "Examining",
    "Synthesizing",
    "Formulating",
    "Deliberating",
    "Contemplating",
    "Interpreting",
    "Computing",
    "Deciphering",
    "Untangling",
    "Calibrating",
    "Orchestrating",
    "Deconstructing",
    "Strategizing",
    "Distilling",
    "Crystallizing",
    "Navigating",
    "Investigating",
    "Composing",
    "Reticulating",
    "Weaving",
    "Converging",
]


# Interval (seconds) between rotating the thinking phrase text.
_PHRASE_ROTATE_INTERVAL: float = 4.0


@dataclass
class _StreamState:
    """Mutable state tracked during a single streaming turn."""

    buffer: str = ""
    in_stream: bool = False
    thinking_active: bool = False
    live_handle: Any = None
    spinner: Any = None
    tool_start: float = 0.0
    turn_start: float = field(default_factory=time.monotonic)
    _phrase_task: Any = field(default=None, repr=False)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    interactive_tool_ids: set[str] = field(default_factory=set)
    confirmation_tool_ids: set[str] = field(default_factory=set)
    pending_tool_ids: set[str] = field(default_factory=set)

    def stop_spinner(self, writer: Writer) -> None:
        """Stop and clear the spinner and phrase rotation."""
        if self._phrase_task is not None:
            self._phrase_task.cancel()
            self._phrase_task = None
        if self.spinner is not None:
            writer.stop_spinner(self.spinner)
            self.spinner = None

    def end_stream(self, writer: Writer, markdown_cls: type) -> None:
        """Stop live preview and clear buffer."""
        if self.in_stream:
            if self.live_handle is not None:
                if self.buffer:
                    self.live_handle.update(markdown_cls(self.buffer))
                writer.stop_live(self.live_handle, keep=True)
                self.live_handle = None
            elif self.buffer:
                writer.print_rich(markdown_cls(self.buffer))
            self.in_stream = False
            self.buffer = ""

    def accumulate_usage(self, event: Event) -> None:
        """Extract token usage from the event's LLM response."""
        if event.llm_response:
            self.prompt_tokens += event.llm_response.input_tokens or 0
            self.completion_tokens += event.llm_response.output_tokens or 0


async def prompt_approval(
    tool_name: str,
    args: dict[str, Any],
    writer: Writer,
    auto_approve: bool,
) -> bool:
    """Ask the user to approve a destructive tool call.

    Parameters
    ----------
    tool_name : str
        Name of the tool requiring approval.
    args : dict[str, Any]
        Arguments passed to the tool call.
    writer : Writer
        Output writer.
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

    # Build a compact label for the approval selector.
    arg_summary = ""
    if "command" in args:
        arg_summary = f"\n  {args['command']}"
    elif "path" in args:
        arg_summary = f"\n  {args['path']}"
    label = f"  {tool_name}{arg_summary}"

    try:
        response: str = (await writer.prompt_input(label)).strip().lower()
    except (EOFError, KeyboardInterrupt, asyncio.CancelledError):
        return False

    if response in ("a", "all"):
        return True
    return response in ("y", "yes")


async def stream_response(
    runner: Runner,
    session_id: str,
    message: str,
    writer: Writer,
    markdown_cls: type,
    *,
    todo_list: TodoList | None = None,
    auto_approve: bool = False,
) -> bool:
    """Stream a single agent turn, rendering events in real time.

    Parameters
    ----------
    runner : Runner
        The agent runner to stream from.
    session_id : str
        Session identifier for the conversation.
    message : str
        User message to send to the agent.
    writer : Writer
        Output writer (TuiWriter or ConsoleWriter).
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

    async def _start_spinner(state: _StreamState) -> None:
        """Start the spinner with rotating phrase text."""
        phrase = _spinner_text(todo_list)
        state.spinner = writer.start_spinner(f"{phrase}...")

        async def _rotate() -> None:
            try:
                while True:
                    await asyncio.sleep(_PHRASE_ROTATE_INTERVAL)
                    if state.spinner is None:
                        break
                    new_phrase = _spinner_text(todo_list)
                    state.spinner.update_text(f"{new_phrase}...")
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
                s.stop_spinner(writer)
                if not s.thinking_active:
                    s.thinking_active = True
                    writer.print_rich(
                        "[orx.thinking]  thinking ...[/orx.thinking]",
                    )
                writer.print_rich(
                    f"[orx.thinking]{event.thinking}[/orx.thinking]",
                    end="",
                )
                continue

            if event.partial and event.type == EventType.AGENT_MESSAGE and event.text:
                s.stop_spinner(writer)
                if s.thinking_active:
                    writer.print_rich()  # newline after thinking
                    s.thinking_active = False
                s.buffer += event.text
                if not s.in_stream:
                    s.in_stream = True
                    s.live_handle = writer.start_live()
                    s.live_handle.update(markdown_cls(s.buffer))
                else:
                    s.live_handle.update(markdown_cls(s.buffer))
                continue

            if event.has_tool_calls:
                s.stop_spinner(writer)
                s.end_stream(writer, markdown_cls)

                has_interactive: bool = False
                has_confirmation: bool = False
                for tc in event.tool_calls:
                    s.pending_tool_ids.add(tc.tool_call_id)
                    if tc.metadata.get("interactive"):
                        s.interactive_tool_ids.add(tc.tool_call_id)
                        has_interactive = True
                    if tc.metadata.get("require_confirmation"):
                        s.confirmation_tool_ids.add(tc.tool_call_id)
                        has_confirmation = True

                render_tool_call(event, writer)

                for tc in event.tool_calls:
                    if tc.tool_name in APPROVE_REQUIRED and not auto_approve:
                        approved: bool = await prompt_approval(
                            tc.tool_name, tc.args or {}, writer, auto_approve
                        )
                        if not approved:
                            writer.print_rich("  [orx.denied]Denied.[/orx.denied]")

                s.tool_start = time.monotonic()
                if not has_interactive and not has_confirmation:
                    n_tools = len(event.tool_calls)
                    if n_tools > 1:
                        tool_label = f"{n_tools} tools"
                    else:
                        tool_label = event.tool_calls[-1].tool_name
                    s.spinner = writer.start_spinner(f"Running {tool_label}...")
                    # No phrase rotation for tool running — fixed text.
                continue

            if event.type == EventType.TOOL_RESPONSE:
                s.stop_spinner(writer)
                tool_call_id: str = ""
                if event.content.tool_responses:
                    tool_call_id = event.content.tool_responses[0].tool_call_id
                s.pending_tool_ids.discard(tool_call_id)
                if tool_call_id in s.interactive_tool_ids:
                    s.interactive_tool_ids.discard(tool_call_id)
                    s.tool_start = 0.0
                    continue
                if tool_call_id in s.confirmation_tool_ids:
                    s.confirmation_tool_ids.discard(tool_call_id)

                is_last = len(s.pending_tool_ids) == 0
                elapsed: float | None = None
                if is_last and s.tool_start > 0:
                    elapsed = time.monotonic() - s.tool_start
                if is_last:
                    render_tool_response(event, writer, elapsed=elapsed)

                if event.tool_name == "write_todos" and todo_list is not None:
                    render_todos(todo_list, writer)

                if is_last and s.spinner is None:
                    await _start_spinner(s)
                continue

            if event.is_final_response():
                s.stop_spinner(writer)
                was_streaming: bool = s.in_stream
                s.end_stream(writer, markdown_cls)
                if not was_streaming and event.text:
                    agent_label: str = (
                        f"[{event.agent_name}] "
                        if event.agent_name and event.agent_name != "coder"
                        else ""
                    )
                    if agent_label:
                        writer.print_rich(f"\n[orx.agent]{agent_label}[/orx.agent]")
                    writer.print_rich(markdown_cls(event.text))
                continue

            if event.metadata.get("compacting"):
                s.spinner = writer.start_spinner("Compacting context...")
                continue

            if event.metadata.get("compaction"):
                s.stop_spinner(writer)
                writer.print_rich(
                    "  [orx.muted]context compacted[/orx.muted]"
                )
                continue

            if event.metadata.get("error") and event.text:
                s.stop_spinner(writer)
                s.end_stream(writer, markdown_cls)
                writer.print_rich(f"[orx.error]Error:[/orx.error] {event.text}")
                continue

    except KeyboardInterrupt:
        s.stop_spinner(writer)
        s.end_stream(writer, markdown_cls)
        writer.print_rich("\n[orx.interrupted]Interrupted.[/orx.interrupted]")
    finally:
        s.stop_spinner(writer)
        if s.in_stream:
            s.end_stream(writer, markdown_cls)

    turn_elapsed: float = time.monotonic() - s.turn_start
    render_turn_summary(
        turn_elapsed,
        writer,
        prompt_tokens=s.prompt_tokens,
        completion_tokens=s.completion_tokens,
    )

    return auto_approve
