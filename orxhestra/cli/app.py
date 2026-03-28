"""Orx CLI - terminal coding agent powered by any LLM.

Usage::

    orx                            # interactive REPL with coding agent
    orx --model claude-sonnet-4-6  # use a specific model
    orx compose.yaml               # run a compose file interactively
    orx -c "fix the tests"         # single-shot command
    orx --coding                   # multi-agent pipeline
    orx --auto-approve             # skip approval prompts
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from orxhestra.cli.config import APP_NAME, DEFAULT_MODEL, DEFAULT_USER_ID, HISTORY_DIR, HISTORY_FILE


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="orx",
        description="Orx - terminal coding agent powered by any LLM.",
    )
    parser.add_argument(
        "compose_file",
        nargs="?",
        default=None,
        help="Optional compose YAML file to run instead of the built-in coding agent.",
    )
    parser.add_argument(
        "-m", "--model",
        default=os.environ.get("ORX_MODEL", DEFAULT_MODEL),
        help=f"Model name (default: {DEFAULT_MODEL}). Also reads $ORX_MODEL.",
    )
    parser.add_argument(
        "-w", "--workspace",
        default=os.getcwd(),
        help="Workspace directory (default: current directory).",
    )
    parser.add_argument(
        "-c", "--command",
        default=None,
        help="Single-shot command. Runs the prompt and exits.",
    )
    parser.add_argument(
        "--coding",
        action="store_true",
        help="Use the built-in multi-agent coding pipeline (plan -> code -> review loop).",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip approval prompts for destructive tools.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------


async def _build_coding_agent(
    model_name: str,
    workspace: str,
) -> tuple[Any, str, Any, Any]:
    """Build an LlmAgent with all CLI features.

    Returns (runner, session_id, todo_list, llm).
    """
    from orxhestra.agents.llm_agent import LlmAgent
    from orxhestra.cli.context_injection import collect_local_context
    from orxhestra.cli.memory import load_agents_md
    from orxhestra.cli.models import create_llm
    from orxhestra.cli.prompt import get_system_prompt
    from orxhestra.cli.task_tool import make_task_tool
    from orxhestra.cli.todo_tool import TodoList, make_todo_tool
    from orxhestra.runner import Runner
    from orxhestra.sessions.in_memory_session_service import InMemorySessionService
    from orxhestra.tools.filesystem import make_filesystem_tools
    from orxhestra.tools.shell import make_shell_tools

    llm = create_llm(model_name)

    # Core tools
    fs_tools = make_filesystem_tools(workspace=workspace)
    sh_tools = make_shell_tools(workspace=workspace, timeout=120, max_output_bytes=200_000)

    # Todo tracking
    todo_list = TodoList()
    todo_tool = make_todo_tool(todo_list)

    # Sub-agent delegation
    task_tool = make_task_tool(llm, [*fs_tools, *sh_tools], workspace)

    all_tools = [*fs_tools, *sh_tools, todo_tool, task_tool]

    # Collect context in parallel
    memory_content: str = load_agents_md(workspace)
    local_context: str = await collect_local_context(workspace)

    instructions: str = get_system_prompt(
        cwd=workspace,
        memory_content=memory_content,
        local_context=local_context,
    )

    agent = LlmAgent(
        name="orx-coder",
        llm=llm,
        tools=all_tools,
        instructions=instructions,
        description="Terminal coding agent with filesystem, shell, planning, and delegation.",
        max_iterations=30,
    )

    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session_id: str = str(uuid4())
    return runner, session_id, todo_list, llm


async def _build_compose_runner(compose_path: str) -> tuple[Any, str, None, None]:
    """Build a Runner from a compose YAML file."""
    from orxhestra.composer import Composer

    path = Path(compose_path)
    if not path.exists():
        print(f"Error: compose file not found: {path}")
        sys.exit(1)

    runner = await Composer.runner_from_yaml_async(path)
    session_id: str = str(uuid4())
    return runner, session_id, None, None


async def _build_coding_pipeline(
    model_name: str,
    workspace: str,
) -> tuple[Any, str, None, None]:
    """Build the multi-agent coding pipeline from the built-in compose."""
    from orxhestra.composer import Composer

    compose_path = Path(__file__).parent.parent.parent / "skills" / "coding" / "compose.yaml"
    if not compose_path.exists():
        print(f"Error: built-in coding compose not found at {compose_path}")
        sys.exit(1)

    os.environ.setdefault("AGENT_WORKSPACE", workspace)
    os.environ.setdefault("ORX_MODEL", model_name)

    runner = await Composer.runner_from_yaml_async(compose_path)
    session_id: str = str(uuid4())
    return runner, session_id, None, None


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------


async def _prompt_approval(
    tool_name: str,
    args: dict[str, Any],
    console: Any,
    auto_approve: bool,
) -> bool:
    """Ask the user to approve a destructive tool call. Returns True if approved."""
    if auto_approve:
        return True

    from orxhestra.cli.approval import APPROVE_REQUIRED, _format_approval_prompt

    if tool_name not in APPROVE_REQUIRED:
        return True

    console.print(_format_approval_prompt(tool_name, args))

    try:
        response: str = input("  approve? [y/n/a(ll)]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if response in ("a", "all"):
        return True  # caller should set auto_approve = True
    return response in ("y", "yes")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_tool_call(event: Any, console: Any) -> None:
    """Print a tool call event."""
    for tc in event.tool_calls:
        args_summary: str = ""
        args: dict = tc.args or {}
        if "path" in args:
            args_summary = args["path"]
        elif "command" in args:
            cmd: str = args["command"]
            args_summary = cmd[:80] + ("..." if len(cmd) > 80 else "")
        elif "pattern" in args:
            args_summary = args["pattern"]
        elif "description" in args:
            desc: str = args["description"]
            args_summary = desc[:60] + ("..." if len(desc) > 60 else "")
        elif "todos" in args:
            args_summary = "(task list update)"
        else:
            args_summary = ", ".join(
                f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:3]
            )
        console.print(f"  [dim]> {tc.tool_name}({args_summary})[/dim]")


def _render_tool_response(event: Any, console: Any) -> None:
    """Print a truncated tool response."""
    text: str = (event.text or "")[:300]
    if text:
        lines: list[str] = text.splitlines()
        preview: str = "\n    ".join(lines[:5])
        if len(lines) > 5:
            preview += f"\n    ... ({len(lines)} lines total)"
        console.print(f"    [dim]{preview}[/dim]")


def _render_todos(todo_list: Any, console: Any) -> None:
    """Render the todo list if it has items."""
    if todo_list is None or not todo_list.todos:
        return
    rendered: str = todo_list.render()
    if rendered:
        console.print(f"\n[bold]Tasks:[/bold]\n{rendered}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


async def _stream_response(
    runner: Any,
    session_id: str,
    message: str,
    console: Any,
    Markdown: type,
    *,
    todo_list: Any = None,
    auto_approve: bool = False,
) -> bool:
    """Stream a single agent turn, rendering events in real time.

    Returns updated auto_approve value (may change if user selects 'all').
    """
    from orxhestra.cli.approval import APPROVE_REQUIRED
    from orxhestra.events.event import EventType

    buffer: str = ""
    in_stream: bool = False
    live: Any = None

    try:
        from rich.live import Live
    except ImportError:
        Live = None

    try:
        async for event in runner.astream(
            user_id=DEFAULT_USER_ID,
            session_id=session_id,
            new_message=message,
        ):
            # Streaming partial tokens
            if event.partial and event.type == EventType.AGENT_MESSAGE and event.text:
                buffer += event.text
                if Live is not None:
                    if not in_stream:
                        in_stream = True
                        live = Live(
                            Markdown(buffer),
                            console=console,
                            refresh_per_second=12,
                            vertical_overflow="visible",
                        )
                        live.start()
                    else:
                        live.update(Markdown(buffer))
                else:
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
                continue

            # Tool call — with approval for destructive tools
            if event.has_tool_calls:
                if in_stream and live:
                    live.stop()
                    in_stream = False
                    if buffer:
                        console.print(Markdown(buffer))
                        buffer = ""

                _render_tool_call(event, console)

                # Show approval prompt for destructive tools
                for tc in event.tool_calls:
                    if tc.tool_name in APPROVE_REQUIRED and not auto_approve:
                        approved: bool = await _prompt_approval(
                            tc.tool_name, tc.args or {}, console, auto_approve
                        )
                        if not approved:
                            console.print("  [dim]Denied.[/dim]")
                            # Note: we can't actually cancel mid-stream in the
                            # current SDK - the tool already ran. This approval
                            # is informational. Future: integrate with
                            # before_tool_callback to block execution.
                continue

            # Tool response
            if event.type == EventType.TOOL_RESPONSE:
                _render_tool_response(event, console)

                # Show updated todo list after write_todos
                if event.tool_name == "write_todos" and todo_list is not None:
                    _render_todos(todo_list, console)
                continue

            # Final response
            if event.is_final_response():
                if in_stream and live:
                    live.stop()
                    in_stream = False
                    buffer = ""
                if event.text:
                    agent_label: str = (
                        f"[{event.agent_name}] "
                        if event.agent_name and event.agent_name != "orx-coder"
                        else ""
                    )
                    if agent_label:
                        console.print(f"\n[bold cyan]{agent_label}[/bold cyan]")
                    console.print(Markdown(event.text))
                continue

            # Error events
            if event.metadata.get("error") and event.text:
                if in_stream and live:
                    live.stop()
                    in_stream = False
                    buffer = ""
                console.print(f"[bold red]Error:[/bold red] {event.text}")
                continue

    except KeyboardInterrupt:
        if in_stream and live:
            live.stop()
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        if in_stream and live:
            live.stop()
            if buffer:
                console.print(Markdown(buffer))

    return auto_approve


async def _repl(
    runner: Any,
    session_id: str,
    model_name: str,
    workspace: str,
    *,
    todo_list: Any = None,
    llm: Any = None,
    auto_approve: bool = False,
) -> None:
    """Run the interactive REPL."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
    except ImportError:
        print("Error: rich is required. Install with: pip install orxhestra[cli]")
        sys.exit(1)

    console = Console()

    # Welcome banner
    console.print("\n[bold blue]  orx[/bold blue] [dim]- terminal coding agent[/dim]")
    console.print(f"  [dim]model: {model_name} | workspace: {workspace}[/dim]")
    console.print(f"  [dim]type /help for commands, Ctrl+D to exit[/dim]\n")

    # Try prompt_toolkit for history support, fall back to plain input
    prompt_session: Any = None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        prompt_session = PromptSession(history=FileHistory(str(HISTORY_FILE)))
    except ImportError:
        pass

    turn_count: int = 0

    while True:
        try:
            if prompt_session:
                user_input: str = await prompt_session.prompt_async("orx> ")
            else:
                user_input = input("orx> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd_parts: list[str] = user_input.split(maxsplit=1)
            cmd: str = cmd_parts[0].lower()

            if cmd in ("/exit", "/quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/clear":
                session_id = str(uuid4())
                if todo_list is not None:
                    todo_list.todos = []
                turn_count = 0
                console.print("[dim]Session cleared.[/dim]")
                continue

            elif cmd == "/compact":
                if llm is not None:
                    console.print("[dim]Compacting conversation...[/dim]")
                    from orxhestra.cli.summarization import summarize_session

                    session = await runner.get_or_create_session(
                        user_id=DEFAULT_USER_ID, session_id=session_id
                    )
                    result = await summarize_session(llm, session.events)
                    if result is not None:
                        session.events[:] = result
                        console.print("[dim]Conversation compacted.[/dim]")
                    else:
                        console.print("[dim]Nothing to compact.[/dim]")
                else:
                    console.print("[dim]Compact not available in compose mode.[/dim]")
                continue

            elif cmd == "/model":
                if len(cmd_parts) > 1:
                    new_model: str = cmd_parts[1].strip()
                    try:
                        runner, session_id, todo_list, llm = await _build_coding_agent(
                            new_model, workspace
                        )
                        model_name = new_model
                        turn_count = 0
                        console.print(f"[dim]Switched to {model_name}[/dim]")
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                else:
                    console.print(f"[dim]Current model: {model_name}[/dim]")
                continue

            elif cmd == "/todos":
                if todo_list is not None:
                    _render_todos(todo_list, console)
                    if not todo_list.todos:
                        console.print("[dim]No tasks.[/dim]")
                else:
                    console.print("[dim]Todos not available in compose mode.[/dim]")
                continue

            elif cmd == "/help":
                console.print("[bold]Commands:[/bold]")
                console.print("  /model <name>  Switch model")
                console.print("  /clear         Clear session")
                console.print("  /compact       Summarize old messages to free context")
                console.print("  /todos         Show current task list")
                console.print("  /exit          Exit")
                console.print("  /help          Show this help")
                continue

            else:
                console.print(f"[dim]Unknown command: {cmd}. Type /help[/dim]")
                continue

        # Auto-summarize if conversation is getting long
        if llm is not None and turn_count > 0 and turn_count % 20 == 0:
            from orxhestra.cli.summarization import summarize_session

            try:
                session = await runner.get_or_create_session(
                    user_id=DEFAULT_USER_ID, session_id=session_id
                )
                result = await summarize_session(llm, session.events)
                if result is not None:
                    session.events[:] = result
                    console.print("[dim](conversation auto-compacted)[/dim]")
            except Exception:
                pass

        auto_approve = await _stream_response(
            runner,
            session_id,
            user_input,
            console,
            Markdown,
            todo_list=todo_list,
            auto_approve=auto_approve,
        )
        turn_count += 1
        console.print()  # blank line between turns


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _async_main() -> None:
    """Async entry point."""
    args = _parse_args()

    if args.compose_file:
        runner, session_id, todo_list, llm = await _build_compose_runner(args.compose_file)
    elif args.coding:
        runner, session_id, todo_list, llm = await _build_coding_pipeline(
            args.model, args.workspace
        )
    else:
        runner, session_id, todo_list, llm = await _build_coding_agent(
            args.model, args.workspace
        )

    auto_approve: bool = args.auto_approve

    # Single-shot mode
    if args.command:
        try:
            from rich.console import Console
            from rich.markdown import Markdown
        except ImportError:
            print("Error: rich is required. Install with: pip install orxhestra[cli]")
            sys.exit(1)

        console = Console()
        await _stream_response(
            runner,
            session_id,
            args.command,
            console,
            Markdown,
            todo_list=todo_list,
            auto_approve=auto_approve,
        )
        return

    # Interactive REPL
    await _repl(
        runner,
        session_id,
        args.model,
        args.workspace,
        todo_list=todo_list,
        llm=llm,
        auto_approve=auto_approve,
    )


def main() -> None:
    """Entry point for the ``orx`` command."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
