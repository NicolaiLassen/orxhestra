"""Orx CLI - terminal agent powered by any LLM.

Usage::

    orx                            # uses ./orx.yaml if present, else built-in default
    orx --model claude-sonnet-4-6  # use a specific model
    orx my-agents.yaml             # run a specific orx file
    orx -c "fix the tests"         # single-shot command
    orx --auto-approve             # skip approval prompts
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from orxhestra.cli.config import APP_NAME, DEFAULT_MODEL, DEFAULT_USER_ID, HISTORY_DIR, HISTORY_FILE

# Built-in orx templates (shipped inside the package)
_CLI_DIR: Path = Path(__file__).parent
_DEFAULT_ORX: Path = _CLI_DIR / "orx.yaml"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="orx",
        description="Orx - terminal agent powered by any LLM.",
    )
    parser.add_argument(
        "orx_file",
        nargs="?",
        default=None,
        help="orx YAML file to run. Defaults to built-in coding agent.",
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
        "--auto-approve",
        action="store_true",
        help="Skip approval prompts for destructive tools.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start as an A2A server instead of interactive REPL.",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port for --serve mode (default: 8000). Also reads $PORT.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ORX_LOG_LEVEL", "WARNING"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: WARNING). Also reads $ORX_LOG_LEVEL.",
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Print version and exit.",
    )
    return parser.parse_args()




def _set_human_input_callbacks(agent: Any, callback: Any) -> None:
    """Walk the agent tree and set the callback on any human_input tools."""
    if hasattr(agent, "_tools"):
        for tool in agent._tools.values():
            if tool.name == "human_input" and hasattr(tool, "set_callback"):
                tool.set_callback(callback)
    for child in getattr(agent, "sub_agents", []):
        _set_human_input_callbacks(child, callback)


async def _build_from_orx(
    orx_path: Path,
    model_name: str,
    workspace: str,
) -> tuple[Any, str, Any, Any]:
    """Build a Runner from an orx YAML with CLI enhancements.

    Returns (runner, session_id, todo_list, llm).
    """
    import yaml

    from orxhestra.cli.builtins import get_todo_list, register_cli_builtins
    from orxhestra.cli.context_injection import collect_local_context
    from orxhestra.cli.memory import load_agents_md
    from orxhestra.cli.models import create_llm, detect_provider
    from orxhestra.composer.composer import Composer
    from orxhestra.composer.schema import ComposeSpec

    if not orx_path.exists():
        print(f"Error: orx file not found: {orx_path}")
        sys.exit(1)

    os.environ["AGENT_WORKSPACE"] = workspace
    llm = create_llm(model_name)
    register_cli_builtins(workspace, llm)

    with open(orx_path) as f:
        raw: dict = yaml.safe_load(f)

    # Override the default model
    if "defaults" not in raw:
        raw["defaults"] = {}
    raw["defaults"]["model"] = {
        "provider": detect_provider(model_name),
        "name": model_name,
    }

    # Inject memory + local context into LLM agent instructions
    memory_content: str = load_agents_md(workspace)
    local_context: str = await collect_local_context(workspace)
    _inject_context(raw, workspace, memory_content, local_context)

    # Ensure local modules (e.g. tools.py) next to orx.yaml are importable
    orx_dir: str = str(orx_path.parent.resolve())
    if orx_dir not in sys.path:
        sys.path.insert(0, orx_dir)

    # Build via Composer
    spec: ComposeSpec = ComposeSpec.model_validate(raw)
    composer = Composer(spec)
    root = await composer._build()

    # Set up human_input callback on any human_input tools in the agent tree
    async def _human_input_prompt(question: str) -> str:
        try:
            return input(f"\n  ? {question}\n  > ")
        except (EOFError, KeyboardInterrupt):
            return "(user declined to answer)"

    _set_human_input_callbacks(root, _human_input_prompt)

    if spec.runner is not None:
        runner = await composer._build_runner(root)
    else:
        from orxhestra.artifacts.in_memory_artifact_service import InMemoryArtifactService
        from orxhestra.runner import Runner
        from orxhestra.sessions.compaction import CompactionConfig
        from orxhestra.sessions.in_memory_session_service import InMemorySessionService

        runner = Runner(
            agent=root,
            app_name=APP_NAME,
            session_service=InMemorySessionService(),
            artifact_service=InMemoryArtifactService(),
            compaction_config=CompactionConfig(
                char_threshold=40_000,
                retention_chars=8_000,
                llm=llm,
            ),
        )

    session_id: str = str(uuid4())
    todo_list = get_todo_list()
    return runner, session_id, todo_list, llm


def _inject_context(
    raw: dict,
    workspace: str,
    memory_content: str,
    local_context: str,
) -> None:
    """Append workspace, memory, and local context to LLM agent instructions."""
    extra: str = f"\n# Workspace\nCurrent working directory: {workspace}\n"
    if memory_content:
        extra += f"\n{memory_content}\n"
    if local_context:
        extra += f"\n{local_context}\n"

    if not extra.strip():
        return

    agents: dict = raw.get("agents", {})
    for agent_def in agents.values():
        agent_type: str = agent_def.get("type", "llm")
        if agent_type in ("llm", "react") and "instructions" in agent_def:
            agent_def["instructions"] += extra



_HELP_TEXT: str = """[bold]Commands:[/bold]
  /model <name>  Switch model (preserves history)
  /clear         Clear session
  /compact       Summarize old messages to free context
  /todos         Show current task list
  /session       Show session info
  /undo          Remove last turn
  /retry         Re-run last message
  /copy          Copy last response to clipboard
  /exit          Exit
  /help          Show this help

[bold]Multi-line input:[/bold]
  Start with \\"\\"\\" or ''' and end with the same delimiter."""


async def _handle_slash_command(
    cmd: str,
    cmd_arg: str | None,
    *,
    runner: Any,
    session_id: str,
    orx_path: Path,
    model_name: str,
    workspace: str,
    todo_list: Any,
    llm: Any,
    turn_count: int,
    console: Any,
) -> tuple[Any, str, Any, Any, str, int, bool]:
    """Handle a slash command.

    Returns updated state tuple:
    (runner, session_id, todo_list, llm, model_name, turn_count, should_continue).
    """
    from orxhestra.cli.render import render_todos

    if cmd in ("/exit", "/quit"):
        console.print("[dim]Goodbye![/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, False

    if cmd == "/clear":
        session_id = str(uuid4())
        if todo_list is not None:
            todo_list.todos = []
        console.print("[dim]Session cleared.[/dim]")
        return runner, session_id, todo_list, llm, model_name, 0, True

    if cmd == "/compact":
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
            console.print("[dim]Compact not available.[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/model":
        if cmd_arg:
            try:
                # Save old session events before rebuilding
                old_session = await runner.get_or_create_session(
                    user_id=DEFAULT_USER_ID, session_id=session_id
                )
                old_events = list(old_session.events)

                runner, _new_sid, todo_list, llm = await _build_from_orx(
                    orx_path, cmd_arg, workspace
                )
                model_name = cmd_arg

                # Restore events into new runner's session (keep old session_id)
                new_session = await runner.get_or_create_session(
                    user_id=DEFAULT_USER_ID, session_id=session_id
                )
                new_session.events.extend(old_events)

                turn_count = 0
                console.print(f"[dim]Switched to {model_name} (history preserved)[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        else:
            console.print(f"[dim]Current model: {model_name}[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/todos":
        if todo_list is not None:
            render_todos(todo_list, console)
            if not todo_list.todos:
                console.print("[dim]No tasks.[/dim]")
        else:
            console.print("[dim]No tasks.[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/session":
        session = await runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=session_id
        )
        event_count: int = len(session.events)
        from orxhestra.sessions.compaction import _estimate_event_chars

        total_chars: int = sum(_estimate_event_chars(e) for e in session.events)
        console.print(f"  [dim]session:  {session_id}[/dim]")
        console.print(f"  [dim]events:   {event_count}[/dim]")
        console.print(f"  [dim]chars:    {total_chars:,} (~{total_chars // 4:,} tokens)[/dim]")
        console.print(f"  [dim]turns:    {turn_count}[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/undo":
        session = await runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=session_id
        )
        from orxhestra.events.event import EventType as _ET

        # Find the last user message and remove everything from it onwards
        last_user_idx: int = -1
        for i in range(len(session.events) - 1, -1, -1):
            if session.events[i].type == _ET.USER_MESSAGE:
                last_user_idx = i
                break
        if last_user_idx >= 0:
            removed: int = len(session.events) - last_user_idx
            session.events[:] = session.events[:last_user_idx]
            turn_count = max(0, turn_count - 1)
            console.print(f"[dim]Removed last turn ({removed} events).[/dim]")
        else:
            console.print("[dim]Nothing to undo.[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/retry":
        session = await runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=session_id
        )
        from orxhestra.events.event import EventType as _ET

        # Find the last user message, save it, remove the turn
        last_msg: str | None = None
        last_user_idx = -1
        for i in range(len(session.events) - 1, -1, -1):
            if session.events[i].type == _ET.USER_MESSAGE:
                last_msg = session.events[i].text
                last_user_idx = i
                break
        if last_msg and last_user_idx >= 0:
            session.events[:] = session.events[:last_user_idx]
            turn_count = max(0, turn_count - 1)
            console.print(f"[dim]Retrying: {last_msg[:60]}[/dim]")
            # Store retry message on runner for the REPL to pick up
            runner._retry_message = last_msg
        else:
            console.print("[dim]Nothing to retry.[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/copy":
        session = await runner.get_or_create_session(
            user_id=DEFAULT_USER_ID, session_id=session_id
        )
        from orxhestra.events.event import EventType as _ET

        # Find last agent response
        last_response: str | None = None
        for i in range(len(session.events) - 1, -1, -1):
            e = session.events[i]
            if e.type == _ET.AGENT_MESSAGE and e.text and not e.partial:
                last_response = e.text
                break
        if last_response:
            try:
                import subprocess

                subprocess.run(
                    ["pbcopy"], input=last_response.encode(), check=True,
                )
                console.print("[dim]Copied to clipboard.[/dim]")
            except Exception:
                console.print("[dim]Clipboard not available.[/dim]")
        else:
            console.print("[dim]No response to copy.[/dim]")
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    if cmd == "/help":
        console.print(_HELP_TEXT)
        return runner, session_id, todo_list, llm, model_name, turn_count, True

    console.print(f"[dim]Unknown command: {cmd}. Type /help[/dim]")
    return runner, session_id, todo_list, llm, model_name, turn_count, True




async def _repl(
    orx_path: Path,
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

    from orxhestra.cli.render import print_banner
    from orxhestra.cli.stream import stream_response

    console = Console()

    # Welcome banner
    print_banner(orx_path, model_name, workspace, console)
    console.print("  [dim]type /help for commands, Ctrl+D to exit[/dim]\n")

    # Try prompt_toolkit for history support
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

        # Multi-line input: triple-quote delimiters
        if user_input.startswith('"""') or user_input.startswith("'''"):
            delimiter: str = user_input[:3]
            ml_lines: list[str] = [user_input[3:]]
            while True:
                try:
                    if prompt_session:
                        ml_line: str = await prompt_session.prompt_async("... ")
                    else:
                        ml_line = input("... ")
                except (EOFError, KeyboardInterrupt):
                    break
                if ml_line.rstrip().endswith(delimiter):
                    ml_lines.append(ml_line.rstrip().removesuffix(delimiter))
                    break
                ml_lines.append(ml_line)
            user_input = "\n".join(ml_lines).strip()
            if not user_input:
                continue

        # Slash commands
        if user_input.startswith("/"):
            cmd_parts: list[str] = user_input.split(maxsplit=1)
            cmd_arg: str | None = cmd_parts[1].strip() if len(cmd_parts) > 1 else None

            runner, session_id, todo_list, llm, model_name, turn_count, should_continue = (
                await _handle_slash_command(
                    cmd_parts[0].lower(),
                    cmd_arg,
                    runner=runner,
                    session_id=session_id,
                    orx_path=orx_path,
                    model_name=model_name,
                    workspace=workspace,
                    todo_list=todo_list,
                    llm=llm,
                    turn_count=turn_count,
                    console=console,
                )
            )
            if not should_continue:
                break
            # Check if /retry set a message to re-send
            retry_msg: str | None = getattr(runner, "_retry_message", None)
            if retry_msg:
                runner._retry_message = None
                user_input = retry_msg
                # Fall through to stream_response below
            else:
                continue

        auto_approve = await stream_response(
            runner,
            session_id,
            user_input,
            console,
            Markdown,
            todo_list=todo_list,
            auto_approve=auto_approve,
        )
        turn_count += 1
        console.print()




async def _async_main() -> None:
    """Async entry point."""
    args = _parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.version:
        import orxhestra

        print(f"orx v{orxhestra.__version__}")
        return

    # Resolve orx file: explicit arg → ./orx.yaml → built-in default
    if args.orx_file:
        orx_path = Path(args.orx_file)
    else:
        local_orx = Path(args.workspace) / "orx.yaml"
        orx_path = local_orx if local_orx.exists() else _DEFAULT_ORX

    # Serve mode — start as A2A server
    if args.serve:
        import yaml

        from orxhestra.composer.composer import Composer
        from orxhestra.composer.schema import ComposeSpec

        if not orx_path.exists():
            print(f"Error: orx file not found: {orx_path}")
            sys.exit(1)

        # Add orx.yaml dir to sys.path for local tool imports
        orx_dir: str = str(orx_path.parent.resolve())
        if orx_dir not in sys.path:
            sys.path.insert(0, orx_dir)

        with open(orx_path) as f:
            raw: dict = yaml.safe_load(f)

        # Auto-inject a server section if missing
        if "server" not in raw:
            app_name: str = raw.get("runner", {}).get(
                "app_name", "orx-server"
            )
            raw["server"] = {
                "app_name": app_name,
                "url": f"http://localhost:{args.port}",
            }

        spec: ComposeSpec = ComposeSpec.model_validate(raw)
        composer = Composer(spec)
        root = await composer._build()
        app = composer._build_server(root)

        try:
            import uvicorn
        except ImportError:
            print(
                "Error: uvicorn is required for --serve:"
                " pip install uvicorn"
            )
            sys.exit(1)

        print(f"Starting A2A server on http://localhost:{args.port}")
        config = uvicorn.Config(app, host="0.0.0.0", port=args.port)
        server = uvicorn.Server(config)
        await server.serve()
        return

    runner, session_id, todo_list, llm = await _build_from_orx(
        orx_path, args.model, args.workspace
    )

    # Single-shot mode
    if args.command:
        try:
            from rich.console import Console
            from rich.markdown import Markdown
        except ImportError:
            print("Error: rich is required. Install with: pip install orxhestra[cli]")
            sys.exit(1)

        from orxhestra.cli.stream import stream_response

        console = Console()
        await stream_response(
            runner,
            session_id,
            args.command,
            console,
            Markdown,
            todo_list=todo_list,
            auto_approve=args.auto_approve,
        )
        return

    # Interactive REPL
    await _repl(
        orx_path,
        runner,
        session_id,
        args.model,
        args.workspace,
        todo_list=todo_list,
        llm=llm,
        auto_approve=args.auto_approve,
    )


def main() -> None:
    """Entry point for the ``orx`` command."""
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
