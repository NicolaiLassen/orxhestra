"""Orx CLI — terminal agent powered by any LLM.

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
from typing import TYPE_CHECKING

from orxhestra.cli.config import DEFAULT_MODEL

if TYPE_CHECKING:
    from rich.console import Console

    from orxhestra.cli.state import ReplState

_CLI_DIR: Path = Path(__file__).parent
_DEFAULT_ORX: Path = _CLI_DIR / "orx.yaml"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="orx",
        description="Orx — terminal agent powered by any LLM.",
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


async def _serve(orx_path: Path, port: int) -> None:
    """Start as an A2A server.

    Parameters
    ----------
    orx_path : Path
        Path to the orx YAML file.
    port : int
        Port number for the HTTP server.
    """
    import yaml

    from orxhestra.composer.composer import Composer
    from orxhestra.composer.schema import ComposeSpec

    if not orx_path.exists():
        print(f"Error: orx file not found: {orx_path}")
        sys.exit(1)

    orx_dir: str = str(orx_path.parent.resolve())
    if orx_dir not in sys.path:
        sys.path.insert(0, orx_dir)

    with open(orx_path) as f:
        raw: dict = yaml.safe_load(f)

    if "server" not in raw:
        app_name: str = raw.get("runner", {}).get(
            "app_name", "orx-server"
        )
        raw["server"] = {
            "app_name": app_name,
            "url": f"http://localhost:{port}",
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

    print(f"Starting A2A server on http://localhost:{port}")
    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def _async_main() -> None:
    """Async entry point."""
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.version:
        import orxhestra

        print(f"orx v{orxhestra.__version__}")
        return

    if args.orx_file:
        orx_path = Path(args.orx_file)
    else:
        local_orx = Path(args.workspace) / "orx.yaml"
        orx_path = local_orx if local_orx.exists() else _DEFAULT_ORX

    if args.serve:
        await _serve(orx_path, args.port)
        return

    from orxhestra.cli.builder import build_from_orx

    state: ReplState = await build_from_orx(
        orx_path, args.model, args.workspace
    )
    state.auto_approve = args.auto_approve

    if args.command:
        try:
            from rich.markdown import Markdown
        except ImportError:
            print(
                "Error: rich is required."
                " Install with: pip install orxhestra[cli]"
            )
            sys.exit(1)

        from orxhestra.cli.stream import stream_response
        from orxhestra.cli.theme import make_console
        from orxhestra.cli.writer import ConsoleWriter

        console: Console = make_console()
        w = ConsoleWriter(console)
        await stream_response(
            state.runner,
            state.session_id,
            args.command,
            w,
            Markdown,
            todo_list=state.todo_list,
            auto_approve=state.auto_approve,
        )
        return

    # Return state so main() can launch the pyink app
    # outside the asyncio loop (pyink manages its own event loop).
    return orx_path, state, args.workspace


def _graceful_shutdown() -> None:
    """Clean up resources and suppress noisy errors on exit."""
    # Flush and shut down OpenTelemetry (Langfuse, etc.)
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=2000)
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass

    # Suppress errors from OTel generators garbage-collected during
    # interpreter shutdown (e.g. "Failed to detach context").
    for name in ("opentelemetry.context", "opentelemetry.trace"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def main() -> None:
    """Entry point for the ``orx`` command."""
    try:
        result = asyncio.run(_async_main())
    except KeyboardInterrupt:
        return
    finally:
        _graceful_shutdown()

    if result is not None:
        orx_path, state, workspace = result
        from orxhestra.cli.ink_app import run_ink_app
        from orxhestra.cli.theme import make_console

        console = make_console()
        try:
            run_ink_app(state, console, orx_path, workspace)
        except KeyboardInterrupt:
            pass
        finally:
            _graceful_shutdown()


if __name__ == "__main__":
    main()
