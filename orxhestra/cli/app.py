"""Orx CLI — terminal agent powered by any LLM.

Default invocation runs an agent from a YAML spec (or the built-in
coding agent when no file is given).  An ``identity`` subcommand
handles Ed25519 signing keys without touching the REPL/serve paths.

Usage::

    orx                                # uses ./orx.yaml if present, else built-in default
    orx --model claude-sonnet-4-6      # use a specific model
    orx my-agents.yaml                 # run a specific orx file
    orx -c "fix the tests"             # single-shot command
    orx --auto-approve                 # skip approval prompts
    orx --identity ~/.orx/agent.key    # sign every emitted event
    orx identity init                  # generate a new Ed25519 identity
    orx identity show                  # print an existing identity's DID
    orx identity did-web example.com   # render a did.json document

See Also
--------
orxhestra.cli.identity : Implementation of the ``identity`` subcommand.
orxhestra.cli.commands : Slash-command handlers available inside the REPL.
orxhestra.composer.schema.IdentityConfig : YAML counterpart of ``--identity``.
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


_SUBCOMMANDS: frozenset[str] = frozenset({"identity"})
"""Names reserved for ``orx`` subcommands — routed to dedicated parsers."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the default ``orx`` invocation (no subcommand).

    Two concerns are kept cleanly separate:

    - This parser owns the default path:
      ``orx [orx_file] [--flags]``.  It never sees a subcommand
      argument.
    - Subcommands like ``identity`` are routed to their own parser in
      :func:`_parse_identity_args` before this function ever runs.

    The split avoids argparse's ambiguity when both a positional
    ``orx_file`` and ``add_subparsers`` are declared on the same
    parser.

    Parameters
    ----------
    argv : list[str], optional
        Argument vector to parse.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        prog="orx",
        description="Orx — terminal agent powered by any LLM.",
        epilog=(
            "Subcommands: "
            + ", ".join(sorted(_SUBCOMMANDS))
            + ".  Run `orx <subcommand> --help` for details."
        ),
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
    from orxhestra.cli.identity import DEFAULT_KEY_PATH as _DEFAULT_KEY_PATH

    parser.add_argument(
        "--identity",
        nargs="?",
        const=str(_DEFAULT_KEY_PATH),
        default=os.environ.get("ORX_IDENTITY"),
        metavar="PATH",
        help=(
            "Path to an Ed25519 signing-key file (created via "
            "`orx identity init`). When set, every event emitted by "
            "agents running under the Runner is signed with this key. "
            "Pass without a value to use the default key at "
            f"{_DEFAULT_KEY_PATH}. Also reads $ORX_IDENTITY."
        ),
    )
    parser.add_argument(
        "--identity-password",
        default=os.environ.get("ORX_KEY_PASSWORD"),
        metavar="PASSWORD",
        help=(
            "Password used to decrypt the signing-key file at rest. "
            "Also reads $ORX_KEY_PASSWORD."
        ),
    )
    return parser.parse_args(argv)


def _parse_identity_args(argv: list[str]) -> argparse.Namespace:
    """Parse ``orx identity <action> ...`` into a dedicated namespace.

    Called by :func:`main` when the first positional argument is
    ``identity``.  Returns a namespace with ``action`` plus the
    fields the selected action declared.

    Parameters
    ----------
    argv : list[str]
        Arguments *after* the literal ``"identity"`` token.

    Returns
    -------
    argparse.Namespace
    """
    from orxhestra.cli.identity import DEFAULT_KEY_PATH

    parser = argparse.ArgumentParser(
        prog="orx identity",
        description=(
            "Generate, inspect, or export Ed25519 signing identities "
            "used by orxhestra's trust layer."
        ),
    )
    subs = parser.add_subparsers(dest="action", required=True)

    init_p = subs.add_parser(
        "init", help="Generate a new Ed25519 identity key.",
    )
    init_p.add_argument(
        "--path",
        default=str(DEFAULT_KEY_PATH),
        help="Where to write the JSON key file.",
    )
    init_p.add_argument(
        "--encrypt",
        action="store_true",
        help="Encrypt the private key at rest using $ORX_KEY_PASSWORD.",
    )

    show_p = subs.add_parser(
        "show", help="Print the DID of an existing key file.",
    )
    show_p.add_argument(
        "--path",
        default=str(DEFAULT_KEY_PATH),
        help="Path to the JSON key file.",
    )

    did_web_p = subs.add_parser(
        "did-web",
        help="Render a did.json document for a did:web identity.",
    )
    did_web_p.add_argument(
        "domain",
        help="Host portion of the did:web (e.g. example.com).",
    )
    did_web_p.add_argument(
        "sub_path",
        nargs="*",
        help="Optional sub-path segments (e.g. agents researcher).",
    )
    did_web_p.add_argument(
        "--path",
        default=str(DEFAULT_KEY_PATH),
        help="Key file whose public key is embedded in the document.",
    )

    return parser.parse_args(argv)


def _apply_identity_to_state(
    state: ReplState,
    key_file: str,
    password: str | None,
) -> None:
    """Stamp an Ed25519 identity onto every agent in the built REPL state.

    Loads or creates the key file via
    :func:`~orxhestra.security.crypto.load_or_create_signing_key`,
    walks the agent tree, and assigns the identity to every agent
    that doesn't already have one.  Also records ``signer_did`` and
    ``identity_key_path`` on ``state`` so the startup banner and the
    ``/session`` slash command can surface the active identity.

    No middleware is registered here — signing activates at event
    emission via :meth:`BaseAgent._emit_event`.  Callers who want
    verification or audit on top of signing should declare ``trust:``
    / ``attestation:`` blocks in their orx YAML, which installs
    :class:`~orxhestra.middleware.trust.TrustMiddleware` and
    :class:`~orxhestra.middleware.attestation.AttestationMiddleware`
    on the Runner.

    Parameters
    ----------
    state : ReplState
        Freshly built REPL state whose ``runner.agent`` is the tree
        root.
    key_file : str
        Path to the JSON key file.
    password : str, optional
        Password for decrypting an encrypted key file.
    """
    from orxhestra.security.crypto import load_or_create_signing_key

    signing_key, did = load_or_create_signing_key(
        key_file, encryption_password=password,
    )

    stack = [state.runner.agent]
    seen: set[int] = set()
    while stack:
        agent = stack.pop()
        if id(agent) in seen:
            continue
        seen.add(id(agent))
        if agent.signing_key is None and not agent.signing_did:
            agent.signing_key = signing_key
            agent.signing_did = did
        stack.extend(agent.sub_agents)

    # Record the active identity so the startup banner and `/session`
    # can surface it to the user — no separate stderr line needed.
    state.signer_did = did
    state.identity_key_path = key_file


async def _serve(orx_path: Path, port: int) -> None:
    """Expose an orx YAML agent tree as an A2A server.

    Loads the YAML via :class:`~orxhestra.composer.composer.Composer`,
    wraps the root agent in an A2A server via
    :meth:`~orxhestra.composer.composer.Composer._build_server`, and
    serves it under uvicorn on the given port.  Adds a default
    ``server:`` block to the spec when the YAML doesn't declare one.

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
    root = await composer.build()
    app = await composer.build_server(root)

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


async def _async_main(args: argparse.Namespace) -> None:
    """Dispatch the default (non-subcommand) orx invocation.

    Handles ``--version``, ``--serve``, and ``-c`` modes inline.  For
    the interactive REPL, returns ``(orx_path, state, workspace)`` so
    :func:`main` can launch the pyink app outside the asyncio loop.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from :func:`_parse_args`.

    Returns
    -------
    tuple or None
        ``(orx_path, state, workspace)`` for REPL mode, ``None`` for
        every other mode.
    """
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

    if args.identity:
        _apply_identity_to_state(
            state, args.identity, args.identity_password,
        )

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
    """Flush OpenTelemetry and suppress interpreter-shutdown noise.

    Called from :func:`main`'s ``finally`` block so spans aren't lost
    when the REPL exits.  Silences the noisy
    ``opentelemetry.context`` / ``opentelemetry.trace`` loggers that
    otherwise print "Failed to detach context" at interpreter shutdown.
    """
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
    """Entry point for the ``orx`` console script.

    Routing:

    - ``orx identity ...`` is parsed by :func:`_parse_identity_args`
      and delegated to :func:`orxhestra.cli.identity.run_parsed`.
    - Every other invocation falls through to :func:`_parse_args` and
      :func:`_async_main` for the REPL, ``--serve``, or ``-c`` flow.

    The pyink REPL is launched outside the asyncio loop because it
    manages its own event loop.
    """
    raw = sys.argv[1:]
    if raw and raw[0] in _SUBCOMMANDS:
        subcommand = raw[0]
        if subcommand == "identity":
            from orxhestra.cli.identity import run_parsed

            args = _parse_identity_args(raw[1:])
            sys.exit(run_parsed(args))

    args = _parse_args(raw)

    try:
        result = asyncio.run(_async_main(args))
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
