"""Orxhestra CLI — a terminal coding agent powered by any LLM.

Entry point for the ``orx`` console script.  Default invocation loads
an agent from a YAML spec (or the built-in coding agent when no file
is given) and drops the user into a pyink-based REPL with slash
commands, tool-approval prompts, and live streaming output.

Sub-modules:

- :mod:`~orxhestra.cli.app` — argparse wiring + main entry point.
- :mod:`~orxhestra.cli.builder` — constructs a :class:`~orxhestra.Runner`
  and :class:`~orxhestra.cli.state.ReplState` from an orx YAML.
- :mod:`~orxhestra.cli.commands` — slash-command handlers (``/clear``,
  ``/model``, ``/session``, ``/memory``, ...).
- :mod:`~orxhestra.cli.identity` — ``orx identity`` subcommand
  (``init`` / ``show`` / ``did-web``) for Ed25519 signing keys.
- :mod:`~orxhestra.cli.render` — Rich/pyink renderables for banner,
  tool calls, tool responses, todos, and turn summaries.
- :mod:`~orxhestra.cli.ink_app` — the pyink REPL component.
- :mod:`~orxhestra.cli.stream` — streams Runner events to a writer.
- :mod:`~orxhestra.cli.summarization` — ``/compact`` implementation.
- :mod:`~orxhestra.cli.memory` — auto-memory file discovery.
- :mod:`~orxhestra.cli.approval` — per-tool approval gating.
- :mod:`~orxhestra.cli.builtins` — register CLI-only tools
  (``write_todos``, ``task``) as composer builtins.
- :mod:`~orxhestra.cli.theme` — dark/light palettes and Rich styles.

Install via::

    pip install 'orxhestra[cli]'
"""
