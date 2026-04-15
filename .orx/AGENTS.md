# Orxhestra repo notes

- Python project managed with uv; primary test command: `pytest`.
- Package name/version live in `pyproject.toml` and `orxhestra/__init__.py`.
- Main SDK entrypoints are exported from `orxhestra/__init__.py`.
- CLI entrypoint is `orx` via `orxhestra.cli.app:main`.
- CLI builds agents from YAML through `orxhestra.cli.builder.build_from_orx()` and injects workspace/memory/local context into LLM/react instructions.
- Composer loads validated YAML specs and builds agent trees/runners/servers from `orxhestra/composer/composer.py`.
- Runner orchestrates session-backed streaming in `orxhestra/runner.py`.
- Todo and memory capabilities are implemented as tools in `orxhestra/tools/todo_tool.py` and `orxhestra/tools/memory_tools.py`.
- Docs are extensive under `docs/`; examples live under `examples/`; tests under `tests/`.
- Current uncommitted changes present in: `orxhestra/cli/ink_app.py`, `orxhestra/cli/render.py`, `orxhestra/cli/writer.py`, `uv.lock`.
