"""Build a Runner and :class:`~orxhestra.cli.state.ReplState` from orx YAML.

The entry point :func:`build_from_orx` takes a YAML path (or the
built-in coding-agent spec), composes the agent tree via
:class:`~orxhestra.composer.composer.Composer`, configures the model
selected by ``--model``, wires up auto-memory + CLI built-in tools,
and returns a ready-to-run :class:`~orxhestra.cli.state.ReplState`.

Called from both the interactive REPL path and the ``-c`` single-shot
path in :mod:`orxhestra.cli.app`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from orxhestra.cli.config import APP_NAME

if TYPE_CHECKING:
    from orxhestra.cli.state import ReplState


def _set_human_input_callbacks(agent: Any, callback: Any) -> None:
    """Walk the agent tree and wire up human_input tool callbacks.

    Parameters
    ----------
    agent : Any
        Root agent (or sub-agent) to traverse.
    callback : Any
        Async callable to set on each human_input tool.
    """
    if hasattr(agent, "_tools"):
        for tool in agent._tools.values():
            if tool.name == "human_input" and hasattr(tool, "set_callback"):
                tool.set_callback(callback)
    for child in getattr(agent, "sub_agents", []):
        _set_human_input_callbacks(child, callback)


def _inject_context(
    raw: dict,
    workspace: str,
    memory_content: str,
    local_context: str,
) -> None:
    """Append workspace, memory, and local context to LLM agent instructions.

    Parameters
    ----------
    raw : dict
        Parsed orx YAML specification (mutated in place).
    workspace : str
        Workspace directory path.
    memory_content : str
        Loaded AGENTS.md content to inject.
    local_context : str
        Detected project context string.
    """
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


async def build_from_orx(
    orx_path: Path,
    model_name: str,
    workspace: str,
    *,
    effort: str | None = None,
) -> ReplState:
    """Build a Runner from an orx YAML and return populated ReplState.

    Parameters
    ----------
    orx_path : Path
        Path to the orx YAML file.
    model_name : str
        Name of the LLM model to use.
    workspace : str
        Workspace directory path.

    Returns
    -------
    ReplState
        Fully initialized REPL state with runner, session, and tools.
    """
    import yaml

    from orxhestra.cli.builtins import get_todo_list, register_cli_builtins
    from orxhestra.cli.context_injection import collect_local_context
    from orxhestra.cli.memory import load_agents_md
    from orxhestra.cli.models import create_llm, detect_provider
    from orxhestra.cli.state import ReplState
    from orxhestra.composer.composer import Composer
    from orxhestra.composer.schema import ComposeSpec

    if not orx_path.exists():
        print(f"Error: orx file not found: {orx_path}")
        sys.exit(1)

    os.environ["AGENT_WORKSPACE"] = workspace
    model = create_llm(model_name, effort=effort)
    register_cli_builtins(workspace, model)

    with open(orx_path) as f:
        raw: dict = yaml.safe_load(f)

    # Only inject CLI model if the YAML doesn't already define one.
    defaults: dict = raw.get("defaults", {})
    if "model" not in defaults:
        if "defaults" not in raw:
            raw["defaults"] = {}
        from orxhestra.cli.config import effort_model_kwargs

        provider = detect_provider(model_name)
        model_spec: dict = {
            "provider": provider,
            "name": model_name,
        }
        if effort:
            model_spec.update(effort_model_kwargs(provider, effort))
        raw["defaults"]["model"] = model_spec
    else:
        # Use the YAML model name for the banner display.
        yaml_model = defaults["model"]
        if isinstance(yaml_model, dict):
            model_name = yaml_model.get("name", model_name)
        elif isinstance(yaml_model, str):
            model_name = yaml_model

    memory_content: str = load_agents_md(workspace)
    local_context: str = await collect_local_context(workspace)
    _inject_context(raw, workspace, memory_content, local_context)

    orx_dir: str = str(orx_path.parent.resolve())
    if orx_dir not in sys.path:
        sys.path.insert(0, orx_dir)

    spec: ComposeSpec = ComposeSpec.model_validate(raw)
    composer = Composer(spec)
    root = await composer.build()

    async def _human_input_prompt(question: str) -> str:
        try:
            return input(f"\n  ? {question}\n  > ")
        except (EOFError, KeyboardInterrupt):
            return "(user declined to answer)"

    _set_human_input_callbacks(root, _human_input_prompt)

    if spec.runner is not None:
        runner = await composer.build_runner(root)
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
                model=model,
            ),
        )

    return ReplState(
        runner=runner,
        session_id=str(uuid4()),
        model_name=model_name,
        effort=effort,
        spec_raw=raw,
        todo_list=get_todo_list(),
        model=model,
    )
