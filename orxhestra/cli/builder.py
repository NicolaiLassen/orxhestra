"""Build a Runner and REPL state from an orx YAML specification."""

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
    """Walk the agent tree and wire up human_input tool callbacks."""
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


async def build_from_orx(
    orx_path: Path,
    model_name: str,
    workspace: str,
) -> ReplState:
    """Build a Runner from an orx YAML and return populated ReplState."""
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
    llm = create_llm(model_name)
    register_cli_builtins(workspace, llm)

    with open(orx_path) as f:
        raw: dict = yaml.safe_load(f)

    if "defaults" not in raw:
        raw["defaults"] = {}
    raw["defaults"]["model"] = {
        "provider": detect_provider(model_name),
        "name": model_name,
    }

    memory_content: str = load_agents_md(workspace)
    local_context: str = await collect_local_context(workspace)
    _inject_context(raw, workspace, memory_content, local_context)

    orx_dir: str = str(orx_path.parent.resolve())
    if orx_dir not in sys.path:
        sys.path.insert(0, orx_dir)

    spec: ComposeSpec = ComposeSpec.model_validate(raw)
    composer = Composer(spec)
    root = await composer._build()

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

    return ReplState(
        runner=runner,
        session_id=str(uuid4()),
        model_name=model_name,
        todo_list=get_todo_list(),
        llm=llm,
    )
