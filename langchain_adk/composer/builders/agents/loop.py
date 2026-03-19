"""LoopAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.composer.errors import ComposerError
from langchain_adk.composer.schema import AgentDef, ComposeSpec

if TYPE_CHECKING:
    from langchain_adk.composer.builders.agents import Helpers


async def build(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,
    *,
    helpers: Helpers,
) -> BaseAgent:
    """Build a ``LoopAgent`` from a YAML definition."""
    from langchain_adk.agents.loop_agent import LoopAgent
    from langchain_adk.composer.builders.tools import import_object

    if not agent_def.agents:
        msg = f"loop agent '{name}' must have an 'agents' list"
        raise ComposerError(msg)

    sub_agents = [await helpers.build_agent(n) for n in agent_def.agents]
    kwargs: dict[str, Any] = {
        "name": name,
        "agents": sub_agents,
        "description": agent_def.description,
        "max_iterations": agent_def.max_iterations
        or spec.defaults.max_iterations,
    }
    if agent_def.should_continue:
        kwargs["should_continue"] = import_object(agent_def.should_continue)

    return LoopAgent(**kwargs)
