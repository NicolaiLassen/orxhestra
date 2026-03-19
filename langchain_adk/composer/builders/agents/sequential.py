"""SequentialAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    """Build a ``SequentialAgent`` from a YAML definition."""
    from langchain_adk.agents.sequential_agent import SequentialAgent

    if not agent_def.agents:
        msg = f"sequential agent '{name}' must have an 'agents' list"
        raise ComposerError(msg)

    sub_agents = [await helpers.build_agent(n) for n in agent_def.agents]
    return SequentialAgent(
        name=name, agents=sub_agents, description=agent_def.description
    )
