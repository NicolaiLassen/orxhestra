"""ReActAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_adk.agents.base_agent import BaseAgent
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
    """Build a ``ReActAgent`` from a YAML definition."""
    from langchain_adk.agents.react_agent import ReActAgent
    from langchain_adk.composer.builders.models import create

    model_cfg = helpers.resolve_model(agent_def)
    extra = {
        k: v
        for k, v in model_cfg.model_dump().items()
        if k not in ("provider", "name", "extra") and v is not None
    }
    extra.update(model_cfg.extra)
    llm = create(model_cfg.provider, model_cfg.name, **extra)

    tools = await helpers.resolve_tools(agent_def)
    max_iter = agent_def.max_iterations or spec.defaults.max_iterations

    return ReActAgent(
        name=name,
        llm=llm,
        tools=tools or None,
        description=agent_def.description,
        max_iterations=max_iter,
    )
