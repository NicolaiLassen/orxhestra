"""LlmAgent builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.composer.schema import AgentDef, ComposeSpec, PlannerDef

if TYPE_CHECKING:
    from langchain_adk.composer.builders.agents import Helpers
    from langchain_adk.planners.base_planner import BasePlanner


async def build(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,
    *,
    helpers: Helpers,
) -> BaseAgent:
    """Build an ``LlmAgent`` from a YAML definition."""
    from langchain_adk.agents.llm_agent import LlmAgent
    from langchain_adk.composer.builders.models import create
    from langchain_adk.composer.builders.tools import import_object, resolve_builtin

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

    kwargs: dict[str, Any] = {
        "name": name,
        "llm": llm,
        "tools": tools or None,
        "description": agent_def.description,
        "max_iterations": max_iter,
    }

    if agent_def.instructions:
        kwargs["instructions"] = agent_def.instructions

    if agent_def.planner:
        planner = _build_planner(agent_def.planner)
        kwargs["planner"] = planner
        if agent_def.planner.type == "task":
            existing = kwargs.get("tools") or []
            kwargs["tools"] = [*existing, planner.get_manage_tasks_tool()]

    if agent_def.output_schema:
        kwargs["output_schema"] = import_object(agent_def.output_schema)

    if spec.skills:
        existing = kwargs.get("tools") or []
        kwargs["tools"] = [
            *existing,
            resolve_builtin("list_skills"),
            resolve_builtin("load_skill"),
        ]

    return LlmAgent(**kwargs)


def _build_planner(planner_def: PlannerDef) -> BasePlanner:
    """Build a planner from its definition."""
    if planner_def.type == "plan_react":
        from langchain_adk.planners.plan_re_act_planner import PlanReActPlanner

        return PlanReActPlanner()
    if planner_def.type == "task":
        from langchain_adk.planners.task_planner import TaskPlanner

        return TaskPlanner(tasks=planner_def.tasks)
    from langchain_adk.composer.errors import ComposerError

    msg = f"Unknown planner type: '{planner_def.type}'"
    raise ComposerError(msg)
