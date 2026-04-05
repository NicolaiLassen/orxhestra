"""Shared builder logic for LLM-based agents (LlmAgent, ReActAgent)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orxhestra.composer.schema import AgentDef, ComposeSpec, PlannerDef

if TYPE_CHECKING:
    from orxhestra.composer.builders.agents import Helpers
    from orxhestra.planners.base_planner import BasePlanner


async def resolve_llm_kwargs(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,
    *,
    helpers: Helpers,
) -> dict[str, Any]:
    """Build the common keyword arguments for LlmAgent / ReActAgent.

    Handles model creation, tool resolution, instructions, planners,
    output schemas, and skills -- everything shared between the two
    agent types.

    Parameters
    ----------
    name : str
        Agent name from the YAML spec.
    agent_def : AgentDef
        Parsed agent definition.
    spec : ComposeSpec
        Full compose specification.
    helpers : Helpers
        Builder dependency bag.

    Returns
    -------
    dict[str, Any]
        Keyword arguments ready to pass to the agent constructor.
    """
    from orxhestra.composer.builders.models import create
    from orxhestra.composer.builders.tools import import_object

    model_cfg = helpers.resolve_model(agent_def)
    # Collect known fields (temperature, etc.) + any extra keys from YAML.
    model_kwargs: dict[str, Any] = {
        k: v
        for k, v in model_cfg.model_dump().items()
        if k not in ("provider", "name") and v is not None
    }
    if model_cfg.model_extra:
        model_kwargs.update(model_cfg.model_extra)
    llm = create(model_cfg.provider, model_cfg.name, **model_kwargs)

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
        from datetime import datetime, timezone

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        kwargs["instructions"] = f"Today's date: {date_str}\n\n{agent_def.instructions}"

    if agent_def.output_key:
        kwargs["output_key"] = agent_def.output_key

    if agent_def.include_contents:
        kwargs["include_contents"] = agent_def.include_contents

    if agent_def.planner:
        planner = _build_planner(agent_def.planner)
        kwargs["planner"] = planner
        if agent_def.planner.type == "task":
            # Wire up a shared TodoList + write_todos tool for the planner.
            from orxhestra.tools.todo_tool import TodoList, make_todo_tool

            todo_list = TodoList()
            planner.set_todo_list(todo_list)
            existing = kwargs.get("tools") or []
            kwargs["tools"] = [*existing, make_todo_tool(todo_list)]

    if agent_def.output_schema:
        kwargs["output_schema"] = import_object(agent_def.output_schema)

    if agent_def.skills:
        from pathlib import Path

        from orxhestra.composer.builders.tools import resolve_mcp_skill
        from orxhestra.composer.errors import ComposerError
        from orxhestra.skills import (
            InMemorySkillStore,
            Skill,
            make_list_skills_tool,
            make_load_skill_resource_tool,
            make_load_skill_tool,
        )

        skill_items = []
        for skill_name in agent_def.skills:
            skill_def = spec.skills.get(skill_name)
            if skill_def is None:
                msg = f"Skill '{skill_name}' not found in skills section"
                raise ComposerError(msg)
            if skill_def.mcp:
                skill = await resolve_mcp_skill(
                    name=skill_def.name,
                    description=skill_def.description,
                    url=skill_def.mcp.url,
                    server_path=skill_def.mcp.server,
                )
            elif skill_def.directory:
                from orxhestra.skills.loader import scan_skill_directory

                skill = scan_skill_directory(Path(skill_def.directory))
            else:
                skill = Skill(
                    name=skill_def.name,
                    description=skill_def.description,
                    content=skill_def.content or "",
                )
            skill_items.append(skill)
        store = InMemorySkillStore(skill_items)
        has_resources = any(s.resources for s in skill_items)
        existing = kwargs.get("tools") or []
        skill_tools = [
            make_list_skills_tool(store),
            make_load_skill_tool(store),
        ]
        if has_resources:
            skill_tools.append(make_load_skill_resource_tool(store))
        kwargs["tools"] = [*existing, *skill_tools]

    return kwargs


def _build_planner(planner_def: PlannerDef) -> BasePlanner:
    """Build a planner from its definition.

    Parameters
    ----------
    planner_def : PlannerDef
        Planner configuration from the YAML spec.

    Returns
    -------
    BasePlanner
        Constructed planner instance.

    Raises
    ------
    ComposerError
        If the planner type is unknown.
    """
    if planner_def.type == "plan_react":
        from orxhestra.planners.plan_re_act_planner import PlanReActPlanner

        return PlanReActPlanner()
    if planner_def.type == "task":
        from orxhestra.planners.task_planner import TaskPlanner

        return TaskPlanner(tasks=planner_def.tasks)
    from orxhestra.composer.errors import ComposerError

    msg = f"Unknown planner type: '{planner_def.type}'"
    raise ComposerError(msg)
