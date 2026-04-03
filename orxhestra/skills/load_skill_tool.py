"""Skill tools — list, load, and load_resource as LangChain tools.

Implements the 3-tier progressive disclosure model:
  - ``list_skills``: L1 catalog (name + description).
  - ``load_skill``: L2 instructions + frontmatter metadata.
  - ``load_skill_resource``: L3 individual resource files on demand.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from orxhestra.skills.skill_store import BaseSkillStore


class LoadSkillInput(BaseModel):
    """Input schema for the load_skill tool."""

    name: str = Field(description="The name of the skill to load.")


class LoadSkillResourceInput(BaseModel):
    """Input schema for the load_skill_resource tool."""

    skill_name: str = Field(description="The name of the skill.")
    resource_path: str = Field(
        description=(
            "Relative path to the resource within the skill directory "
            "(e.g. 'scripts/run.py', 'references/api.md')."
        ),
    )


def make_load_skill_tool(store: BaseSkillStore) -> BaseTool:
    """Create a load_skill tool bound to the given skill store.

    Returns the skill's L2 instructions along with frontmatter metadata
    and a list of available L3 resources.

    Parameters
    ----------
    store : BaseSkillStore
        The skill store to load skills from.

    Returns
    -------
    BaseTool
    """

    async def load_skill(name: str) -> str:
        """Load a skill's full instruction content by name."""
        skill = await store.get_by_name(name)
        if skill is None:
            available = [s.name for s in await store.list_skills()]
            return (
                f"Skill '{name}' not found. "
                f"Available skills: {', '.join(available) or 'none'}."
            )

        sections: list[str] = [f"# {skill.name}"]

        # Frontmatter metadata
        if skill.frontmatter:
            fm = skill.frontmatter
            if fm.allowed_tools:
                sections.append(f"Allowed tools: {', '.join(fm.allowed_tools)}")
            if fm.compatibility:
                sections.append(f"Compatibility: {fm.compatibility}")
            if fm.license:
                sections.append(f"License: {fm.license}")

        # L2 instructions
        sections.append("")
        sections.append(skill.content)

        # L3 resource listing
        if skill.resources:
            sections.append("")
            sections.append("Available resources (use load_skill_resource to access):")
            for r in skill.resources:
                sections.append(f"  - {r.path} ({r.category})")

        return "\n".join(sections)

    return StructuredTool.from_function(
        coroutine=load_skill,
        name="load_skill",
        description=(
            "Load a skill's instructions by name. "
            "Use this when a skill is relevant to the current task."
        ),
        args_schema=LoadSkillInput,
    )


def make_list_skills_tool(store: BaseSkillStore) -> BaseTool:
    """Create a list_skills tool that returns the L1 skill catalog.

    Parameters
    ----------
    store : BaseSkillStore
        The skill store to list skills from.

    Returns
    -------
    BaseTool
    """

    async def list_skills() -> str:
        """List all available skills with their names and descriptions."""
        skills = await store.list_skills()
        if not skills:
            return "No skills available."
        lines = ["Available skills:"] + [
            f"  - {s.name}: {s.description}" for s in skills
        ]
        return "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=list_skills,
        name="list_skills",
        description="List all available skills with their names and descriptions.",
    )


def make_load_skill_resource_tool(store: BaseSkillStore) -> BaseTool:
    """Create a load_skill_resource tool for L3 on-demand resource loading.

    Parameters
    ----------
    store : BaseSkillStore
        The skill store to load resources from.

    Returns
    -------
    BaseTool
    """

    async def load_skill_resource(skill_name: str, resource_path: str) -> str:
        """Load a specific resource file from a skill."""
        result = await store.get_skill_resource(skill_name, resource_path)
        if result is None:
            skill = await store.get_by_name(skill_name)
            if skill is None:
                return f"Skill '{skill_name}' not found."
            available = [r.path for r in skill.resources]
            return (
                f"Resource '{resource_path}' not found in skill '{skill_name}'. "
                f"Available: {', '.join(available) or 'none'}."
            )
        return result

    return StructuredTool.from_function(
        coroutine=load_skill_resource,
        name="load_skill_resource",
        description=(
            "Load a specific resource file from a skill by its relative path. "
            "Use after loading a skill to access its scripts, references, or assets."
        ),
        args_schema=LoadSkillResourceInput,
    )
