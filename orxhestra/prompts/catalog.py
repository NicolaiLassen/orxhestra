"""Prompt catalog - programmatic system prompt builder.

Builds a system prompt from a typed PromptContext by assembling
independent blocks and joining them with double newlines. Empty blocks
are automatically excluded.

Examples
--------
>>> from prompts.catalog import build_system_prompt
>>> from prompts.context import PromptContext
>>>
>>> ctx = PromptContext(
...     agent_name="ResearchAgent",
...     goal="Find and summarise recent AI papers.",
...     instructions="Always cite sources. Be concise.",
...     skills=[{"name": "web_search", "description": "Search the web"}],
... )
>>> system_prompt = build_system_prompt(ctx)
"""

from __future__ import annotations

from orxhestra.prompts.context import PromptContext


def _skills_block(skills: list[dict]) -> str:
    """Render the available skills section."""
    lines = ["Available skills (call load_skill to access):"] + [
        f"  - {s['name']}" + (f": {s['description']}" if s.get("description") else "")
        for s in skills
    ]
    return "\n".join(lines)


def _agents_block(agents: list[dict]) -> str:
    """Render the available agents section."""
    lines = ["Available agents (call transfer_to_agent to delegate):"] + [
        f"  - {a['name']}" + (f" - {a['description']}" if a.get("description") else "")
        for a in agents
    ]
    return "\n".join(lines)


def _tasks_block(tasks: list[dict]) -> str:
    """Render the current tasks section."""
    lines = ["Tasks:"] + [
        f"  [{t.get('tag', 'TASK')}] {t['title']}: {t.get('description', '')}"
        for t in tasks
    ]
    return "\n".join(lines)




def build_system_prompt(ctx: PromptContext) -> str:
    """Render a system prompt from a PromptContext.

    Each section is only included when the relevant field is non-empty.
    Sections are separated by double newlines.

    Parameters
    ----------
    ctx : PromptContext
        The populated prompt context.

    Returns
    -------
    str
        A fully rendered system prompt string.
    """
    return "\n\n".join(filter(None, [
        f"You are {ctx.agent_name}.",
        f"Today's date: {ctx.current_date}.",
        f"Your goal: {ctx.goal}" if ctx.goal else "",
        ctx.instructions,
        _skills_block(ctx.skills) if ctx.skills else "",
        _agents_block(ctx.agents) if ctx.agents else "",
        _tasks_block(ctx.tasks) if ctx.tasks else "",
        "Workflow:\n" + ctx.workflow_instructions if ctx.workflow_instructions else "",
        *ctx.extra_sections,
    ]))
