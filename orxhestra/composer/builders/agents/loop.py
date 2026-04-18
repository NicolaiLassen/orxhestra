"""LoopAgent builder — thin wrapper over the composite helper."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.builders.agents._common import build_composite
from orxhestra.composer.schema import AgentDef, ComposeSpec

if TYPE_CHECKING:
    from orxhestra.composer.builders.agents import Helpers


async def build(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,
    *,
    helpers: Helpers,
) -> BaseAgent:
    """Build a :class:`~orxhestra.agents.loop_agent.LoopAgent`.

    Delegates to
    :func:`~orxhestra.composer.builders.agents._common.build_composite`
    for the shared composite-agent construction path, layering
    loop-specific knobs (``max_iterations``, ``should_continue``) on
    top via ``extra_kwargs``.

    Parameters
    ----------
    name : str
        Agent name from the YAML spec.
    agent_def : AgentDef
        YAML agent definition.  Must carry a non-empty ``agents`` list.
    spec : ComposeSpec
        Full compose specification — used to pick up the default
        ``max_iterations`` when the loop-level field is unset.
    helpers : Helpers
        Builder dependencies; ``build_agent`` resolves each sub-agent.

    Returns
    -------
    BaseAgent
        The constructed ``LoopAgent``.

    Raises
    ------
    ComposerError
        When ``agent_def.agents`` is missing or empty.
    """
    from orxhestra.agents.loop_agent import LoopAgent
    from orxhestra.composer.builders.tools import import_object

    extra: dict[str, Any] = {
        "max_iterations": agent_def.max_iterations or spec.defaults.max_iterations,
    }
    if agent_def.should_continue:
        extra["should_continue"] = import_object(agent_def.should_continue)

    return await build_composite(
        name,
        agent_def,
        helpers=helpers,
        agent_cls=LoopAgent,
        kind="loop",
        extra_kwargs=extra,
    )
