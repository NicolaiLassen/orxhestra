"""ParallelAgent builder — thin wrapper over the composite helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.builders.agents._common import build_composite
from orxhestra.composer.schema import AgentDef, ComposeSpec

if TYPE_CHECKING:
    from orxhestra.composer.builders.agents import Helpers


async def build(
    name: str,
    agent_def: AgentDef,
    spec: ComposeSpec,  # noqa: ARG001  (parameter required by BuildFn protocol)
    *,
    helpers: Helpers,
) -> BaseAgent:
    """Build a :class:`~orxhestra.agents.parallel_agent.ParallelAgent`.

    Delegates to :func:`~orxhestra.composer.builders.agents._common.build_composite`
    for the shared "validate ``agents`` list, build sub-agents,
    instantiate" path.

    Parameters
    ----------
    name : str
        Agent name from the YAML spec.
    agent_def : AgentDef
        YAML agent definition.  Must carry a non-empty ``agents`` list.
    spec : ComposeSpec
        Full compose specification (unused here; required by the
        :class:`~orxhestra.composer.builders.agents.BuildFn` protocol).
    helpers : Helpers
        Builder dependencies; ``build_agent`` resolves each sub-agent.

    Returns
    -------
    BaseAgent
        The constructed ``ParallelAgent``.

    Raises
    ------
    ComposerError
        When ``agent_def.agents`` is missing or empty.
    """
    from orxhestra.agents.parallel_agent import ParallelAgent

    return await build_composite(
        name,
        agent_def,
        helpers=helpers,
        agent_cls=ParallelAgent,
        kind="parallel",
    )
