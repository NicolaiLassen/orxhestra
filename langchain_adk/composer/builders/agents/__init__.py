"""Agent builder registry — one build function per agent type."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.composer.builders.agents import llm, loop, parallel, react, sequential
from langchain_adk.composer.schema import AgentDef, ComposeSpec, ModelConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

BuildFn = Callable[
    [str, AgentDef, ComposeSpec],
    Awaitable[BaseAgent],
]

_REGISTRY: dict[str, BuildFn] = {}


def register(agent_type: str, fn: BuildFn) -> None:
    """Register a build function for a YAML agent type.

    Example::

        async def build_custom(name, agent_def, spec, *, helpers):
            return MyCustomAgent(name=name, ...)

        register("custom", build_custom)
    """
    _REGISTRY[agent_type] = fn


def get(agent_type: str) -> BuildFn | None:
    """Look up a build function by agent type name."""
    return _REGISTRY.get(agent_type)


class Helpers:
    """Dependency bag passed to every build function.

    Attributes
    ----------
    resolve_tools:
        Resolve all tool references for an agent definition.
    resolve_model:
        Merge agent-level model config with defaults.
    build_agent:
        Recursively build a sub-agent by name.
    """

    __slots__ = ("build_agent", "resolve_model", "resolve_tools")

    def __init__(
        self,
        *,
        resolve_tools: Callable[[AgentDef], Awaitable[list[BaseTool]]],
        resolve_model: Callable[[AgentDef], ModelConfig],
        build_agent: Callable[[str], Awaitable[BaseAgent]],
    ) -> None:
        self.resolve_tools = resolve_tools
        self.resolve_model = resolve_model
        self.build_agent = build_agent


# Auto-register built-in agent builders.
register("llm", llm.build)
register("react", react.build)
register("sequential", sequential.build)
register("parallel", parallel.build)
register("loop", loop.build)
