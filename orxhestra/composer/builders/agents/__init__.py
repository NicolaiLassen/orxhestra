"""Agent builder registry — one build function per YAML ``type:`` key.

Every built-in agent type (``llm``, ``react``, ``sequential``,
``parallel``, ``loop``, ``a2a``) ships its own builder in this
package.  Third-party types plug in the same way — call
:func:`register` (re-exported as
:func:`orxhestra.composer.register_builder`) at import time with an
async callable implementing the :class:`BuildFn` protocol.

The :class:`Helpers` bag threaded through every builder carries the
three cross-cutting resolvers (``resolve_tools``, ``resolve_model``,
``build_agent``) so builders don't have to know about the composer's
private state.

See Also
--------
orxhestra.composer.composer.Composer : Orchestrator that dispatches
    to the registered builders.
orxhestra.composer.builders.agents._common.build_composite : Shared
    composite-agent construction path (sequential / parallel / loop).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.builders.agents import a2a, llm, loop, parallel, react, sequential
from orxhestra.composer.schema import AgentDef, ComposeSpec, ModelConfig

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class BuildFn(Protocol):
    """Signature every composer agent builder implements.

    Defined as a :class:`Protocol` (rather than a bare
    :data:`typing.Callable`) so the keyword-only ``helpers`` kwarg is
    part of the type — type checkers catch a forgotten ``helpers``
    parameter instead of it working by accident at runtime.

    Implementors should write::

        async def build(
            name: str,
            agent_def: AgentDef,
            spec: ComposeSpec,
            *,
            helpers: Helpers,
        ) -> BaseAgent:
            ...

    See Also
    --------
    Helpers : Dependency bag the builder uses to resolve tools,
        models, and sub-agents.
    register : Register a builder under a YAML ``type:`` key.
    """

    async def __call__(
        self,
        name: str,
        agent_def: AgentDef,
        spec: ComposeSpec,
        *,
        helpers: Helpers,
    ) -> BaseAgent:
        ...


_REGISTRY: dict[str, BuildFn] = {}


def register(agent_type: str, fn: BuildFn) -> None:
    """Register a build function for a YAML agent type.

    Parameters
    ----------
    agent_type : str
        Key used in YAML ``type:`` field.
    fn : BuildFn
        Async callable implementing the :class:`BuildFn` protocol.
        Must take ``(name, agent_def, spec, *, helpers)`` and return
        an awaitable :class:`BaseAgent`.

    Example::

        async def build_custom(name, agent_def, spec, *, helpers):
            return MyCustomAgent(name=name, ...)

        register("custom", build_custom)
    """
    _REGISTRY[agent_type] = fn


def get(agent_type: str) -> BuildFn | None:
    """Look up a build function by agent type name.

    Parameters
    ----------
    agent_type : str
        The agent type to look up.

    Returns
    -------
    BuildFn | None
        The registered build function, or ``None`` if not found.
    """
    return _REGISTRY.get(agent_type)


def registered_types() -> list[str]:
    """Return all registered agent type names in sorted order.

    Used by the composer to enumerate legal ``type:`` values in
    error messages when a YAML spec declares an unknown agent type.

    Returns
    -------
    list[str]
        Sorted list of registered names (``["a2a", "llm", "loop",
        "parallel", "react", "sequential"]`` by default, plus any
        names added via :func:`register`).
    """
    return sorted(_REGISTRY)


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
        """Initialize helper dependencies for agent builders.

        Parameters
        ----------
        resolve_tools : Callable[[AgentDef], Awaitable[list[BaseTool]]]
            Resolve all tool references for an agent definition.
        resolve_model : Callable[[AgentDef], ModelConfig]
            Merge agent-level model config with defaults.
        build_agent : Callable[[str], Awaitable[BaseAgent]]
            Recursively build a sub-agent by name.
        """
        self.resolve_tools = resolve_tools
        self.resolve_model = resolve_model
        self.build_agent = build_agent


# Auto-register built-in agent builders.
register("llm", llm.build)
register("react", react.build)
register("sequential", sequential.build)
register("parallel", parallel.build)
register("loop", loop.build)
register("a2a", a2a.build)
