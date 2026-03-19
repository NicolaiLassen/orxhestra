"""Tool builder registry — resolves YAML tool definitions to BaseTool instances.

Builtin tools are registered lazily on first access. Custom builtins can
be added via :func:`register_builtin`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from langchain_adk.composer.errors import ComposerError

if TYPE_CHECKING:
    from langchain_adk.agents.base_agent import BaseAgent
    from langchain_adk.skills.skill_store import BaseSkillStore

# ---------------------------------------------------------------------------
# Generic import helper
# ---------------------------------------------------------------------------


def import_object(dotted_path: str) -> Any:
    """Import an object by its fully-qualified dotted path."""
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        msg = f"Invalid import path: {dotted_path}"
        raise ComposerError(msg)
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr_name)
    except AttributeError:
        msg = f"Module '{module_path}' has no attribute '{attr_name}'"
        raise ComposerError(msg) from None


# ---------------------------------------------------------------------------
# Builtin tool registry
# ---------------------------------------------------------------------------

_BUILTIN_REGISTRY: dict[str, Callable[[], BaseTool]] = {}
_SKILL_STORE: BaseSkillStore | None = None


def register_builtin(name: str, factory: Callable[[], BaseTool]) -> None:
    """Register a builtin tool factory by name.

    Example::

        register_builtin("my_tool", lambda: my_tool_instance)
    """
    _BUILTIN_REGISTRY[name] = factory


def set_skill_store(store: BaseSkillStore) -> None:
    """Set the skill store for ``list_skills`` / ``load_skill`` builtins."""
    global _SKILL_STORE
    _SKILL_STORE = store
    _register_skill_builtins()


def _register_skill_builtins() -> None:
    """Register list_skills and load_skill using the current skill store."""
    if _SKILL_STORE is None:
        return
    store = _SKILL_STORE

    def _list_skills() -> BaseTool:
        from langchain_adk.skills import make_list_skills_tool

        return make_list_skills_tool(store)

    def _load_skill() -> BaseTool:
        from langchain_adk.skills import make_load_skill_tool

        return make_load_skill_tool(store)

    _BUILTIN_REGISTRY["list_skills"] = _list_skills
    _BUILTIN_REGISTRY["load_skill"] = _load_skill


def _register_defaults() -> None:
    """Register the SDK's built-in tools (called once on first access)."""
    if "exit_loop" not in _BUILTIN_REGISTRY:

        def _exit_loop() -> BaseTool:
            from langchain_adk.tools.exit_loop import exit_loop_tool

            return exit_loop_tool

        _BUILTIN_REGISTRY["exit_loop"] = _exit_loop
    _register_skill_builtins()


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def resolve_builtin(name: str) -> BaseTool:
    """Look up a builtin tool by name."""
    _register_defaults()
    factory = _BUILTIN_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(_BUILTIN_REGISTRY)) or "(none)"
        msg = f"Unknown builtin tool: '{name}'. Available: {available}"
        raise ComposerError(msg)
    return factory()


def resolve_function(
    path: str,
    name: str | None = None,
    description: str | None = None,
) -> BaseTool:
    """Import a Python callable and wrap it as a LangChain tool."""
    from langchain_adk.tools.function_tool import function_tool

    fn = import_object(path)
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if description is not None:
        kwargs["description"] = description
    return function_tool(fn, **kwargs)


async def resolve_mcp(
    url: str | None = None,
    server_path: str | None = None,
) -> list[BaseTool]:
    """Connect to an MCP server and load its tools."""
    from langchain_adk.integrations.mcp import MCPClient, MCPToolAdapter

    if url:
        client = MCPClient(url)
    elif server_path:
        server_obj = import_object(server_path)
        client = MCPClient(server_obj)
    else:
        msg = "MCP config must have 'url' or 'server'"
        raise ComposerError(msg)
    adapter = MCPToolAdapter(client)
    return await adapter.load_tools()


def resolve_agent_tool(
    agent: BaseAgent,
    skip_summarization: bool = False,
) -> BaseTool:
    """Wrap an agent as an ``AgentTool``."""
    from langchain_adk.tools.agent_tool import AgentTool

    return AgentTool(agent, skip_summarization=skip_summarization)


def resolve_transfer(target_agents: list[BaseAgent]) -> BaseTool:
    """Create a ``transfer_to_agent`` tool for the given targets."""
    from langchain_adk.tools.transfer_tool import make_transfer_tool

    return make_transfer_tool(target_agents)
