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

_ToolResult = BaseTool | list[BaseTool]
_BUILTIN_REGISTRY: dict[str, Callable[[], _ToolResult]] = {}


def register_builtin(name: str, factory: Callable[[], _ToolResult]) -> None:
    """Register a builtin tool factory by name.

    The factory may return a single ``BaseTool`` or a list of them.

    Example::

        register_builtin("my_tool", lambda: my_tool_instance)
    """
    _BUILTIN_REGISTRY[name] = factory


def _register_defaults() -> None:
    """Register the SDK's built-in tools (called once on first access)."""
    if "exit_loop" not in _BUILTIN_REGISTRY:

        def _exit_loop() -> BaseTool:
            from langchain_adk.tools.exit_loop import exit_loop_tool

            return exit_loop_tool

        def _filesystem() -> list[BaseTool]:
            from langchain_adk.tools.filesystem import make_filesystem_tools

            return make_filesystem_tools()

        def _shell() -> list[BaseTool]:
            from langchain_adk.tools.shell import make_shell_tools

            return make_shell_tools()

        _BUILTIN_REGISTRY["exit_loop"] = _exit_loop
        _BUILTIN_REGISTRY["filesystem"] = _filesystem
        _BUILTIN_REGISTRY["shell"] = _shell


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def resolve_builtin(name: str) -> _ToolResult:
    """Look up a builtin tool by name.

    Returns a single ``BaseTool`` or a list of them.
    """
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


async def resolve_mcp_skill(
    name: str,
    description: str,
    url: str | None = None,
    server_path: str | None = None,
) -> Any:
    """Fetch a skill's content from a FastMCP server.

    Reads the ``skill://{name}/SKILL.md`` resource and returns a ``Skill``
    object populated with the remote content.
    """
    from langchain_adk.integrations.mcp import MCPClient
    from langchain_adk.skills import Skill

    if url:
        client = MCPClient(url)
    elif server_path:
        server_obj = import_object(server_path)
        client = MCPClient(server_obj)
    else:
        msg = "MCP skill config must have 'url' or 'server'"
        raise ComposerError(msg)

    resource = await client.read_resource(f"skill://{name}/SKILL.md")
    # FastMCP read_resource returns content; extract text
    if hasattr(resource, "content"):
        content = resource.content
    elif isinstance(resource, list):
        content = "\n".join(
            item.text if hasattr(item, "text") else str(item)
            for item in resource
        )
    else:
        content = str(resource)

    return Skill(name=name, description=description, content=content)
