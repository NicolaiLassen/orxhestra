"""Tool builder registry — resolves YAML tool definitions to BaseTool instances.

Builtin tools are registered lazily on first access. Custom builtins can
be added via :func:`register_builtin`.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool

from orxhestra.composer.errors import ComposerError

if TYPE_CHECKING:
    from orxhestra.agents.base_agent import BaseAgent



def import_object(dotted_path: str) -> Any:
    """Import an object by its fully-qualified dotted path.

    Parameters
    ----------
    dotted_path : str
        Fully-qualified path, e.g. ``"my_module.MyClass"``.

    Returns
    -------
    Any
        The imported object.

    Raises
    ------
    ComposerError
        If the path is invalid or the module/attribute cannot be found.
    """
    module_path, _, attr_name = dotted_path.rpartition(".")
    if not module_path:
        msg = f"Invalid import path '{dotted_path}' — expected 'module.attribute'"
        raise ComposerError(msg)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        msg = (
            f"Module '{module_path}' not found. "
            f"If this is a local file, ensure it's in the same "
            f"directory as your orx.yaml."
        )
        raise ComposerError(msg) from None
    except Exception as exc:
        msg = f"Failed to import module '{module_path}': {exc}"
        raise ComposerError(msg) from None
    try:
        return getattr(module, attr_name)
    except AttributeError:
        msg = f"Module '{module_path}' has no attribute '{attr_name}'"
        raise ComposerError(msg) from None



_ToolResult = BaseTool | list[BaseTool]
_BUILTIN_REGISTRY: dict[str, Callable[[], _ToolResult]] = {}

#: Sync or async resolver that turns a ``custom:`` tool config into
#: one or more LangChain tools.  Registered via
#: :func:`register_tool_resolver` and consumed by
#: :func:`resolve_custom_tool`.
ToolResolver = Callable[[dict[str, Any]], _ToolResult | Awaitable[_ToolResult]]

_TOOL_RESOLVER_REGISTRY: dict[str, ToolResolver] = {}


def register_tool_resolver(tool_type: str, resolver: ToolResolver) -> None:
    """Register a resolver for a custom YAML ``tool.custom.type``.

    Third-party tool types extend the composer the same way agents
    and models are extended — plug a resolver into this registry and
    reference it from YAML via
    ``tools: { my_tool: { custom: { type: "<tool_type>", ... } } }``.

    Parameters
    ----------
    tool_type : str
        The value of ``custom.type`` that should route to this
        resolver.
    resolver : ToolResolver
        Sync or async callable that takes the full ``custom`` dict
        (including the ``type`` key) and returns a
        :class:`~langchain_core.tools.BaseTool` or a list of them.

    Examples
    --------
    >>> from orxhestra.composer import register_tool_resolver
    >>> from langchain_core.tools import Tool
    >>>
    >>> def make_webhook(config):
    ...     url = config["url"]
    ...     return Tool.from_function(
    ...         name=config.get("name", "webhook"),
    ...         description="POST to a webhook.",
    ...         func=lambda body: requests.post(url, json=body).text,
    ...     )
    >>>
    >>> register_tool_resolver("webhook", make_webhook)

    YAML::

        tools:
          notifier:
            custom:
              type: webhook
              url: https://example.com/notify
              name: notify
    """
    _TOOL_RESOLVER_REGISTRY[tool_type] = resolver


def registered_tool_resolvers() -> list[str]:
    """Return all registered custom tool-type names in sorted order.

    Used by the composer to cite legal ``custom.type`` values in
    error messages.

    Returns
    -------
    list[str]
    """
    return sorted(_TOOL_RESOLVER_REGISTRY)


async def resolve_custom_tool(config: dict[str, Any]) -> _ToolResult:
    """Dispatch a ``ToolDef.custom`` payload to its registered resolver.

    Parameters
    ----------
    config : dict[str, Any]
        The full ``custom`` dict.  Must include a ``type`` key
        matching an entry registered via
        :func:`register_tool_resolver`.

    Returns
    -------
    BaseTool or list[BaseTool]

    Raises
    ------
    ComposerError
        When the ``type`` key is missing or no resolver is registered
        for it.
    """
    tool_type = config.get("type")
    if not tool_type:
        raise ComposerError(
            "ToolDef.custom must include a 'type' key naming a "
            "resolver registered via register_tool_resolver()",
        )
    resolver = _TOOL_RESOLVER_REGISTRY.get(tool_type)
    if resolver is None:
        known = ", ".join(registered_tool_resolvers()) or "<none>"
        raise ComposerError(
            f"No custom tool resolver registered for type "
            f"{tool_type!r} (known: {known})",
        )
    result = resolver(config)
    if inspect.isawaitable(result):
        result = await result
    return result


def register_builtin(name: str, factory: Callable[[], _ToolResult]) -> None:
    """Register a builtin tool factory by name.

    The factory may return a single ``BaseTool`` or a list of them.

    Parameters
    ----------
    name : str
        Tool name used in YAML ``tools:`` lists.
    factory : Callable
        Zero-argument callable returning ``BaseTool`` or ``list[BaseTool]``.

    Example::

        register_builtin("my_tool", lambda: my_tool_instance)
    """
    _BUILTIN_REGISTRY[name] = factory


def _register_defaults() -> None:
    """Register the SDK's built-in tools (called once on first access)."""
    if "exit_loop" not in _BUILTIN_REGISTRY:

        def _exit_loop() -> BaseTool:
            from orxhestra.tools.exit_loop import exit_loop_tool

            return exit_loop_tool

        def _filesystem() -> list[BaseTool]:
            from orxhestra.tools.filesystem import make_filesystem_tools

            return make_filesystem_tools()

        def _shell() -> list[BaseTool]:
            from orxhestra.tools.shell import make_shell_tools

            return make_shell_tools()

        def _artifacts() -> list[BaseTool]:
            from orxhestra.tools.artifact_tools import make_artifact_tools

            return make_artifact_tools()

        def _human_input() -> BaseTool:
            from orxhestra.cli.tools.human_input import make_human_input_tool

            return make_human_input_tool()

        def _sleep() -> BaseTool:
            from orxhestra.tools.sleep_tool import make_sleep_tool

            return make_sleep_tool()

        def _memory() -> list[BaseTool]:
            import os

            from orxhestra.memory.file_memory_service import get_memory_dir
            from orxhestra.tools.memory_tools import make_memory_tools

            ws = os.environ.get("AGENT_WORKSPACE", os.getcwd())
            return make_memory_tools(get_memory_dir(ws))

        _BUILTIN_REGISTRY["exit_loop"] = _exit_loop
        _BUILTIN_REGISTRY["memory"] = _memory
        _BUILTIN_REGISTRY["filesystem"] = _filesystem
        _BUILTIN_REGISTRY["shell"] = _shell
        _BUILTIN_REGISTRY["artifacts"] = _artifacts
        _BUILTIN_REGISTRY["sleep"] = _sleep
        _BUILTIN_REGISTRY["human_input"] = _human_input




def resolve_builtin(name: str) -> _ToolResult:
    """Look up a builtin tool by name.

    Parameters
    ----------
    name : str
        Registered builtin tool name.

    Returns
    -------
    BaseTool | list[BaseTool]
        The resolved tool(s).

    Raises
    ------
    ComposerError
        If the name is not registered.
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
    """Import a Python callable and wrap it as a LangChain tool.

    Parameters
    ----------
    path : str
        Dotted import path to the callable.
    name : str | None
        Optional override for the tool name.
    description : str | None
        Optional override for the tool description.

    Returns
    -------
    BaseTool
        Wrapped LangChain tool.
    """
    from orxhestra.tools.function_tool import function_tool

    fn = import_object(path)

    # Already a LangChain tool (e.g. decorated with @tool) — use directly
    if isinstance(fn, BaseTool):
        if name is not None:
            fn.name = name
        if description is not None:
            fn.description = description
        return fn

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
    """Connect to an MCP server and load its tools.

    Parameters
    ----------
    url : str | None
        MCP server URL.
    server_path : str | None
        Dotted import path to a local MCP server object.

    Returns
    -------
    list[BaseTool]
        Tools loaded from the MCP server.
    """
    from orxhestra.integrations.mcp import MCPClient, MCPToolAdapter

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
    """Wrap an agent as an ``AgentTool``.

    Parameters
    ----------
    agent : BaseAgent
        The agent to wrap.
    skip_summarization : bool
        If ``True``, skip LLM summarization of the agent's output.

    Returns
    -------
    BaseTool
        An ``AgentTool`` wrapping the agent.
    """
    from orxhestra.tools.agent_tool import AgentTool

    return AgentTool(agent, skip_summarization=skip_summarization)


def resolve_transfer(target_agents: list[BaseAgent]) -> BaseTool:
    """Create a ``transfer_to_agent`` tool for the given targets.

    Parameters
    ----------
    target_agents : list[BaseAgent]
        Agents that can be transferred to.

    Returns
    -------
    BaseTool
        A transfer tool bound to the target agents.
    """
    from orxhestra.tools.transfer_tool import make_transfer_tool

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

    Parameters
    ----------
    name : str
        Skill name used in the resource URI.
    description : str
        Human-readable skill description.
    url : str | None
        MCP server URL.
    server_path : str | None
        Dotted import path to a local MCP server object.

    Returns
    -------
    Skill
        Populated skill instance.
    """
    from orxhestra.integrations.mcp import MCPClient
    from orxhestra.skills import Skill

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
