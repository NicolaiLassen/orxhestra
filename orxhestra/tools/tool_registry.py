"""Tool registry - discover and manage tools by name.

Tools can be registered via the @register_tool decorator or by calling
tool_registry.register() directly. The registry holds a global singleton
but you can also instantiate ToolRegistry() for isolated contexts.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool


class ToolRegistry:
    """Central registry for LangChain tools.

    Provides name-based lookup and listing of all registered tools.
    Use the module-level :data:`tool_registry` singleton, or create a
    local registry for test isolation.

    See Also
    --------
    register_tool : Decorator that auto-registers a tool on import.
    tool_registry : Module-level singleton.
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by name.

        Parameters
        ----------
        tool : BaseTool
            The tool to register.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Use a unique name or deregister the existing tool first."
            )
        self._tools[tool.name] = tool

    def deregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Parameters
        ----------
        name : str
            The name of the tool to remove.
        """
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Return a tool by name, or None if not found.

        Parameters
        ----------
        name : str
            The tool name to look up.

        Returns
        -------
        BaseTool or None
            The matching tool, or None if not registered.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools.

        Returns
        -------
        list[BaseTool]
            All currently registered tools.
        """
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        """Check whether a tool with the given name is registered."""
        return name in self._tools


# Module-level singleton
tool_registry = ToolRegistry()


def register_tool(tool: BaseTool) -> BaseTool:
    """Decorator / helper to register a LangChain tool in the global registry.

    Parameters
    ----------
    tool : BaseTool
        The tool instance to register.

    Returns
    -------
    BaseTool
        The same tool, unchanged (allows use as a decorator).

    Examples
    --------
    >>> @register_tool
    ... @tool
    ... def my_tool(x: str) -> str: ...

    >>> register_tool(some_tool_instance)
    """
    tool_registry.register(tool)
    return tool
