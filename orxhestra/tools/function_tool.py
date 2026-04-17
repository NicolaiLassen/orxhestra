"""FunctionTool - wrap any async Python function as a LangChain tool."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool


def function_tool(
    fn: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
) -> BaseTool:
    """Wrap an async Python function as a LangChain BaseTool.

    The function's type annotations become the tool's input schema.
    The docstring becomes the description if none is provided.

    Parameters
    ----------
    fn : callable
        The async function to wrap.
    name : str, optional
        Tool name. Defaults to fn.__name__.
    description : str, optional
        Tool description. Defaults to fn.__doc__.

    Returns
    -------
    BaseTool
        A LangChain BaseTool instance.

    See Also
    --------
    LongRunningFunctionTool : Similar wrapper for long-running
        operations that should stream progress events.
    AgentTool : Wrap a whole :class:`BaseAgent` as a tool instead.

    Examples
    --------
    >>> async def search(query: str) -> str:
    ...     \"\"\"Search the web for a query.\"\"\"
    ...     ...
    >>> search_tool = function_tool(search)
    """
    return StructuredTool.from_function(
        coroutine=fn,
        name=name or fn.__name__,
        description=description or (fn.__doc__ or "").strip(),
    )
