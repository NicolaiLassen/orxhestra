"""LongRunningFunctionTool - wraps a function that may take significant time.

Marks the tool as long-running so the framework knows not to re-invoke it
while it's still pending. Appends an instruction to the tool description
telling the LLM not to call it again if it already returned a pending status.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool


class LongRunningFunctionTool:
    """A function tool for long-running operations.

    Wraps an async function and marks it as long-running. The
    framework uses ``is_long_running`` to handle the tool
    differently — for example, by not re-invoking it while a previous
    call is still pending.

    The tool description is automatically appended with an instruction
    telling the LLM not to call the tool again if it has already
    returned an intermediate or pending status.

    Parameters
    ----------
    func : callable
        An async function to wrap as a long-running tool.
    name : str, optional
        Tool name. Defaults to the function name.
    description : str, optional
        Tool description. Defaults to the function docstring.

    See Also
    --------
    function_tool : Wrapper for short, synchronous-feeling tools.
    AgentTool : Wrap a whole agent as a tool.

    Examples
    --------
    >>> async def deploy_service(env: str) -> str:
    ...     '''Deploy the service to the given environment.'''
    ...     await some_long_operation(env)
    ...     return f"Deployed to {env}"
    >>> tool = LongRunningFunctionTool(deploy_service)
    >>> agent = LlmAgent(name="deployer", model=model, tools=[tool.as_tool()])
    """

    _LONG_RUNNING_NOTE = (
        "\n\nNOTE: This is a long-running operation. Do not call this tool"
        " again if it has already returned some intermediate or pending"
        " status."
    )

    def __init__(
        self,
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._func = func
        self._name = name or func.__name__
        base_desc = description or (func.__doc__ or "").strip()
        self._description = base_desc + self._LONG_RUNNING_NOTE
        self.is_long_running = True

    def as_tool(self) -> BaseTool:
        """Convert to a LangChain ``BaseTool`` for use in an agent.

        Returns
        -------
        BaseTool
            A LangChain StructuredTool wrapping the long-running function.
        """
        return StructuredTool.from_function(
            coroutine=self._func,
            name=self._name,
            description=self._description,
        )
