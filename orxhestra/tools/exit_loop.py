"""exit_loop - signal a LoopAgent to stop iterating.

Call this tool from within a sub-agent when the goal assigned by the
``LoopAgent`` has been completed. The agent intercepts the sentinel
return value and emits an event with ``actions.escalate = True``, which
causes the ``LoopAgent`` to terminate the loop.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

# Sentinel string returned by exit_loop; LlmAgent intercepts this to set
# EventActions.escalate = True on the emitted TOOL_RESPONSE event.
EXIT_LOOP_SENTINEL = "__EXIT_LOOP__"


def make_exit_loop_tool() -> BaseTool:
    """Create the ``exit_loop`` tool.

    Returns
    -------
    BaseTool
        A LangChain tool that, when invoked, signals the parent
        :class:`LoopAgent` to stop iterating.

    See Also
    --------
    LoopAgent : The agent whose iteration this tool terminates.
    EventActions.escalate : The flag this tool triggers.
    exit_loop_tool : Pre-built module-level singleton.

    Examples
    --------
    >>> from orxhestra import LoopAgent, LlmAgent, exit_loop_tool
    >>> writer = LlmAgent(name="writer", model=m, tools=[exit_loop_tool])
    >>> loop = LoopAgent(name="retry_writer", agents=[writer])
    """

    async def exit_loop() -> str:
        """Exit the current loop. Call this when the loop goal is complete."""
        return EXIT_LOOP_SENTINEL

    return StructuredTool.from_function(
        coroutine=exit_loop,
        name="exit_loop",
        description=(
            "Exit the current loop. Call this tool when you have completed "
            "the goal assigned by the loop and are ready to stop iterating."
        ),
    )


#: Module-level singleton - import and pass directly to ``LlmAgent``.
exit_loop_tool = make_exit_loop_tool()
