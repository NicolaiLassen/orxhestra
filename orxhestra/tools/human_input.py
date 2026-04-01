"""Human input tool — lets agents request clarification from the human.

The tool accepts a ``question`` string and returns the user's typed
response.  An async callback must be injected by the CLI (or any
other host) before the tool can be used.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


def make_human_input_tool(
    input_callback: Callable[[str], Awaitable[str]] | None = None,
) -> BaseTool:
    """Create a ``human_input`` tool.

    Parameters
    ----------
    input_callback : callable, optional
        Async function that presents a question to the user and returns
        their answer.  If not provided, callers must set the callback
        later via ``tool.set_callback(cb)``.

    Returns
    -------
    BaseTool
        A LangChain tool named ``human_input``.
    """
    _holder: dict[str, Any] = {"cb": input_callback}

    def _set_callback(cb: Callable[[str], Awaitable[str]]) -> None:
        _holder["cb"] = cb

    async def human_input(question: str) -> str:
        """Ask the user a question and wait for their response.

        Use this when you need clarification, confirmation, or additional
        information from the user before proceeding.

        Args:
            question: The question to present to the user.
        """
        cb = _holder["cb"]
        if cb is None:
            return "(human_input is not available in this environment)"
        return await cb(question)

    tool: BaseTool = StructuredTool.from_function(
        coroutine=human_input,
        name="human_input",
        description=(
            "Ask the user a clarifying question. Use when you need "
            "more information, confirmation, or a choice before proceeding."
        ),
    )
    object.__setattr__(tool, "set_callback", _set_callback)
    return tool
