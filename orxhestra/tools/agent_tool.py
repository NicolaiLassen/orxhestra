"""AgentTool - wrap a BaseAgent as a callable LangChain tool.

Enables an LlmAgent to delegate work to a sub-agent by calling it as a
tool. The parent agent sends a request string; the child agent runs and
its final answer is returned as the tool result.

Sub-agent events are pushed to the parent in real-time via
``ctx.event_callback`` (if set), so they appear in the parent's event
stream as the child is still running.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from orxhestra.agents.invocation_context import InvocationContext as Context  # noqa: TC001

if TYPE_CHECKING:
    from orxhestra.agents.base_agent import BaseAgent
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event


class AgentToolInput(BaseModel):
    """Input schema for AgentTool.

    Attributes
    ----------
    request : str
        The request or task to send to the agent.
    """
    request: str = Field(description="The request or task to send to the agent.")


class AgentTool(BaseTool):
    """Wraps a BaseAgent as a LangChain tool.

    When invoked, runs the wrapped agent with the given request and
    returns the final answer text as the tool result. Sub-agent events
    are pushed via ``ctx.event_callback`` so the parent can yield them
    in real-time.

    Parameters
    ----------
    agent : BaseAgent
        The agent to wrap.
    skip_summarization : bool
        If True, signals the parent to skip LLM summarization of this
        tool's result.
    output_limit : int, optional
        Maximum number of characters in the returned answer. If the
        sub-agent's response exceeds this limit, it is truncated at
        a newline boundary using ``truncate_output()``. ``None``
        (default) means no limit.
    before_agent_callback : callable, optional
        Called with ``(event, child_ctx)`` for each child event before
        it is processed. Return a string to short-circuit and use it
        as the tool result, or ``None`` to continue. Supports both
        sync and async callables.
    after_agent_callback : callable, optional
        Called with ``(event, child_ctx)`` after the child agent
        finishes streaming all events. Supports both sync and async
        callables.
    """

    name: str
    description: str
    args_schema: type[BaseModel] = AgentToolInput
    skip_summarization: bool = False

    # Injected at runtime by LlmAgent before tool execution
    _ctx: Any | None = None

    def __init__(
        self,
        agent: BaseAgent,
        *,
        skip_summarization: bool = False,
        output_limit: int | None = None,
        before_agent_callback: (
            Callable[[Event, Context], str | None | Awaitable[str | None]] | None
        ) = None,
        after_agent_callback: (
            Callable[[Event, Context], Awaitable[None] | None] | None
        ) = None,
    ) -> None:
        super().__init__(
            name=agent.name,
            description=agent.description or f"Delegate work to the {agent.name} agent.",
        )
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "skip_summarization", skip_summarization)
        object.__setattr__(self, "_output_limit", output_limit)
        object.__setattr__(self, "_before_agent_callback", before_agent_callback)
        object.__setattr__(self, "_after_agent_callback", after_agent_callback)
        object.__setattr__(self, "_ctx", None)

    def inject_context(self, ctx: InvocationContext) -> None:
        """Inject the parent invocation context before tool execution.

        Called by :class:`LlmAgent` just before the tool runs so the
        wrapped sub-agent can derive a child context that preserves
        branch attribution and session identity.

        Parameters
        ----------
        ctx : InvocationContext
            The parent agent's invocation context.
        """
        object.__setattr__(self, "_ctx", ctx)

    def _run(self, request: str, **kwargs: Any) -> str:
        """Raise an error - AgentTool is async-only."""
        raise NotImplementedError("Use async ainvoke - AgentTool is async-only.")

    async def _arun(
        self,
        request: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Run the wrapped agent asynchronously and return its final answer.

        Sub-agent events are pushed to ``ctx.event_callback`` as they
        arrive, so the parent stream sees them in real-time.

        Parameters
        ----------
        request : str
            The request or task to send to the wrapped agent.
        run_manager : AsyncCallbackManagerForToolRun, optional
            LangChain callback manager (injected by the framework).

        Returns
        -------
        str
            The wrapped agent's final answer text, or an error message
            if no final answer was produced.
        """
        import asyncio

        agent: BaseAgent = object.__getattribute__(self, "_agent")  # type: ignore[assignment]
        ctx: InvocationContext | None = object.__getattribute__(self, "_ctx")
        before_cb = object.__getattribute__(self, "_before_agent_callback")
        after_cb = object.__getattribute__(self, "_after_agent_callback")

        if ctx is None:
            raise RuntimeError(
                f"AgentTool '{self.name}' has no context. "
                "Call inject_context(ctx) before invoking."
            )

        child_ctx = ctx.derive(agent_name=agent.name).clear_session()
        final_answer: str | None = None
        async for event in agent.astream(request, ctx=child_ctx):
            if ctx.event_callback is not None:
                ctx.event_callback(event)
            if before_cb is not None:
                result = before_cb(event, child_ctx)
                if asyncio.iscoroutine(result):
                    result = await result
                if result is not None:
                    return result
            if event.is_final_response():
                final_answer = event.text

        if after_cb is not None:
            last_event = event if event is not None else None
            result = after_cb(last_event, child_ctx)
            if asyncio.iscoroutine(result):
                await result

        answer = final_answer or f"Agent '{agent.name}' produced no final answer."

        # Truncate if output_limit is set
        output_limit: int | None = object.__getattribute__(self, "_output_limit")
        if output_limit is not None:
            from orxhestra.tools.output import truncate_output

            answer = truncate_output(answer, output_limit)

        return answer
