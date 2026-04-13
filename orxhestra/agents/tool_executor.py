"""ToolExecutor — runs tool calls and builds response events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from orxhestra.concurrency import gather_with_event_queue
from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.models.part import Content, ToolCallPart, ToolResponsePart
from orxhestra.tools.exit_loop import EXIT_LOOP_SENTINEL
from orxhestra.tools.transfer_tool import TRANSFER_SENTINEL

if TYPE_CHECKING:
    from orxhestra.agents.callbacks import LlmAgentCallbacks
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.models.llm_response import LlmResponse


class ToolExecutor:
    """Executes LLM tool calls in parallel and yields events.

    Handles tool lookup, context injection, before/after callbacks,
    sentinel interception (transfer and exit-loop), and concurrent
    execution with real-time event streaming.

    Parameters
    ----------
    tools : dict[str, BaseTool]
        Name-keyed tool registry.
    callbacks : LlmAgentCallbacks
        Lifecycle callbacks for before/after tool execution.
    emit_event : callable
        The ``BaseAgent._emit_event`` method (or equivalent) used to
        construct ``Event`` objects.
    """

    def __init__(
        self,
        tools: dict[str, BaseTool],
        callbacks: LlmAgentCallbacks,
        emit_event: Callable[..., Event],
    ) -> None:
        self._tools = tools
        self._callbacks = callbacks
        self._emit_event = emit_event

    def _tool_metadata(self, tool_name: str) -> dict[str, bool]:
        """Build metadata dict for a tool call event.

        Checks for ``interactive`` and ``require_confirmation`` attributes
        on the tool instance.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            return {}
        meta: dict[str, bool] = {}
        if getattr(tool, "interactive", False):
            meta["interactive"] = True
        if getattr(tool, "require_confirmation", False):
            meta["require_confirmation"] = True
        return meta

    async def execute(
        self,
        ctx: InvocationContext,
        llm_response: LlmResponse,
    ) -> AsyncIterator[Event | tuple[Event, ToolMessage]]:
        """Execute tool calls in parallel, yielding events as they complete.

        Yields
        ------
        Event
            The initial tool call event and intermediate child events.
        tuple[Event, ToolMessage]
            Tool response event paired with its ``ToolMessage`` for
            appending to the conversation history.
        """
        event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        tool_ctx: InvocationContext = ctx.model_copy(
            update={"event_callback": event_queue.put_nowait}
        )

        # Yield the tool call event.
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content(parts=[
                ToolCallPart(
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    args=tc["args"],
                    metadata=self._tool_metadata(tc["name"]),
                )
                for tc in llm_response.tool_calls
            ]),
            llm_response=llm_response,
        )

        async def _execute_one(tool_call: dict[str, Any]) -> tuple[Event, ToolMessage]:
            """Execute a single tool call and return response event + message."""
            t_name: str = tool_call["name"]
            t_args: dict[str, Any] = tool_call["args"]
            t_id: str = tool_call["id"]

            tool: BaseTool | None = self._tools.get(t_name)
            if tool is None:
                return self._build_response(
                    ctx, t_id, t_name,
                    error=f"Tool '{t_name}' not found. "
                    f"Available: {list(self._tools)}",
                )

            if hasattr(tool, "inject_context"):
                tool.inject_context(tool_ctx)

            tool_succeeded = False
            try:
                if self._callbacks.before_tool:
                    await self._callbacks.before_tool(ctx, t_name, t_args)
                result = await tool.ainvoke(t_args, config=ctx.run_config)
                tool_succeeded = True
                if self._callbacks.after_tool:
                    try:
                        await self._callbacks.after_tool(ctx, t_name, result)
                    except Exception:
                        pass  # Don't let callback errors affect the tool result.
                return self._build_response(ctx, t_id, t_name, result=str(result))
            except Exception as exc:
                if self._callbacks.after_tool and not tool_succeeded:
                    try:
                        await self._callbacks.after_tool(ctx, t_name, None)
                    except Exception:
                        pass
                # Only suppress ToolMessage when the tool itself was blocked
                # by a permission denial (before_tool raised).  If the tool
                # succeeded, always return a ToolMessage so the API sees a
                # matching response for every tool_call_id.
                is_pre_tool_denial = (
                    not tool_succeeded
                    and "PermissionDenied" in type(exc).__name__
                )
                event, msg = self._build_response(
                    ctx, t_id, t_name, error=str(exc),
                )
                return (event, None) if is_pre_tool_denial else (event, msg)

        # Run all tool calls concurrently, streaming intermediate events.
        async for item in gather_with_event_queue(
            [_execute_one(tc) for tc in llm_response.tool_calls],
            event_queue,
        ):
            yield item

    def _build_response(
        self,
        ctx: InvocationContext,
        t_id: str,
        t_name: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> tuple[Event, ToolMessage]:
        """Build a tool response event and ``ToolMessage`` pair.

        Parameters
        ----------
        ctx : InvocationContext
            Current invocation context.
        t_id : str
            Tool call identifier.
        t_name : str
            Tool name.
        result : str, optional
            Successful tool result.
        error : str, optional
            Error message from tool execution.

        Returns
        -------
        tuple[Event, ToolMessage]
            The response event and corresponding LangChain message.
        """
        part = ToolResponsePart(
            tool_call_id=t_id,
            tool_name=t_name,
            result=result or "",
            error=error,
        )
        actions = EventActions()
        content_str: str = result or error or ""
        if result and result.startswith(TRANSFER_SENTINEL):
            actions = EventActions(
                transfer_to_agent=result.removeprefix(TRANSFER_SENTINEL).strip()
            )
        elif result == EXIT_LOOP_SENTINEL:
            actions = EventActions(escalate=True)
        event: Event = self._emit_event(
            ctx,
            EventType.TOOL_RESPONSE,
            content=Content(parts=[part]),
            actions=actions,
        )
        msg_kwargs: dict[str, str] = {"status": "error"} if error else {}
        msg = ToolMessage(
            content=content_str,
            tool_call_id=t_id,
            **msg_kwargs,
        )
        return (event, msg)
