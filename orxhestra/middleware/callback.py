"""Adapter that wraps ``LlmAgentCallbacks`` as a Middleware.

This lets existing callback-based code compose cleanly with new
middleware without any migration.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from orxhestra.middleware.base import BaseMiddleware, ToolCall

if TYPE_CHECKING:
    from orxhestra.agents.callbacks import LlmAgentCallbacks
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse


class CallbackMiddleware(BaseMiddleware):
    """Wraps a :class:`LlmAgentCallbacks` instance as a middleware.

    Parameters
    ----------
    callbacks : LlmAgentCallbacks
        The callback bundle to adapt.

    Notes
    -----
    ``on_model_error`` is intentionally not exposed on the
    :class:`Middleware` protocol — error recovery is still the agent's
    responsibility and stays on the callback interface. This adapter
    only maps the four synchronous lifecycle hooks.

    See Also
    --------
    Middleware : The protocol this adapter implements.
    LlmAgentCallbacks : Source callback bundle.
    MiddlewareStack : Ordered execution of multiple middleware.
    """

    def __init__(self, callbacks: LlmAgentCallbacks) -> None:
        self._cb = callbacks

    async def before_model(
        self, ctx: InvocationContext, request: LlmRequest,
    ) -> LlmRequest:
        """Fire ``before_model`` callback if configured, return request."""
        if self._cb.before_model is not None:
            await self._cb.before_model(ctx, request)
        return request

    async def after_model(
        self, ctx: InvocationContext, response: LlmResponse,
    ) -> LlmResponse:
        """Fire ``after_model`` callback if configured, return response."""
        if self._cb.after_model is not None:
            await self._cb.after_model(ctx, response)
        return response

    async def wrap_tool(
        self,
        ctx: InvocationContext,
        call: ToolCall,
        call_next: Callable[[ToolCall], Awaitable[Any]],
    ) -> Any:
        """Run ``before_tool`` and ``after_tool`` callbacks around the call."""
        if self._cb.before_tool is not None:
            await self._cb.before_tool(ctx, call.name, call.args)
        result = await call_next(call)
        if self._cb.after_tool is not None:
            await self._cb.after_tool(ctx, call.name, result)
        return result
