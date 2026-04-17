"""Middleware stack executor — onion-pattern composition."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from orxhestra.middleware.base import BaseMiddleware, Middleware, ToolCall

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse


_EMPTY: tuple[Middleware, ...] = ()


class MiddlewareStack:
    """Ordered stack of middleware with composed lifecycle dispatch.

    First middleware in the list is the outermost layer. For event
    transforms, a ``None`` return drops the event. For request/response
    transforms, the value is threaded through each middleware in order.
    For ``wrap_tool``, the stack composes so outer middleware wraps
    inner middleware.

    Parameters
    ----------
    middleware : list[Middleware]
        :class:`Middleware` instances to run, in outer-to-inner order.

    Notes
    -----
    Empty stacks are zero-overhead: every hook short-circuits when the
    middleware tuple is empty. Partial implementations are also
    supported — missing hooks fall back to :class:`BaseMiddleware`'s
    pass-through defaults.

    See Also
    --------
    Middleware : Protocol each entry must satisfy.
    BaseMiddleware : Default pass-through implementation.
    Runner : Hosts a stack at the invocation boundary.
    """

    def __init__(self, middleware: list[Middleware] | None = None) -> None:
        self._stack: tuple[Middleware, ...] = (
            tuple(middleware) if middleware else _EMPTY
        )

    def __bool__(self) -> bool:
        """True if the stack contains at least one middleware."""
        return bool(self._stack)

    def __len__(self) -> int:
        """Number of middleware in the stack."""
        return len(self._stack)

    def __iter__(self):
        """Iterate middleware in registered (outer-to-inner) order."""
        return iter(self._stack)

    def extend(self, more: list[Middleware]) -> MiddlewareStack:
        """Return a new stack with additional middleware appended.

        Parameters
        ----------
        more : list[Middleware]
            Middleware to add after the existing ones.

        Returns
        -------
        MiddlewareStack
            A new stack instance. The original is not modified.
        """
        if not more:
            return self
        return MiddlewareStack([*self._stack, *more])

    async def before_invoke(self, ctx: InvocationContext) -> None:
        """Fire ``before_invoke`` on every middleware in order.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        """
        for m in self._stack:
            await _call(m, "before_invoke", ctx)

    async def after_invoke(
        self, ctx: InvocationContext, error: Exception | None = None,
    ) -> None:
        """Fire ``after_invoke`` on every middleware in reverse order.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        error : Exception, optional
            Exception raised, if any.
        """
        # Reverse order: innermost unwinds first.
        for m in reversed(self._stack):
            await _call(m, "after_invoke", ctx, error)

    async def on_event(
        self, ctx: InvocationContext, event: Event,
    ) -> Event | None:
        """Thread an event through every middleware's ``on_event``.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        event : Event
            The event to transform.

        Returns
        -------
        Event or None
            The final event, or ``None`` if any middleware dropped it.
        """
        current: Event | None = event
        for m in self._stack:
            if current is None:
                return None
            result = await _call(m, "on_event", ctx, current)
            current = result
        return current

    async def before_model(
        self, ctx: InvocationContext, request: LlmRequest,
    ) -> LlmRequest:
        """Thread an LLM request through every middleware's ``before_model``.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        request : LlmRequest
            The request to transform.

        Returns
        -------
        LlmRequest
            The final (possibly transformed) request.
        """
        current = request
        for m in self._stack:
            result = await _call(m, "before_model", ctx, current)
            if result is not None:
                current = result
        return current

    async def after_model(
        self, ctx: InvocationContext, response: LlmResponse,
    ) -> LlmResponse:
        """Thread an LLM response through every middleware's ``after_model``.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        response : LlmResponse
            The response to transform.

        Returns
        -------
        LlmResponse
            The final (possibly transformed) response.
        """
        current = response
        for m in self._stack:
            result = await _call(m, "after_model", ctx, current)
            if result is not None:
                current = result
        return current

    async def wrap_tool(
        self,
        ctx: InvocationContext,
        call: ToolCall,
        executor: Callable[[ToolCall], Awaitable[Any]],
    ) -> Any:
        """Execute ``executor(call)`` wrapped by all middleware.

        Middleware are composed so the first in the stack is the
        outermost wrapper and ``executor`` is the innermost callable.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        call : ToolCall
            The tool call to wrap.
        executor : callable
            The actual tool implementation. Called last.

        Returns
        -------
        Any
            The tool's result (possibly transformed by middleware).
        """
        if not self._stack:
            return await executor(call)

        async def _wrap_index(i: int, c: ToolCall) -> Any:
            if i == len(self._stack):
                return await executor(c)
            m = self._stack[i]

            async def _next(next_call: ToolCall) -> Any:
                return await _wrap_index(i + 1, next_call)

            fn = getattr(m, "wrap_tool", None)
            if fn is None:
                return await _next(c)
            return await fn(ctx, c, _next)

        return await _wrap_index(0, call)


async def _call(
    m: Middleware,
    method: str,
    *args: Any,
) -> Any:
    """Call a middleware hook if defined, else use the base default.

    Parameters
    ----------
    m : Middleware
        The middleware instance.
    method : str
        The hook name to invoke.
    *args : Any
        Arguments forwarded to the hook.

    Returns
    -------
    Any
        The hook's return value, or the :class:`BaseMiddleware`
        pass-through default if ``m`` has not defined the method.
    """
    fn = getattr(m, method, None)
    if fn is None:
        # Fall through to BaseMiddleware's pass-through default.
        fn = getattr(BaseMiddleware, method)
        return await fn(None, *args)  # type: ignore[arg-type]
    return await fn(*args)
