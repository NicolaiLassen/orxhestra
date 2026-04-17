"""Middleware protocol for agent lifecycle interception."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse


@dataclass
class ToolCall:
    """A tool invocation intercepted by middleware.

    Attributes
    ----------
    name : str
        Tool name.
    args : dict
        Arguments passed to the tool.
    """

    name: str
    args: dict[str, Any]


@runtime_checkable
class Middleware(Protocol):
    """Composable interceptor for agent lifecycle events.

    All methods are optional — middleware implementations can inherit
    from :class:`BaseMiddleware` and override only the hooks they need.
    Default implementations are pass-throughs.

    Lifecycle order for a single invocation:

    1. ``before_invoke`` (outermost wraps innermost)
    2. Agent execution, which may involve:
       - ``before_model`` → LLM call → ``after_model`` (per iteration)
       - ``wrap_tool`` around each tool call
       - ``on_event`` for every event emitted
    3. ``after_invoke`` (innermost unwraps to outermost)

    See Also
    --------
    BaseMiddleware : Default pass-through base for new middleware.
    MiddlewareStack : Ordered dispatch over multiple middleware.
    CallbackMiddleware : Adapter wrapping :class:`LlmAgentCallbacks`.
    LoggingMiddleware : Simple example middleware.
    Runner : Hosts the stack at the invocation boundary.

    Examples
    --------
    >>> class TimingMiddleware(BaseMiddleware):
    ...     async def before_invoke(self, ctx):
    ...         self._started = time.monotonic()
    ...     async def after_invoke(self, ctx, error=None):
    ...         dur = time.monotonic() - self._started
    ...         print(f"{ctx.agent_name} took {dur:.2f}s")
    >>> runner = Runner(agent=my_agent, middleware=[TimingMiddleware()])
    """

    async def before_invoke(self, ctx: InvocationContext) -> None:
        """Called once before the agent begins an invocation.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        """
        ...

    async def after_invoke(
        self, ctx: InvocationContext, error: Exception | None = None,
    ) -> None:
        """Called once after the agent finishes.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        error : Exception, optional
            Exception raised by the invocation, if any. ``None`` on
            successful completion.
        """
        ...

    async def on_event(
        self, ctx: InvocationContext, event: Event,
    ) -> Event | None:
        """Transform or drop an event before it is yielded.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        event : Event
            The event emitted by the agent.

        Returns
        -------
        Event or None
            The (possibly transformed) event to yield, or ``None`` to
            drop this event from the stream.
        """
        ...

    async def before_model(
        self, ctx: InvocationContext, request: LlmRequest,
    ) -> LlmRequest:
        """Transform the LLM request before it is sent.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        request : LlmRequest
            The request about to be sent to the LLM.

        Returns
        -------
        LlmRequest
            The (possibly transformed) request.
        """
        ...

    async def after_model(
        self, ctx: InvocationContext, response: LlmResponse,
    ) -> LlmResponse:
        """Transform the LLM response after it is received.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        response : LlmResponse
            The response returned by the LLM.

        Returns
        -------
        LlmResponse
            The (possibly transformed) response.
        """
        ...

    async def wrap_tool(
        self,
        ctx: InvocationContext,
        call: ToolCall,
        call_next: Callable[[ToolCall], Awaitable[Any]],
    ) -> Any:
        """Wrap a tool invocation.

        Parameters
        ----------
        ctx : InvocationContext
            The context for this invocation.
        call : ToolCall
            The tool call being made.
        call_next : callable
            The next middleware or the tool executor. Must be awaited
            with a ``ToolCall`` and returns the tool's result.

        Returns
        -------
        Any
            The tool's result (possibly transformed).
        """
        ...


class BaseMiddleware:
    """Default middleware with pass-through implementations.

    Subclass this to override only the hooks you need. Using
    ``BaseMiddleware`` ensures forward compatibility if new hooks are
    added to the protocol.
    """

    async def before_invoke(self, ctx: InvocationContext) -> None:
        """Pass-through default. See :class:`Middleware`."""
        return None

    async def after_invoke(
        self, ctx: InvocationContext, error: Exception | None = None,
    ) -> None:
        """Pass-through default. See :class:`Middleware`."""
        return None

    async def on_event(
        self, ctx: InvocationContext, event: Event,
    ) -> Event | None:
        """Pass-through default. Returns ``event`` unchanged."""
        return event

    async def before_model(
        self, ctx: InvocationContext, request: LlmRequest,
    ) -> LlmRequest:
        """Pass-through default. Returns ``request`` unchanged."""
        return request

    async def after_model(
        self, ctx: InvocationContext, response: LlmResponse,
    ) -> LlmResponse:
        """Pass-through default. Returns ``response`` unchanged."""
        return response

    async def wrap_tool(
        self,
        ctx: InvocationContext,
        call: ToolCall,
        call_next: Callable[[ToolCall], Awaitable[Any]],
    ) -> Any:
        """Pass-through default. Invokes ``call_next(call)`` and returns it."""
        return await call_next(call)
