"""Simple logging middleware — example + smoke test target."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from orxhestra.middleware.base import BaseMiddleware, ToolCall

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event


_log = logging.getLogger("orxhestra.middleware.logging")


class LoggingMiddleware(BaseMiddleware):
    """Logs every invocation, event, and tool call at DEBUG level.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger to write to. Defaults to
        ``orxhestra.middleware.logging``.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._log = logger or _log

    async def before_invoke(self, ctx: InvocationContext) -> None:
        """Log the start of an invocation."""
        self._log.debug(
            "invoke start agent=%s session=%s",
            ctx.agent_name, ctx.session_id,
        )

    async def after_invoke(
        self, ctx: InvocationContext, error: Exception | None = None,
    ) -> None:
        """Log the end of an invocation, including errors."""
        if error is not None:
            self._log.debug(
                "invoke error agent=%s session=%s error=%s",
                ctx.agent_name, ctx.session_id, error,
            )
        else:
            self._log.debug(
                "invoke end agent=%s session=%s",
                ctx.agent_name, ctx.session_id,
            )

    async def on_event(
        self, ctx: InvocationContext, event: Event,
    ) -> Event | None:
        """Log each event and pass it through unchanged."""
        self._log.debug(
            "event agent=%s type=%s partial=%s",
            ctx.agent_name, event.type, event.partial,
        )
        return event

    async def wrap_tool(
        self,
        ctx: InvocationContext,
        call: ToolCall,
        call_next: Callable[[ToolCall], Awaitable[Any]],
    ) -> Any:
        """Log the tool call, then invoke ``call_next``."""
        self._log.debug(
            "tool call agent=%s tool=%s",
            ctx.agent_name, call.name,
        )
        return await call_next(call)
