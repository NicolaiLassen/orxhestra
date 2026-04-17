"""Composable middleware for agent lifecycle.

Middleware intercepts agent invocations, LLM calls, tool calls, and
events. Stack members are called in order; first registered is the
outermost layer (onion pattern).

Usage::

    from orxhestra.middleware import Middleware, LoggingMiddleware
    from orxhestra import Runner

    runner = Runner(
        agent=my_agent,
        middleware=[LoggingMiddleware()],
    )

Existing ``LlmAgentCallbacks`` keep working unchanged — they are
equivalent to a single ``CallbackMiddleware`` at the end of the stack.
"""

from __future__ import annotations

from orxhestra.middleware.base import Middleware, ToolCall
from orxhestra.middleware.callback import CallbackMiddleware
from orxhestra.middleware.logging import LoggingMiddleware
from orxhestra.middleware.stack import MiddlewareStack

__all__ = [
    "Middleware",
    "ToolCall",
    "MiddlewareStack",
    "CallbackMiddleware",
    "LoggingMiddleware",
]
