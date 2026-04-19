"""Composable middleware for agent lifecycle.

Middleware intercepts agent invocations, LLM calls, tool calls, and
events. Stack members are called in order; first registered is the
outermost layer (onion pattern).

Every middleware ships here. The two opt-in security-focused members
— :class:`TrustMiddleware` and :class:`AttestationMiddleware` — live
alongside the rest; their domain types (policy, providers, claims)
live under :mod:`orxhestra.trust`.

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

from orxhestra.middleware.attestation import AttestationMiddleware
from orxhestra.middleware.base import Middleware, ToolCall
from orxhestra.middleware.callback import CallbackMiddleware
from orxhestra.middleware.logging import LoggingMiddleware
from orxhestra.middleware.stack import MiddlewareStack
from orxhestra.middleware.trust import TrustMiddleware

__all__ = [
    "Middleware",
    "ToolCall",
    "MiddlewareStack",
    "CallbackMiddleware",
    "LoggingMiddleware",
    "TrustMiddleware",        # opt-in, requires orxhestra[auth]
    "AttestationMiddleware",  # opt-in, requires orxhestra[auth]
]
