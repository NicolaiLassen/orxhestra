"""Grouped lifecycle callbacks for LlmAgent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse


@dataclass
class LlmAgentCallbacks:
    """Grouped lifecycle callbacks for an LlmAgent.

    Bundles the five optional hooks that LlmAgent supports so they
    can be passed around as a single unit rather than five separate
    constructor arguments.

    Attributes
    ----------
    before_model : callable, optional
        Called with ``(ctx, request)`` before each LLM call.
    after_model : callable, optional
        Called with ``(ctx, response)`` after each LLM call.
    on_model_error : callable, optional
        Called with ``(ctx, request, exception)`` when an LLM call
        raises.  Return an ``LlmResponse`` to recover, or ``None``
        to push an error event.
    before_tool : callable, optional
        Called with ``(ctx, tool_name, tool_args)`` before each tool
        execution.
    after_tool : callable, optional
        Called with ``(ctx, tool_name, result)`` after each tool
        execution.
    """

    before_model: (
        Callable[[InvocationContext, LlmRequest], Awaitable[None]] | None
    ) = None
    after_model: (
        Callable[[InvocationContext, LlmResponse], Awaitable[None]] | None
    ) = None
    on_model_error: (
        Callable[
            [InvocationContext, LlmRequest, Exception],
            Awaitable[LlmResponse | None],
        ]
        | None
    ) = None
    before_tool: (
        Callable[[InvocationContext, str, dict], Awaitable[None]] | None
    ) = None
    after_tool: (
        Callable[[InvocationContext, str, Any], Awaitable[None]] | None
    ) = None
