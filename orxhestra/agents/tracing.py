"""Hierarchical tracing via LangChain's AsyncCallbackManager.

Emits ``on_chain_start`` / ``on_chain_end`` spans so that any LangChain
callback handler (Langfuse, LangSmith, custom) automatically builds a
nested execution trace.  Zero overhead when no callbacks are configured.

Usage — decorator (preferred)::

    @trace("LlmAgent")
    async def astream(self, input, config=None, *, ctx=None):
        ...  # yield events normally, no tracing code needed

Usage — manual (for non-BaseAgent callers like Runner)::

    ctx, run_mgr = await start_agent_span(ctx, name, "Runner", {"input": msg})
    _span_err = None
    try:
        ...
    except BaseException as exc:
        _span_err = exc
        raise
    finally:
        if _span_err:
            await error_agent_span(run_mgr, _span_err)
        else:
            await end_agent_span(run_mgr)
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
)

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event

logger: logging.Logger = logging.getLogger(__name__)


def _get_callbacks(
    ctx: InvocationContext,
) -> list[Any] | AsyncCallbackManager | None:
    """Extract callbacks from ``run_config``, or ``None`` if absent."""
    cbs = ctx.run_config.get("callbacks")
    if cbs is None:
        return None
    if isinstance(cbs, list) and len(cbs) == 0:
        return None
    return cbs


async def start_agent_span(
    ctx: InvocationContext,
    agent_name: str,
    agent_type: str,
    inputs: dict[str, Any],
) -> tuple[InvocationContext, AsyncCallbackManagerForChainRun | None]:
    """Open a trace span for an agent and return an updated context.

    If no callbacks are configured the original *ctx* is returned
    unchanged alongside ``None`` — zero allocation, zero overhead.

    The returned context has ``run_config["callbacks"]`` replaced with
    a **child** ``AsyncCallbackManager`` whose ``parent_run_id`` points
    to the newly opened span.  Any nested operation (sub-agent, LLM
    call, tool invocation) that uses this child manager will appear as
    a child span in the trace.

    Parameters
    ----------
    ctx :
        Current invocation context.
    agent_name :
        Human-readable name shown in the trace (e.g. ``"Researcher"``).
    agent_type :
        Agent class name (e.g. ``"LlmAgent"``, ``"SequentialAgent"``).
    inputs :
        Serializable dict passed to ``on_chain_start`` as the inputs.

    Returns
    -------
    tuple[InvocationContext, AsyncCallbackManagerForChainRun | None]
        Updated context and the run manager (call ``end_agent_span`` /
        ``error_agent_span`` when the agent finishes).
    """
    callbacks = _get_callbacks(ctx)
    if callbacks is None:
        return ctx, None

    manager: AsyncCallbackManager = AsyncCallbackManager.configure(
        inheritable_callbacks=callbacks,
        inheritable_metadata=ctx.run_config.get("metadata"),
        inheritable_tags=ctx.run_config.get("tags"),
    )

    serialized: dict[str, Any] = {
        "name": agent_name,
        "id": [agent_type, agent_name],
    }

    run_manager: AsyncCallbackManagerForChainRun = await manager.on_chain_start(
        serialized, inputs,
    )

    child_manager: AsyncCallbackManager = run_manager.get_child(tag=agent_name)

    new_config: dict[str, Any] = {
        **ctx.run_config,
        "callbacks": child_manager,
    }
    new_ctx: InvocationContext = ctx.model_copy(
        update={"run_config": new_config},
    )
    return new_ctx, run_manager


async def end_agent_span(
    run_manager: AsyncCallbackManagerForChainRun | None,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Close a previously opened agent span.

    Safe to call with ``None`` (no-op when tracing is disabled).
    """
    if run_manager is not None:
        await run_manager.on_chain_end(outputs or {})


async def error_agent_span(
    run_manager: AsyncCallbackManagerForChainRun | None,
    error: BaseException,
) -> None:
    """Record an error and close a previously opened agent span.

    Safe to call with ``None`` (no-op when tracing is disabled).
    """
    if run_manager is not None:
        await run_manager.on_chain_error(error)


def trace(agent_type: str):
    """Decorator that wraps an agent's ``astream()`` with a trace span.

    Handles ``_ensure_ctx``, ``start_agent_span``, ``end_agent_span``,
    and ``error_agent_span`` automatically.  The decorated method
    receives a fully prepared ``ctx`` — it must **not** call
    ``_ensure_ctx()`` itself.

    Usage::

        @trace("LlmAgent")
        async def astream(self, input, config=None, *, ctx=None):
            # ctx is already set up — just yield events
            yield self._emit_event(ctx, ...)
    """

    def decorator(
        fn: Any,
    ) -> Any:
        """Wrap ``fn`` (an ``astream`` coroutine) with trace span management.

        Parameters
        ----------
        fn : callable
            The agent's ``astream`` method being decorated.

        Returns
        -------
        callable
            A replacement coroutine that opens a span, forwards events,
            and closes the span on success or error.
        """
        @functools.wraps(fn)
        async def wrapper(
            self: Any,
            input: str,
            config: Any = None,
            *,
            ctx: Any = None,
        ) -> AsyncIterator[Event]:
            """Run the wrapped ``astream`` under an agent span.

            Creates a trace span, yields every event from ``fn``, and
            closes the span with the final event text on success or
            the exception on error.
            """
            ctx = self._ensure_ctx(config, ctx)
            ctx, run_mgr = await start_agent_span(
                ctx, self.name, agent_type, {"input": input},
            )
            last_text: str = ""
            err: BaseException | None = None
            try:
                async for event in fn(self, input, config, ctx=ctx):
                    if hasattr(event, "text") and event.text:
                        last_text = event.text
                    yield event
            except GeneratorExit:
                # Parent stopped consuming — normal cleanup, not an error.
                pass
            except BaseException as exc:
                err = exc
                raise
            finally:
                if err is not None:
                    await error_agent_span(run_mgr, err)
                else:
                    await end_agent_span(
                        run_mgr, {"output": last_text} if last_text else None,
                    )

        return wrapper

    return decorator
