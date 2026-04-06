"""Hierarchical tracing via LangChain's AsyncCallbackManager.

Emits ``on_chain_start`` / ``on_chain_end`` spans so that any LangChain
callback handler (Langfuse, LangSmith, custom) automatically builds a
nested execution trace.  Zero overhead when no callbacks are configured.

Usage in an agent's ``astream()``::

    ctx, run_manager = await start_agent_span(
        ctx, self.name, "LlmAgent", {"input": input}
    )
    try:
        ...  # yield events
    except BaseException as exc:
        await error_agent_span(run_manager, exc)
        raise
    else:
        await end_agent_span(run_manager)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
)

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext

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
    ctx:
        Current invocation context.
    agent_name:
        Human-readable name shown in the trace (e.g. ``"Researcher"``).
    agent_type:
        Agent class name (e.g. ``"LlmAgent"``, ``"SequentialAgent"``).
    inputs:
        Serialisable dict passed to ``on_chain_start`` as the inputs.

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
