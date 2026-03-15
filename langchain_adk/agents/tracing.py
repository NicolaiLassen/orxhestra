"""Tracing utilities for agent execution.

Opens a parent LangChain trace so that all LLM and tool calls within an
agent run nest under a single span — matching LangGraph's behavior.
"""

from __future__ import annotations

from typing import Any


async def open_trace(
    name: str, lc_config: dict[str, Any], input_text: str,
) -> tuple[dict[str, Any], Any]:
    """Open a parent LangChain trace so child calls nest under one span.

    Uses LangChain's ``ensure_config`` and
    ``get_async_callback_manager_for_config`` — the same utilities
    ``Runnable.ainvoke()`` uses internally.

    Parameters
    ----------
    name : str
        The agent name, used as the trace/span name.
    lc_config : dict
        A LangChain ``RunnableConfig`` dict (or ``AgentConfig``).
    input_text : str
        The user message, recorded as the chain input.

    Returns
    -------
    tuple[dict[str, Any], Any]
        ``(child_config, run_manager)``. When no callbacks are configured,
        returns ``({}, None)`` — tracing is a no-op. Call
        ``run_manager.on_chain_end()`` / ``on_chain_error()`` to close.
    """
    if not lc_config.get("callbacks"):
        return lc_config, None

    from langchain_core.runnables.config import (
        ensure_config,
        get_async_callback_manager_for_config,
    )

    config = ensure_config({
        **lc_config,
        "tags": lc_config.get("tags", []) + [f"agent:{name}"],
        "metadata": {"agent_name": name, **(lc_config.get("metadata") or {})},
        "run_name": lc_config.get("run_name", name),
    })
    manager = get_async_callback_manager_for_config(config)
    run_manager = await manager.on_chain_start(
        {"name": config.get("run_name", name)},
        {"input": input_text},
    )
    return {"callbacks": run_manager.get_child()}, run_manager
