"""AgentConfig - per-run configuration for agent execution.

Extends LangChain's ``RunnableConfig`` TypedDict with ADK-specific fields.
Since it's a superset of ``RunnableConfig``, it works anywhere LangChain
expects a config dict.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig


class AgentConfig(RunnableConfig, total=False):
    """LangChain ``RunnableConfig`` extended with ADK-specific fields.

    All standard LangChain config keys (``callbacks``, ``tags``, ``metadata``,
    ``run_name``, ``max_concurrency``, ``configurable``, ``run_id``,
    ``recursion_limit``) are inherited from ``RunnableConfig``.

    ADK-specific fields:

    Attributes
    ----------
    max_llm_calls : int
        Maximum total LLM calls allowed in this run. Defaults to 500
        when not set. Set to 0 for no limit.
    """

    max_llm_calls: int
