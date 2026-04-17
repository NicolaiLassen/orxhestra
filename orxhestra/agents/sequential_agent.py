"""SequentialAgent - run sub-agents one after another.

Each sub-agent receives the previous agent's final answer as its input,
creating a processing pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.agents.tracing import trace
from orxhestra.events.event import Event


class SequentialAgent(BaseAgent):
    """Runs a list of sub-agents sequentially, chaining their output.

    The output (final event's text) of each agent becomes the input
    to the next. All events from all agents are yielded upstream.

    Escalate events from children (e.g. :func:`exit_loop_tool` inside
    a :class:`LoopAgent`) are yielded but do NOT stop the pipeline —
    only the child that escalated stops. The sequential pipeline
    always continues to the next step.

    Attributes
    ----------
    agents : list[BaseAgent]
        Ordered list of agents to run in sequence.

    See Also
    --------
    BaseAgent : Base class this extends.
    ParallelAgent : Run sub-agents concurrently instead.
    LoopAgent : Repeat sub-agents until a stop condition.
    InvocationContext.derive : Used to build per-step child contexts.

    Examples
    --------
    >>> pipeline = SequentialAgent(
    ...     name="extract_then_summarize",
    ...     agents=[extractor_agent, summarizer_agent],
    ... )
    >>> async for event in pipeline.astream("Analyze this doc"):
    ...     if event.is_final_response():
    ...         print(event.text)
    """

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        *,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        for agent in agents:
            self.register_sub_agent(agent)

    @trace("SequentialAgent")
    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Run sub-agents in sequence, chaining final answers as input.

        Parameters
        ----------
        input : str
            The initial user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : InvocationContext, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            All events from all sub-agents in order. Each sub-agent
            receives the previous agent's final answer text as input.
        """
        if not self.sub_agents:
            return

        current_input = input

        for sub_agent in self.sub_agents:
            if ctx.end_invocation:
                return
            child_ctx = ctx.derive(agent_name=sub_agent.name)
            async for event in sub_agent.astream(current_input, ctx=child_ctx):
                yield event

                if event.is_final_response():
                    current_input = event.text
