"""SequentialAgent - run sub-agents one after another.

Each sub-agent receives the previous agent's final answer as its input,
creating a processing pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.agents.tracing import end_agent_span, error_agent_span, start_agent_span
from orxhestra.events.event import Event


class SequentialAgent(BaseAgent):
    """Runs a list of sub-agents sequentially, chaining their output.

    The output (final event's text) of each agent becomes the input
    to the next. All events from all agents are yielded upstream.

    Escalate events from children (e.g. exit_loop inside a LoopAgent)
    are yielded but do NOT stop the pipeline — only the child that
    escalated stops. The sequential pipeline always continues to the
    next step.

    Attributes
    ----------
    agents : list[BaseAgent]
        Ordered list of agents to run in sequence.
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
        ctx = self._ensure_ctx(config, ctx)
        ctx, _run_mgr = await start_agent_span(
            ctx, self.name, "SequentialAgent", {"input": input},
        )

        if not self.sub_agents:
            await end_agent_span(_run_mgr)
            return

        current_input = input

        try:
            for sub_agent in self.sub_agents:
                if ctx.end_invocation:
                    return
                child_ctx = ctx.derive(agent_name=sub_agent.name)
                async for event in sub_agent.astream(current_input, ctx=child_ctx):
                    yield event

                    if event.is_final_response():
                        current_input = event.text
        except BaseException as exc:
            await error_agent_span(_run_mgr, exc)
            raise
        else:
            await end_agent_span(_run_mgr)
