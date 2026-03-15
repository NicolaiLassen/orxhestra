"""SequentialAgent - run sub-agents one after another.

Each sub-agent receives the previous agent's final answer as its input,
creating a processing pipeline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event


class SequentialAgent(BaseAgent):
    """Runs a list of sub-agents sequentially, chaining their output.

    The output (final event's text) of each agent becomes the input
    to the next. All events from all agents are yielded upstream.

    If a sub-agent emits an event with `actions.escalate = True`, the
    pipeline stops early.

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
        ctx: Context | None = None,
    ) -> AsyncIterator[Event]:
        """Run sub-agents in sequence, chaining final answers as input.

        Parameters
        ----------
        input : str
            The initial user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : Context, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            All events from all sub-agents in order. Each sub-agent
            receives the previous agent's final answer text as input.
        """
        ctx = self._ensure_ctx(config, ctx)

        if not self.sub_agents:
            return

        current_input = input

        for sub_agent in self.sub_agents:
            async for event in sub_agent.astream(current_input, ctx=ctx):
                yield event

                if event.actions.escalate:
                    return

                if event.is_final_response():
                    current_input = event.text
