"""ParallelAgent - run sub-agents concurrently.

Each sub-agent runs concurrently. Events from all agents are merged
into a single stream in the order they are produced.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event


class ParallelAgent(BaseAgent):
    """Runs sub-agents concurrently, merging their event streams.

    All sub-agents start at the same time. Their events are merged
    in the order they are produced.

    Attributes
    ----------
    agents : list[BaseAgent]
        Agents to run concurrently.
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
        """Run all sub-agents concurrently and merge their event streams.

        Parameters
        ----------
        input : str
            The user message or task description sent to all sub-agents.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : Context, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            Events from all sub-agents, interleaved in the order
            they are produced.
        """
        ctx = self._ensure_ctx(config, ctx)

        if not self.sub_agents:
            return

        # Use an internal queue to merge concurrent streams
        merged: asyncio.Queue[Event | None] = asyncio.Queue()

        async def run_and_forward(agent: BaseAgent) -> None:
            async for event in agent.astream(input, ctx=ctx):
                await merged.put(event)
            await merged.put(None)  # signal this child is done

        tasks = [
            asyncio.create_task(run_and_forward(agent))
            for agent in self.sub_agents
        ]

        try:
            done_count = 0
            while done_count < len(tasks):
                event = await merged.get()
                if event is None:
                    done_count += 1
                else:
                    yield event

            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
