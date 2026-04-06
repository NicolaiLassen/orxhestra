"""ParallelAgent - run sub-agents concurrently.

Each sub-agent runs concurrently. Events from all agents are merged
into a single stream in the order they are produced.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.agents.tracing import end_agent_span, error_agent_span, start_agent_span
from orxhestra.events.event import Event


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
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Run all sub-agents concurrently and merge their event streams.

        Parameters
        ----------
        input : str
            The user message or task description sent to all sub-agents.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : InvocationContext, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            Events from all sub-agents, interleaved in the order
            they are produced. Each event carries the sub-agent's
            branch path for attribution.
        """
        ctx = self._ensure_ctx(config, ctx)
        ctx, _run_mgr = await start_agent_span(
            ctx, self.name, "ParallelAgent", {"input": input},
        )

        if not self.sub_agents:
            await end_agent_span(_run_mgr)
            return

        # Use an internal queue to merge concurrent streams
        merged: asyncio.Queue[Event | None] = asyncio.Queue()

        async def run_and_forward(agent: BaseAgent) -> None:
            child_ctx = ctx.derive(agent_name=agent.name)
            async for event in agent.astream(input, ctx=child_ctx):
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
        except BaseException as exc:
            await error_agent_span(_run_mgr, exc)
            raise
        else:
            await end_agent_span(_run_mgr)
        finally:
            for t in tasks:
                t.cancel()
