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
from orxhestra.agents.tracing import trace
from orxhestra.events.event import Event


class ParallelAgent(BaseAgent):
    """Runs sub-agents concurrently, merging their event streams.

    All sub-agents start at the same time. Their events are merged
    in the order they are produced. Events from different sub-agents
    carry distinct ``branch`` attribution so consumers can demultiplex.

    Attributes
    ----------
    agents : list[BaseAgent]
        Agents to run concurrently.

    See Also
    --------
    BaseAgent : Base class this extends.
    SequentialAgent : Chain agents instead of running them in parallel.
    LoopAgent : Iterate over sub-agents until a stop condition.
    Event.branch : Carries the originating sub-agent path.

    Examples
    --------
    >>> fanout = ParallelAgent(
    ...     name="research_fanout",
    ...     agents=[web_agent, docs_agent, code_agent],
    ... )
    >>> async for event in fanout.astream("Find info about X"):
    ...     print(f"[{event.branch}] {event.text}")
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

    @trace("ParallelAgent")
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
        if not self.sub_agents:
            return

        # Use an internal queue to merge concurrent streams
        merged: asyncio.Queue[Event | None] = asyncio.Queue()

        async def run_and_forward(agent: BaseAgent) -> None:
            """Stream events from one sub-agent into the merged queue.

            Parameters
            ----------
            agent : BaseAgent
                The sub-agent to run.
            """
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
        finally:
            for t in tasks:
                t.cancel()
            # Suppress CancelledError from already-finished tasks.
            await asyncio.gather(*tasks, return_exceptions=True)
