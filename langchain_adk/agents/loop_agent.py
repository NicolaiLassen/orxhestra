"""LoopAgent - repeat sub-agents until escalation or max_iterations.

The loop terminates when:
  - A sub-agent emits an event with actions.escalate = True
  - max_iterations is reached (if set)

The escalate signal is the canonical way for a sub-agent to signal completion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from langchain_core.runnables import RunnableConfig

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event, EventType
from langchain_adk.models.part import Content


class LoopAgent(BaseAgent):
    """Runs sub-agents in a loop until a stop condition is met.

    Sub-agents run in order each iteration, exactly like SequentialAgent
    within a single loop. Between iterations the same input is reused
    (unless a sub-agent updates ctx.state which an instruction provider
    can use to build new input).

    Termination:
      - Any event with actions.escalate = True stops the loop.
      - max_iterations reached stops the loop (yields an error event).
      - should_continue callback returning False stops the loop.

    Attributes
    ----------
    agents : list[BaseAgent]
        Sub-agents to run each iteration (in order).
    max_iterations : int, optional
        Maximum number of full loop cycles. None = unlimited.
    should_continue : callable, optional
        Optional callable inspecting the last event to decide whether to
        keep looping. Return False to stop.
    """

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        *,
        description: str = "",
        max_iterations: int | None = 10,
        should_continue: Callable[[Event], bool] | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        for agent in agents:
            self.register_sub_agent(agent)
        self.max_iterations = max_iterations
        self.should_continue = should_continue

    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: Context | None = None,
    ) -> AsyncIterator[Event]:
        """Run sub-agents in a loop until a termination condition is met.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : Context, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            All events from all sub-agents across all iterations.
            The loop stops when a sub-agent escalates, ``max_iterations``
            is reached, or ``should_continue`` returns False.
        """
        ctx = self._ensure_ctx(config, ctx)

        if not self.sub_agents:
            return

        iteration = 0
        last_event: Event | None = None

        while True:
            if self.max_iterations is not None and iteration >= self.max_iterations:
                yield Event(
                    type=EventType.AGENT_MESSAGE,
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    author=self.name,
                    content=Content.from_text(
                        f"LoopAgent '{self.name}' reached max_iterations "
                        f"({self.max_iterations}) without escalating."
                    ),
                    metadata={"error": True},
                )
                return

            for sub_agent in self.sub_agents:
                async for event in sub_agent.astream(input, ctx=ctx):
                    yield event
                    last_event = event

                    if event.actions.escalate:
                        return

            # Custom termination check
            if self.should_continue is not None and last_event is not None:
                if not self.should_continue(last_event):
                    return

            iteration += 1
