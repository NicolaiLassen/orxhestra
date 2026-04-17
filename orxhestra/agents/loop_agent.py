"""LoopAgent - repeat sub-agents until escalation or max_iterations.

The loop terminates when:
  - A sub-agent emits an event with actions.escalate = True
  - max_iterations is reached (if set)

The escalate signal is the canonical way for a sub-agent to signal completion.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.agents.tracing import trace
from orxhestra.events.event import Event, EventType
from orxhestra.models.part import Content


class LoopAgent(BaseAgent):
    """Runs sub-agents in a loop until a stop condition is met.

    Sub-agents run in order each iteration, exactly like
    :class:`SequentialAgent` within a single loop. Between iterations
    the same input is reused (unless a sub-agent updates
    :attr:`InvocationContext.state` which an instruction provider can
    use to build new input).

    Termination:
      - Any event with :attr:`EventActions.escalate` = True stops the loop.
      - ``max_iterations`` reached stops the loop (yields an error event).
      - ``should_continue`` callback returning False stops the loop.

    Attributes
    ----------
    agents : list[BaseAgent]
        Sub-agents to run each iteration (in order).
    max_iterations : int, optional
        Maximum number of full loop cycles. None = unlimited.
    should_continue : callable, optional
        Optional callable inspecting the last event to decide whether to
        keep looping. Return False to stop.

    See Also
    --------
    BaseAgent : Base class this extends.
    SequentialAgent : Runs sub-agents once, in order.
    ParallelAgent : Runs sub-agents concurrently.
    exit_loop_tool : Tool a sub-agent calls to escalate and stop.
    EventActions.escalate : Flag that terminates the loop.

    Examples
    --------
    >>> researcher = LoopAgent(
    ...     name="iterative_research",
    ...     agents=[critic_agent, refiner_agent],
    ...     max_iterations=5,
    ... )
    >>> final = await researcher.ainvoke("Research question X")
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

    @trace("LoopAgent")
    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Run sub-agents in a loop until a termination condition is met.

        Each iteration runs all sub-agents in order. Between iterations
        the child context gets a unique branch suffix (``iter_0``,
        ``iter_1``, …) so that the child LLM only sees conversation
        history from the *current* iteration — preventing token
        accumulation across loops (analogous to invocation-level
        filtering).

        ``ctx.state`` is shared across all iterations so structured
        data (e.g. accumulated sources) survives the loop.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : InvocationContext, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            All events from all sub-agents across all iterations.
            The loop stops when a sub-agent escalates, ``max_iterations``
            is reached, or ``should_continue`` returns False.
        """
        if not self.sub_agents:
            return

        iteration = 0
        last_event: Event | None = None

        while True:
            if ctx.end_invocation:
                return

            if self.max_iterations is not None and iteration >= self.max_iterations:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_text(
                        f"LoopAgent '{self.name}' reached max_iterations "
                        f"({self.max_iterations}) without escalating."
                    ),
                    metadata={"error": True},
                )
                return

            for sub_agent in self.sub_agents:
                # Unique branch per iteration so the child's LLM doesn't
                # re-read conversation history from previous iterations
                child_ctx = ctx.derive(
                    agent_name=sub_agent.name,
                    branch_suffix=f"{sub_agent.name}.iter_{iteration}",
                )
                async for event in sub_agent.astream(input, ctx=child_ctx):
                    yield event
                    last_event = event

                    if event.actions.escalate:
                        return

            # Custom termination check
            if self.should_continue is not None and last_event is not None:
                if not self.should_continue(last_event):
                    return

            # Reset sub-agent states between iterations so children
            # start fresh each loop cycle.
            ctx.reset_sub_agent_states(self.name)

            iteration += 1
