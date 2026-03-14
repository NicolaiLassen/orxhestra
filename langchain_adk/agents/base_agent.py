"""Base agent abstraction.

All agents in the SDK extend BaseAgent. The contract is simple:
given an input and an InvocationContext, yield a stream of Events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from langchain_adk.agents.tracing import open_trace
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import Event, EventType
from langchain_adk.events.event_actions import EventActions

# Avoid circular import — RunConfig/StreamingMode used only at runtime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_adk.agents.run_config import RunConfig


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Agents are composable async generators. Every agent, whether a single
    LLM call or a complex orchestration tree, exposes the same interface:
    `run(input, *, ctx)` yields `Event` objects as execution proceeds.

    Sub-agents are registered via `sub_agents` and can be looked up by
    name with `find_agent()`. Parent agents set `parent_agent` when they
    register a child.

    Attributes
    ----------
    name : str
        Unique name identifying this agent within an agent tree.
    description : str
        Short description used by LLMs for routing decisions.
    sub_agents : list[BaseAgent]
        Child agents registered under this agent.
    parent_agent : BaseAgent, optional
        The parent agent (set automatically on registration).
    before_agent_callback : callable, optional
        Called before run() starts.
    after_agent_callback : callable, optional
        Called after run() completes.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = []
        self.parent_agent: Optional[BaseAgent] = None
        self.before_agent_callback: Optional[
            Callable[[InvocationContext], Awaitable[None]]
        ] = None
        self.after_agent_callback: Optional[
            Callable[[InvocationContext], Awaitable[None]]
        ] = None

    def is_streaming(self, ctx: InvocationContext) -> bool:
        """Check if SSE streaming is enabled for this run.

        When True, agents should emit partial events (``partial=True``)
        with token-level deltas. When False, only complete events.
        """
        from langchain_adk.agents.run_config import StreamingMode

        rc = ctx.run_config
        return rc is not None and rc.streaming_mode == StreamingMode.SSE

    @abstractmethod
    async def run(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the agent and stream events.

        Parameters
        ----------
        input : str
            The user message or task description.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            Events emitted during execution.
        """
        ...

    async def run_with_callbacks(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the agent, firing before/after callbacks and managing tracing.

        When ``RunConfig.langchain_config`` has callbacks, opens a parent
        LangChain trace so every LLM and tool call nests under a single span.
        The child config is stored on ``ctx.langchain_run_config`` and
        propagated to sub-agents via ``ctx.derive()``.

        Yields an AGENT_START event before execution and AGENT_END after.

        Parameters
        ----------
        input : str
            The user message or task description.
        ctx : InvocationContext
            The invocation context for this run.

        Yields
        ------
        Event
            Events emitted during execution, wrapped with start/end events.
        """
        # Set up tracing — opens parent span, stores child config on ctx
        lc_config = ctx.run_config.as_langchain_config() if ctx.run_config else {}
        child_config, run_manager = await open_trace(self.name, lc_config, input)
        if child_config:
            ctx.langchain_run_config = child_config

        if self.before_agent_callback:
            await self.before_agent_callback(ctx)

        yield Event(
            type=EventType.AGENT_START,
            session_id=ctx.session_id,
            agent_name=self.name,
        )

        try:
            async for event in self.run(input, ctx=ctx):
                yield event
        except Exception as exc:
            if run_manager:
                await run_manager.on_chain_error(exc)
            raise

        yield Event(
            type=EventType.AGENT_END,
            session_id=ctx.session_id,
            agent_name=self.name,
        )

        if run_manager:
            await run_manager.on_chain_end({"output": "completed"})

        if self.after_agent_callback:
            await self.after_agent_callback(ctx)

    def register_sub_agent(self, agent: BaseAgent) -> None:
        """Register a child agent under this agent.

        Parameters
        ----------
        agent : BaseAgent
            The child agent to register.
        """
        agent.parent_agent = self
        self.sub_agents.append(agent)

    def find_agent(self, name: str) -> Optional[BaseAgent]:
        """Recursively search the agent tree for an agent by name.

        Parameters
        ----------
        name : str
            The name of the agent to find.

        Returns
        -------
        BaseAgent or None
            The matching agent, or None if not found.
        """
        if self.name == name:
            return self
        for child in self.sub_agents:
            found = child.find_agent(name)
            if found:
                return found
        return None

    @property
    def root_agent(self) -> BaseAgent:
        """Walk up to the root of the agent tree."""
        agent = self
        while agent.parent_agent is not None:
            agent = agent.parent_agent
        return agent

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
