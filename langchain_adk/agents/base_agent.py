"""Base agent abstraction.

All agents in the SDK extend BaseAgent. Subclasses override ``astream()``
to yield events. The public API matches LangChain's Runnable interface:

  - ``astream(input, config, *, ctx)`` — async iterator of events
  - ``ainvoke(input, config, *, ctx)`` — async, returns final answer event
  - ``stream(input, config)``          — sync iterator of events
  - ``invoke(input, config)``          — sync, returns final answer event
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from langchain_adk.agents.context import Context
from langchain_adk.agents.tracing import open_trace
from langchain_adk.events.event import Event, EventType


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Subclasses override ``astream()`` to yield ``Event`` objects.
    The public API matches LangChain's ``Runnable`` interface.

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
        Called before the agent runs.
    after_agent_callback : callable, optional
        Called after the agent completes.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = []
        self.parent_agent: BaseAgent | None = None
        self.before_agent_callback: Callable[[Context], Awaitable[None]] | None = None
        self.after_agent_callback: Callable[[Context], Awaitable[None]] | None = None

    def _ensure_ctx(
        self,
        config: RunnableConfig | None = None,
        ctx: Context | None = None,
    ) -> Context:
        """Return the provided ctx or create a fresh one."""
        if ctx is not None:
            return ctx
        return Context(
            session_id=str(uuid4()),
            agent_name=self.name,
            run_config=config or {},
        )

    # ------------------------------------------------------------------
    # Abstract: subclasses override this
    # ------------------------------------------------------------------

    @abstractmethod
    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: Context | None = None,
    ) -> AsyncIterator[Event]:
        """Stream events from the agent asynchronously.

        Subclasses must override this and ``yield`` events.

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
            Events emitted during execution.
        """
        yield  # type: ignore[misc]  # pragma: no cover

    # ------------------------------------------------------------------
    # Public API (matches LangChain Runnable interface)
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: Context | None = None,
    ) -> Event:
        """Run to completion, return the final answer event.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict.
        ctx : Context, optional
            Invocation context. Auto-created if not provided.

        Returns
        -------
        Event
            The agent's final answer event.

        Raises
        ------
        RuntimeError
            If the agent finishes without producing a final answer.
        """
        last_answer: Event | None = None
        async for event in self.astream(input, config, ctx=ctx):
            if event.is_final_response():
                last_answer = event
        if last_answer is None:
            raise RuntimeError(f"Agent {self.name!r} produced no final answer")
        return last_answer

    def stream(
        self,
        input: str,
        config: RunnableConfig | None = None,
    ) -> Iterator[Event]:
        """Stream events from the agent synchronously.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict.

        Returns
        -------
        Iterator[Event]
            Events emitted during execution.
        """
        async def _collect() -> list[Event]:
            return [e async for e in self.astream(input, config)]
        return iter(asyncio.run(_collect()))

    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
    ) -> Event:
        """Run to completion synchronously, return the final answer event.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict.

        Returns
        -------
        Event
            The agent's final answer event.
        """
        return asyncio.run(self.ainvoke(input, config))

    # ------------------------------------------------------------------
    # Internal: queue-based wrapper for Runner / AgentTool
    # ------------------------------------------------------------------

    async def run_async(
        self,
        input: str,
        *,
        ctx: Context,
    ) -> None:
        """Drain ``astream()`` into ``ctx.events`` queue.

        Used by Runner, AgentTool, and orchestration internals that need
        queue-based event consumption. Wraps with AGENT_START/AGENT_END
        lifecycle events, callbacks, and tracing.

        Parameters
        ----------
        input : str
            The user message or task description.
        ctx : Context
            The invocation context for this run.
        """
        lc_config = ctx.run_config
        child_config, run_manager = await open_trace(self.name, lc_config, input)
        if child_config:
            ctx.langchain_run_config = child_config

        if self.before_agent_callback:
            await self.before_agent_callback(ctx)

        await ctx.events.put(Event(
            type=EventType.AGENT_START,
            session_id=ctx.session_id,
            agent_name=self.name,
        ))

        try:
            async for event in self.astream(input, ctx=ctx):
                await ctx.events.put(event)
        except Exception as exc:
            if run_manager:
                await run_manager.on_chain_error(exc)
            raise

        await ctx.events.put(Event(
            type=EventType.AGENT_END,
            session_id=ctx.session_id,
            agent_name=self.name,
        ))

        if run_manager:
            await run_manager.on_chain_end({"output": "completed"})

        if self.after_agent_callback:
            await self.after_agent_callback(ctx)

        await ctx.events.put(None)  # sentinel: done

    # ------------------------------------------------------------------
    # Agent tree management
    # ------------------------------------------------------------------

    def register_sub_agent(self, agent: BaseAgent) -> None:
        """Register a child agent under this agent.

        Parameters
        ----------
        agent : BaseAgent
            The child agent to register.
        """
        agent.parent_agent = self
        self.sub_agents.append(agent)

    def find_agent(self, name: str) -> BaseAgent | None:
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
