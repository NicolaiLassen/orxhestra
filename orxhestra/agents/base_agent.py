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
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from orxhestra.events.event import Event, EventType

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext


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
    signing_key : Ed25519PrivateKey, optional
        Ed25519 private key giving this agent its own identity.
        When set, events emitted by this agent are signed with
        this key (takes priority over the context-level key).
        Requires ``orxhestra[auth]``.
    signing_did : str
        The ``did:key`` identifier for ``signing_key``.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        signing_key: Any | None = None,
        signing_did: str = "",
    ) -> None:
        """Initialize the agent.

        Parameters
        ----------
        name : str
            Unique name for this agent within its tree.
        description : str
            Short description used by LLMs for routing decisions.
        signing_key : Ed25519PrivateKey, optional
            Ed25519 private key giving this agent its own identity.
            When set, events emitted by this agent are signed with
            this key (takes priority over any context-level key).
            Requires ``orxhestra[auth]``.
        signing_did : str
            The ``did:key`` identifier that corresponds to
            ``signing_key``. Required when ``signing_key`` is set.
        """
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = []
        self.parent_agent: BaseAgent | None = None
        self.signing_key: Any | None = signing_key
        self.signing_did: str = signing_did

    def _ensure_ctx(
        self,
        config: RunnableConfig | None = None,
        ctx: InvocationContext | None = None,
    ) -> InvocationContext:
        """Return the provided ``ctx`` or create a fresh one.

        Parameters
        ----------
        config : RunnableConfig, optional
            LangChain run config attached to a fresh context.
        ctx : InvocationContext, optional
            Pre-existing context. Returned unchanged if provided.

        Returns
        -------
        InvocationContext
            ``ctx`` if given; otherwise a new context with a random
            session id and this agent's name.
        """
        if ctx is not None:
            return ctx
        from orxhestra.agents.invocation_context import InvocationContext as _IC

        return _IC(
            session_id=str(uuid4()),
            agent_name=self.name,
            run_config=config or {},
        )

    def _emit_event(
        self,
        ctx: InvocationContext,
        type: EventType,
        **kwargs: Any,
    ) -> Event:
        """Create an Event with context attribution (branch, session, agent).

        When the agent or context provides a signing key, the event is
        signed and chained to the previous non-partial event emitted
        on the same ``branch`` via :attr:`Event.prev_signature`.  The
        per-branch chain heads are stored in
        ``ctx.state["_orx_chain_heads"]`` so they round-trip through
        any :class:`SessionService` that persists ``state``.

        Parameters
        ----------
        ctx : InvocationContext
            The invocation context providing session_id and branch.
        type : EventType
            The event type.
        **kwargs
            Additional fields passed to the Event constructor.

        Returns
        -------
        Event
            A new Event with branch, session_id, and agent_name set.
            When signing is enabled, also carries ``signature``,
            ``signer_did``, and ``prev_signature``.
        """
        kwargs.setdefault("author", self.name)
        event = Event(
            type=type,
            session_id=ctx.session_id,
            invocation_id=ctx.invocation_id,
            agent_name=self.name,
            branch=ctx.branch,
            **kwargs,
        )
        key = self.signing_key or ctx.signing_key
        did = self.signing_did or ctx.signing_did
        if key is not None and did:
            try:
                from orxhestra.security.crypto import sign_json_payload

                chain_heads = ctx.state.setdefault("_orx_chain_heads", {})
                event.prev_signature = chain_heads.get(ctx.branch)
                event.signature = sign_json_payload(
                    key, event.signable_payload(),
                )
                event.signer_did = did
                # Only non-partial events advance the chain head — partials
                # are ephemeral streaming chunks and shouldn't fragment it.
                if not event.partial:
                    chain_heads[ctx.branch] = event.signature
            except ImportError:
                pass
        return event


    @abstractmethod
    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Stream events from the agent asynchronously.

        Subclasses must override this and ``yield`` events.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : InvocationContext, optional
            Invocation context. Auto-created via :meth:`_ensure_ctx`
            if not provided.

        Yields
        ------
        Event
            :class:`Event` objects emitted during execution.

        See Also
        --------
        Runner.astream : Preferred entry point when you need session
            persistence.
        InvocationContext : Runtime state passed through the agent tree.
        """
        yield  # type: ignore[misc]  # pragma: no cover


    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> Event:
        """Run to completion, return the final answer event.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict.
        ctx : InvocationContext, optional
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
        """Walk up to the root of the agent tree.

        Returns
        -------
        BaseAgent
            The top-most ancestor (an agent with no parent). Returns
            ``self`` if this agent has no parent.
        """
        agent = self
        while agent.parent_agent is not None:
            agent = agent.parent_agent
        return agent

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
