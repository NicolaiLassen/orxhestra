"""Runner - orchestrates agent execution with session management.

The Runner is the main entry point for running agents. It:

1. Fetches or creates the session via the session service.
2. Builds a ``InvocationContext`` from the session and run config.
3. Streams the agent via ``astream()``, following transfers.
4. Persists every event to the session via ``append_event()``.
5. Yields the event stream back to the caller.

Basic usage::

    from orxhestra.runner import Runner
    from orxhestra.sessions import InMemorySessionService

    runner = Runner(
        agent=my_agent,
        app_name="my_app",
        session_service=InMemorySessionService(),
    )

    async for event in runner.astream(
        user_id="user_1",
        session_id="session_1",
        new_message="Hello!",
    ):
        if event.is_final_response():
            print(event.text)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.events.event import Event, EventType
from orxhestra.models.part import Content
from orxhestra.artifacts.base_artifact_service import BaseArtifactService
from orxhestra.sessions.base_session_service import BaseSessionService
from orxhestra.sessions.compaction import CompactionConfig, compact_session
from orxhestra.sessions.session import Session

if TYPE_CHECKING:
    from orxhestra.agents.base_agent import BaseAgent


class Runner:
    """Orchestrates a single agent with session persistence.

    Ties together an agent, a session service, and the invocation context.
    All events are automatically persisted to the session so that callers
    only need to consume the event stream.

    When ``compaction_config`` is provided, the runner automatically
    compacts old session events after each invocation using an LLM
    summarizer.  This keeps the context window manageable across long
    multi-turn conversations.

    Parameters
    ----------
    agent : BaseAgent
        The root agent to run.
    app_name : str
        Application identifier. Used to namespace sessions.
    session_service : BaseSessionService
        Where sessions are stored and retrieved.
    artifact_service : BaseArtifactService, optional
        Where artifacts (files, blobs) are stored.
    compaction_config : CompactionConfig, optional
        If set, enables automatic session compaction after each
        invocation.
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        app_name: str,
        session_service: BaseSessionService,
        artifact_service: BaseArtifactService | None = None,
        compaction_config: CompactionConfig | None = None,
        default_config: dict | None = None,
    ) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service
        self.artifact_service = artifact_service
        self.compaction_config = compaction_config
        self._base_config = default_config or {}

    async def get_or_create_session(
        self,
        *,
        user_id: str,
        session_id: str,
        initial_state: dict | None = None,
    ) -> Session:
        """Fetch an existing session or create a new one.

        Parameters
        ----------
        user_id : str
            The user identifier.
        session_id : str
            The session identifier.
        initial_state : dict, optional
            Initial state for a newly created session.

        Returns
        -------
        Session
            The existing or newly created session.
        """
        session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if session is None:
            session = await self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id,
                state=initial_state or {},
            )
        return session

    async def astream(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: str,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[Event]:
        """Stream events from the agent, persisting to the session.

        Persists the user message, then streams agent events. If the agent
        emits a ``transfer_to_agent`` action, the runner resolves the target
        in the agent tree and continues streaming from that agent.

        Parameters
        ----------
        user_id : str
            The user identifier.
        session_id : str
            The session identifier.
        new_message : str
            The user's input message.
        config : RunnableConfig, optional
            LangChain runnable configuration forwarded to the agent.

        Yields
        ------
        Event
            Each event produced by the agent (and any transfer targets).
        """
        session = await self.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
        )

        run_config = {**self._base_config, **(config or {})}

        # Expose session metadata so callbacks (tracing, etc.) can use it
        meta = dict(run_config.get("metadata", {}))
        meta.setdefault("session_id", session.id)
        meta.setdefault("user_id", user_id)
        meta.setdefault("app_name", self.app_name)
        run_config["metadata"] = meta

        ctx = InvocationContext(
            session_id=session.id,
            user_id=user_id,
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            input_content=new_message,
            session=session,
            run_config=run_config,
            current_agent=self.agent,
            artifact_service=self.artifact_service,
        )

        user_event = Event(
            type=EventType.USER_MESSAGE,
            author="user",
            session_id=session.id,
            invocation_id=ctx.invocation_id,
            content=Content.from_text(new_message),
        )
        await self.session_service.append_event(session, user_event)

        current_agent = self.agent

        while True:
            transfer_target: str | None = None

            async for event in current_agent.astream(new_message, ctx=ctx):
                await self.session_service.append_event(session, event)
                yield event

                if event.actions and event.actions.transfer_to_agent:
                    transfer_target = event.actions.transfer_to_agent

            if transfer_target is None:
                break

            target = self.agent.find_agent(transfer_target)
            if target is None:
                break

            current_agent = target
            ctx = ctx.model_copy(update={"agent_name": target.name})

        # Run compaction after all events are yielded from the agent.
        # Only compact at the end of an invocation, never mid-stream.
        if self.compaction_config is not None:
            await compact_session(
                session, self.session_service, self.compaction_config,
            )

