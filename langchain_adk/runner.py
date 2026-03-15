"""Runner - orchestrates agent execution with session management.

The Runner is the main entry point for running agents. It:

1. Fetches or creates the session via the session service.
2. Builds an ``Context`` from the session and run config.
3. Iterates the agent's ``astream()`` method.
4. Persists every event to the session via ``append_event()``.
5. Yields the event stream back to the caller.

Basic usage::

    from langchain_adk.runner import Runner
    from langchain_adk.sessions import InMemorySessionService

    runner = Runner(
        agent=my_agent,
        app_name="my_app",
        session_service=InMemorySessionService(),
    )

    async for event in runner.run_async(
        user_id="user_1",
        session_id="session_1",
        new_message="Hello!",
    ):
        if event.is_final_response():
            print(event.text)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event, EventType
from langchain_adk.models.part import Content
from langchain_adk.sessions.base_session_service import BaseSessionService
from langchain_adk.sessions.session import Session

if TYPE_CHECKING:
    from langchain_adk.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Runner:
    """Orchestrates a single agent with session persistence.

    Ties together an agent, a session service, and the invocation context.
    All events are automatically persisted to the session so that callers
    only need to consume the event stream.

    Parameters
    ----------
    agent : BaseAgent
        The root agent to run.
    app_name : str
        Application identifier. Used to namespace sessions.
    session_service : BaseSessionService
        Where sessions are stored and retrieved.
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        app_name: str,
        session_service: BaseSessionService,
    ) -> None:
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service

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

    async def run_async(
        self,
        *,
        user_id: str,
        session_id: str,
        new_message: str,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[Event]:
        """Run the agent and yield its event stream.

        Launches the agent, drains events from the queue, persists each
        event, and yields to the caller.
        """
        resolved_config = config or {}

        session = await self.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
        )

        ctx = Context(
            session_id=session.id,
            user_id=user_id,
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            session=session,
            run_config=resolved_config,
        )

        user_event = Event(
            type=EventType.USER_MESSAGE,
            author="user",
            session_id=session.id,
            invocation_id=ctx.invocation_id,
            content=Content.from_text(new_message),
        )
        await self.session_service.append_event(session, user_event)

        logger.debug(
            "Runner starting: agent=%s session=%s user=%s",
            self.agent.name,
            session_id,
            user_id,
        )

        async for event in self.agent.astream(new_message, ctx=ctx):
            await self.session_service.append_event(session, event)
            yield event
