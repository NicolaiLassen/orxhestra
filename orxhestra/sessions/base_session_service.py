"""Abstract session service interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from orxhestra.events.event import Event
from orxhestra.sessions.session import Session


class BaseSessionService(ABC):
    """Abstract base for session storage backends.

    Implement this to add persistence (MongoDB, Postgres, Redis, etc.).
    The SDK ships with :class:`InMemorySessionService` for local dev
    and tests and :class:`DatabaseSessionService` for SQLAlchemy-backed
    persistence.

    See Also
    --------
    Session : The model this service stores.
    InMemorySessionService : Non-persistent backend.
    DatabaseSessionService : SQLAlchemy async backend.
    Runner : Uses a session service to persist every invocation.
    """

    @abstractmethod
    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create a new session.

        Parameters
        ----------
        app_name : str
            Application namespace for scoping sessions.
        user_id : str
            The user who owns this session.
        state : dict[str, Any], optional
            Initial state to populate the session with.
        session_id : str, optional
            Explicit session ID; auto-generated if omitted.

        Returns
        -------
        Session
            The newly created session.
        """
        ...

    @abstractmethod
    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> Session | None:
        """Retrieve a session by its identifiers.

        Parameters
        ----------
        app_name : str
            Application namespace the session belongs to.
        user_id : str
            The user who owns the session.
        session_id : str
            The unique session identifier.

        Returns
        -------
        Session or None
            The matching session, or None if not found.
        """
        ...

    async def append_event(self, session: Session, event: Event) -> Event:
        """Apply an event's side-effects and append it to the session.

        Applies ``event.actions.state_delta`` to ``session.state``,
        then appends the event to ``session.events``, and updates
        ``session.last_update_time`` to the current time.

        Parameters
        ----------
        session : Session
            The session to update.
        event : Event
            The event to apply and append.

        Returns
        -------
        Event
            The same event that was appended.
        """
        if event.partial:
            return event
        if event.actions.state_delta:
            session.state.update(event.actions.state_delta)
        session.events.append(event)
        session.last_update_time = time.time()
        return event

    @abstractmethod
    async def update_session(
        self,
        session_id: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> Session:
        """Persist state changes back to the session store.

        Parameters
        ----------
        session_id : str
            The unique session identifier.
        state : dict[str, Any], optional
            State key-value pairs to merge into the session.

        Returns
        -------
        Session
            The updated session.
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session from the store.

        Parameters
        ----------
        session_id : str
            The unique session identifier to delete.
        """
        ...

    @abstractmethod
    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: str,
    ) -> list[Session]:
        """List all sessions for a given app and user.

        Parameters
        ----------
        app_name : str
            Application namespace to filter by.
        user_id : str
            User ID to filter by.

        Returns
        -------
        list[Session]
            All matching sessions.
        """
        ...
