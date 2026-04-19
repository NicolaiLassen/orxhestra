"""Abstract session service — the persistence seam for conversations.

Implementations store and retrieve :class:`Session` objects along
with the event log each session carries.  The runner calls into this
interface on every turn; any async backend that can round-trip a
:class:`Session` works (SQL, key/value stores, cloud databases).

:meth:`BaseSessionService.append_event` provides a default
implementation that applies ``event.actions.state_delta`` to the
session and appends the event — subclasses only need to implement
the five abstract methods (``create_session``, ``get_session``,
``update_session``, ``delete_session``, ``list_sessions``) unless
they want to override persistence of append semantics.

Built-in backends:

- :class:`~orxhestra.sessions.in_memory_session_service.InMemorySessionService` —
  dict-backed, process-local.
- :class:`~orxhestra.sessions.database_session_service.DatabaseSessionService` —
  SQLAlchemy async backend (SQLite, Postgres, MySQL, ...).

See Also
--------
orxhestra.sessions.session.Session : The unit this service stores.
orxhestra.runner.Runner : Primary consumer.
"""

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
