"""In-memory session service — dict-backed, process-local.

Stores every :class:`~orxhestra.sessions.session.Session` in a plain
``dict`` keyed by ``session_id``.  State is lost on process restart,
which is fine for:

- ``orx`` REPL invocations (the CLI lives for one interactive session).
- Unit and integration tests.
- Single-shot automation (``orx -c "..."``).

For anything that needs to resume across processes, use
:class:`~orxhestra.sessions.database_session_service.DatabaseSessionService`.

See Also
--------
orxhestra.sessions.base_session_service.BaseSessionService :
    Interface this implements.
"""

from __future__ import annotations

import time
from typing import Any

from orxhestra.errors.session_not_found_error import SessionNotFoundError
from orxhestra.sessions.base_session_service import BaseSessionService
from orxhestra.sessions.session import Session


class InMemorySessionService(BaseSessionService):
    """Dict-backed session service. State is lost on process restart."""

    def __init__(self) -> None:
        self._store: dict[str, Session] = {}

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> Session:
        """Create and store a new in-memory session.

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
        session = Session(
            app_name=app_name,
            user_id=user_id,
            state=state or {},
            last_update_time=time.time(),
        )
        if session_id:
            session = session.model_copy(update={"id": session_id})
        self._store[session.id] = session
        return session

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
        session = self._store.get(session_id)
        if session is None:
            return None
        if session.app_name != app_name or session.user_id != user_id:
            return None
        return session

    async def update_session(
        self,
        session_id: str,
        *,
        state: dict[str, Any] | None = None,
    ) -> Session:
        """Merge state updates into the stored session.

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
        session = self._store.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        updates: dict[str, Any] = {
            "last_update_time": time.time(),
        }
        if state is not None:
            updates["state"] = {**session.state, **state}
        updated = session.model_copy(update=updates)
        self._store[session_id] = updated
        return updated

    async def delete_session(self, session_id: str) -> None:
        """Remove a session from the in-memory store.

        Parameters
        ----------
        session_id : str
            The unique session identifier to delete.
        """
        self._store.pop(session_id, None)

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
        return [
            s for s in self._store.values()
            if s.app_name == app_name and s.user_id == user_id
        ]
