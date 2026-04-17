"""InMemoryMemoryService - simple in-process memory implementation."""

from __future__ import annotations

from orxhestra.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from orxhestra.memory.memory import Memory


class InMemoryMemoryService(BaseMemoryService):
    """Dict-backed memory service.

    Stores memories keyed by ``(app_name, user_id)``. Search is
    naive substring matching — swap for a vector store in production.

    See Also
    --------
    BaseMemoryService : Interface this implements.
    FileMemoryService : Markdown-on-disk alternative.
    Memory : Individual memory entry stored here.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[Memory]] = {}

    def _key(self, app_name: str, user_id: str) -> tuple[str, str]:
        """Build a compound store key from app_name and user_id."""
        return (app_name, user_id)

    async def add_session_to_memory(self, session: object) -> None:
        """Extract events from a Session and store them as memories.

        Stores AGENT_MESSAGE events that have text content — this covers
        final answers and react thoughts.

        Parameters
        ----------
        session : object
            The session to extract events from. Must be a
            ``sessions.session.Session`` instance.
        """
        from orxhestra.events.event import EventType
        from orxhestra.sessions.session import Session

        if not isinstance(session, Session):
            return

        key = self._key(session.app_name, session.user_id)
        if key not in self._store:
            self._store[key] = []

        for event in session.events:
            if event.type != EventType.AGENT_MESSAGE:
                continue
            if event.partial:
                continue
            content = event.text
            author = event.agent_name or event.author or "agent"

            if content:
                self._store[key].append(
                    Memory(
                        content=content,
                        author=author,
                        metadata={"session_id": session.id},
                    )
                )

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Substring search over stored memories.

        Parameters
        ----------
        app_name : str
            The application name for memory scope.
        user_id : str
            The user ID for memory scope.
        query : str
            The substring to search for (case-insensitive).

        Returns
        -------
        SearchMemoryResponse
            All memories whose content contains the query string.
        """
        key = self._key(app_name, user_id)
        memories = self._store.get(key, [])
        query_lower = query.lower()
        matches = [
            m for m in memories
            if query_lower in m.content.lower()
        ]
        return SearchMemoryResponse(memories=matches)
