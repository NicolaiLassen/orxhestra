"""Base memory service abstraction for agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from orxhestra.memory.memory import Memory


class SearchMemoryResponse(BaseModel):
    """Represents the response from a memory search.

    Attributes
    ----------
    memories : list[Memory]
        A list of memories that relate to the search query.
    """

    memories: list[Memory]
    """A list of memories that relate to the search query."""


class BaseMemoryService(ABC):
    """Abstract base class for agent memory services.

    Attributes
    ----------
    memories : list[Memory]
        The list of memories held in this service.
    """

    memories: list[Memory] = []

    @abstractmethod
    async def add_session_to_memory(
        self,
        session: Any,
    ) -> None:
        """Add a session to the memory service.

        A session may be added multiple times during its lifetime.

        Parameters
        ----------
        session : Any
            The session to add.
        """
        ...

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Any],
        session_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Add an explicit list of events to the memory service.

        Implementations should treat events as an incremental update (delta) and
        must not assume it represents the full session.

        Parameters
        ----------
        app_name : str
            The application name for memory scope.
        user_id : str
            The user ID for memory scope.
        events : Sequence[Any]
            The events to add to memory.
        session_id : str, optional
            Optional session ID for memory scope/partitioning.
        metadata : Mapping[str, object], optional
            Optional metadata for memory generation.

        Raises
        ------
        NotImplementedError
            If this memory service does not support adding event deltas.
        """
        raise NotImplementedError(
            "This memory service does not support adding event deltas. "
            "Call add_session_to_memory(session) to ingest the full session."
        )

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[Memory],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Add explicit memory items directly to the memory service.

        Parameters
        ----------
        app_name : str
            The application name for memory scope.
        user_id : str
            The user ID for memory scope.
        memories : Sequence[Memory]
            Explicit memory items to add.
        metadata : Mapping[str, object], optional
            Optional metadata for memory writes.

        Raises
        ------
        NotImplementedError
            If this memory service does not support direct memory writes.
        """
        raise NotImplementedError(
            "This memory service does not support direct memory writes. "
            "Call add_events_to_memory(...) or add_session_to_memory(session) instead."
        )

    @abstractmethod
    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search for memories that match the query.

        Parameters
        ----------
        app_name : str
            The name of the application.
        user_id : str
            The id of the user.
        query : str
            The query to search for.

        Returns
        -------
        SearchMemoryResponse
            A SearchMemoryResponse containing the matching memories.
        """
        ...
