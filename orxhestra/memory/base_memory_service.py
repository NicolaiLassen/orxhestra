"""Abstract memory service — the persistence seam for long-term recall.

Implementations store and search :class:`Memory` entries scoped by
``(app_name, user_id)``.  Agents hit this interface twice per turn:
once at the start to pull in relevant context
(:meth:`search_memory`), once after the agent finishes to persist new
information (:meth:`add_session_to_memory`, :meth:`add_events_to_memory`,
or :meth:`add_memory`).

Three ingestion shapes are supported so backends can pick the
granularity that suits them:

- :meth:`add_session_to_memory` — full session; required.
- :meth:`add_events_to_memory` — delta batch; optional (default
  raises ``NotImplementedError``).
- :meth:`add_memory` — explicit :class:`Memory` entries; optional.

Built-in backends:

- :class:`~orxhestra.memory.in_memory_service.InMemoryMemoryService`
  — dict-backed, process-local.
- :class:`~orxhestra.memory.file_memory_service.FileMemoryService`
  — Markdown-on-disk, per-project directories.

See Also
--------
orxhestra.memory.memory.Memory : The unit stored in a memory service.
orxhestra.tools.memory_tools : ``save_memory`` / ``load_memory`` tool
    factories — the primary way agents interact with memory.
"""

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

    Memory services are long-term stores that outlive a single
    session. Agents search memory at the start of a turn to recall
    relevant context, and write memory after a turn to persist new
    information.

    Attributes
    ----------
    memories : list[Memory]
        The list of memories held in this service.

    See Also
    --------
    Memory : Individual memory entry.
    SearchMemoryResponse : Return type of :meth:`search_memory`.
    InMemoryMemoryService : Non-persistent backend.
    FileMemoryService : Markdown-on-disk backend.
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
