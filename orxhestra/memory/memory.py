"""Memory model — a single persisted recall entry.

A :class:`Memory` is an opaque serializable blob agents can save and
retrieve by semantic search.  Only ``content`` is required;
everything else is optional metadata the backing
:class:`~orxhestra.memory.base_memory_service.BaseMemoryService` may
use to filter, rank, or display results.

Memories are intentionally schema-lite.  Richer shapes (tagged
entities, typed facts, provenance chains) belong in a purpose-built
store; this type is the default-good-enough envelope the built-in
backends and the ``save_memory`` tool exchange.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory entry persisted across sessions.

    Memories are opaque blobs of text that agents can save and retrieve
    by semantic search. Only ``content`` is required — everything else
    is optional metadata the memory service may use to filter, rank,
    or display results.

    Attributes
    ----------
    content : str
        The main content of the memory.
    metadata : dict[str, Any]
        Optional metadata associated with the memory. Keys are
        service-specific; common keys include ``source`` and ``tags``.
    id : str, optional
        The unique identifier of the memory. Typically assigned by the
        memory service on save.
    author : str, optional
        The author of the memory (often an agent or user name).
    timestamp : str, optional
        The timestamp when the original content of this memory
        happened. This string is forwarded to the LLM at recall time.
        Preferred format is ISO 8601.
    """

    content: str
    """The main content of the memory."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Optional metadata associated with the memory."""

    id: str | None = None
    """The unique identifier of the memory."""

    author: str | None = None
    """The author of the memory."""

    timestamp: str | None = None
    """The timestamp when the original content of this memory happened.

    This string will be forwarded to LLM. Preferred format is ISO 8601 format.
    """
