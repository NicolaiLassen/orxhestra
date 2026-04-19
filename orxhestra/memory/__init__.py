"""Memory — persistent recall across sessions.

A memory service is a long-term store that outlives a single
:class:`~orxhestra.sessions.session.Session`.  Agents search memory
at the start of a turn to pull in relevant context, and write memory
during or after a turn to persist new information.

Backends shipped here:

- :class:`InMemoryMemoryService` — dict-backed, process-local.  Use
  for tests and single-shot REPL runs.
- :class:`~orxhestra.memory.file_memory_service.FileMemoryService` —
  Markdown files on disk with YAML frontmatter, scoped per project
  directory.  Powers the auto-memory feature of the ``orx`` CLI.

The agent-facing primitives live in :mod:`orxhestra.tools.memory_tools`
(``save_memory`` / ``load_memory``) — use those rather than calling
the service directly from an agent.

See Also
--------
orxhestra.tools.memory_tools : Tool factories the LLM uses.
orxhestra.sessions.session.Session : Short-lived conversation state;
    the complement to this longer-lived memory store.
"""

from orxhestra.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from orxhestra.memory.in_memory_service import InMemoryMemoryService
from orxhestra.memory.memory import Memory

__all__ = [
    "BaseMemoryService",
    "InMemoryMemoryService",
    "Memory",
    "SearchMemoryResponse",
]
