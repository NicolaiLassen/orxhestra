"""Sessions — durable conversation state across agent turns.

A :class:`Session` holds the ordered :class:`Event` log for a
(``app_name``, ``user_id``, ``session_id``) triple plus a mutable
``state`` dict that agents can read and write through
``EventActions.state_delta``.  The :class:`Runner` fetches or creates
the session for every invocation, applies events to it via
:meth:`BaseSessionService.append_event`, and passes the populated
session through to each agent in the tree — which is how multi-turn
context, transfer-based sub-agent resumption, and persistence-backed
workflows all work.

Backends:

- :class:`InMemorySessionService` — dict-backed, process-local.  Use
  for tests, single-shot CLI runs, and the built-in ``orx`` REPL.
- :class:`DatabaseSessionService` — SQLAlchemy async backend.
  Suitable for production deployments that need checkpointing and
  resume across processes.  Supports any async SQLAlchemy driver
  (SQLite via ``aiosqlite``, PostgreSQL via ``asyncpg``, ...).

Compaction:

- :func:`compact_session` + :class:`CompactionConfig` — summarise old
  events with an LLM when the session grows beyond a character
  threshold, preserving recent events verbatim.  Invoked
  automatically by the :class:`Runner` when ``compaction_config`` is
  set, or on demand via the ``/compact`` slash command in the CLI.

See Also
--------
orxhestra.events.event.Event : The unit stored in a session's log.
orxhestra.runner.Runner : Primary consumer; persists every emitted
    event and propagates state across turns.
"""

from orxhestra.sessions.base_session_service import BaseSessionService
from orxhestra.sessions.compaction import CompactionConfig, compact_session
from orxhestra.sessions.database_session_service import DatabaseSessionService
from orxhestra.sessions.in_memory_session_service import InMemorySessionService
from orxhestra.sessions.session import Session

__all__ = [
    "BaseSessionService",
    "CompactionConfig",
    "DatabaseSessionService",
    "InMemorySessionService",
    "Session",
    "compact_session",
]
