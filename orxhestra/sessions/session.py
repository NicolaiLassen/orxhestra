"""Session model — durable state for a multi-turn agent conversation.

A :class:`Session` is the single source of truth for what happened
in a conversation: its ordered :class:`~orxhestra.events.event.Event`
log, a mutable ``state`` dict that agents read and write via
``EventActions.state_delta``, and the timestamps / ownership metadata
that let a :class:`~orxhestra.sessions.base_session_service.BaseSessionService`
look it up again later.

The model is intentionally serializable (plain :mod:`pydantic`) so it
can round-trip through any backend — in-memory, SQL, cache.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from orxhestra.events.event import Event


class Session(BaseModel):
    """A single conversation session between a user and an agent.

    Holds the full event history and any persisted state from the
    agent's execution. Designed to be serializable — pass
    :attr:`Session.state` back into the next invocation to resume
    where the agent left off.

    Attributes
    ----------
    id : str
        Unique session identifier.
    app_name : str
        Application namespace (used to scope sessions).
    user_id : str
        The user who owns this session.
    state : dict[str, Any]
        Arbitrary key-value state persisted across turns.
    events : list[Event]
        Ordered list of :class:`Event` objects in this conversation.
    last_update_time : float
        Unix timestamp of the last update.

    See Also
    --------
    BaseSessionService : Persistence interface over this model.
    InMemorySessionService : Non-persistent backend.
    DatabaseSessionService : SQLAlchemy-backed persistent backend.
    Runner : Uses sessions for multi-turn state continuity.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    app_name: str
    user_id: str
    state: dict[str, Any] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    last_update_time: float = 0.0
