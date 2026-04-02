"""EventActions and EventCompaction - side-effects attached to events.

Every ``Event`` carries an ``EventActions`` instance that the session service
and composite agents inspect after each event to apply state changes, route
control, compact history, or signal loop termination.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventCompaction(BaseModel):
    """A compacted summary covering a range of session events.

    When a session history grows too long for the model context window, old
    events are summarized into a compaction event appended to the session.
    Original events are preserved; the ``apply_compaction`` view-layer filter
    hides them when building LLM context.

    Attributes
    ----------
    start_timestamp : float
        Unix timestamp of the earliest event in the compacted range.
    end_timestamp : float
        Unix timestamp of the latest event in the compacted range.
    summary : str
        LLM-generated prose summary of the compacted events.
    event_count : int
        Number of original events collapsed into this compaction.
    """

    start_timestamp: float
    end_timestamp: float
    summary: str
    event_count: int = 0


class EventActions(BaseModel):
    """Side-effects declared on an event.

    Composite agents and the session service inspect these after each event
    to apply state changes, route control to another agent, stop a loop,
    or compact old session history.

    Attributes
    ----------
    state_delta : dict[str, Any]
        Key-value updates merged into ``InvocationContext.state`` (and
        persisted to ``Session.state``) when the event is committed via
        ``append_event()``.
    artifact_delta : dict[str, int]
        Maps artifact filenames to the version number saved during this
        event.  Populated by ``CallContext.save_artifact()`` so that
        artifact changes are visible in the session event log.
    transfer_to_agent : str, optional
        Name of the agent to hand control to after this event. The parent
        agent intercepts this to perform the handoff.
    escalate : bool, optional
        When ``True``, signals the parent ``LoopAgent`` to stop iterating.
        Set by the ``exit_loop`` tool.
    skip_summarization : bool, optional
        When ``True``, instructs the parent to pass the tool result through
        to the user directly without an LLM summarization step.
    end_of_agent : bool, optional
        When ``True``, marks that the current agent has finished its turn.
        Unlike ``escalate``, this does not stop a ``LoopAgent`` - it just
        signals a natural agent boundary to the runtime.
    compaction : EventCompaction, optional
        Present when this event represents a compacted summary replacing
        a range of older session events.
    """

    state_delta: dict[str, Any] = Field(default_factory=dict)
    artifact_delta: dict[str, int] = Field(default_factory=dict)
    transfer_to_agent: str | None = None
    escalate: bool | None = None
    skip_summarization: bool | None = None
    end_of_agent: bool | None = None
    compaction: EventCompaction | None = None
