"""ReadonlyContext and CallbackContext - scoped views over InvocationContext.

Two context types are provided for different call sites:

ReadonlyContext
    A read-only snapshot passed to instruction providers and planners. State
    is exposed as an immutable ``MappingProxyType`` to prevent accidental
    mutations during the planning/instruction-building phase.

CallbackContext
    A mutable context passed to before/after callbacks. Exposes a writable
    state dict and a local ``EventActions`` instance so callbacks can apply
    state deltas and signal side-effects without coupling to the full
    InvocationContext.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from orxhestra.events.event_actions import EventActions

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext


class ReadonlyContext:
    """Read-only view over a InvocationContext.

    Passed to instruction providers (callable instructions) and planners
    so they can read session state and metadata without being able to
    mutate the context directly.

    Parameters
    ----------
    invocation_context : InvocationContext
        The underlying invocation context.

    Attributes
    ----------
    invocation_id : str
        Unique ID for this agent invocation.
    session_id : str
        The session this invocation belongs to.
    user_id : str
        The user who initiated the session.
    app_name : str
        The application running the agent.
    agent_name : str
        The name of the currently executing agent.
    state : MappingProxyType[str, Any]
        Read-only view of the invocation state.
    """

    def __init__(self, invocation_context: InvocationContext) -> None:
        self._ctx = invocation_context

    @property
    def invocation_id(self) -> str:
        """Unique ID for this agent invocation."""
        return self._ctx.invocation_id

    @property
    def session_id(self) -> str:
        """The session this invocation belongs to."""
        return self._ctx.session_id

    @property
    def user_id(self) -> str:
        """The user who initiated the session."""
        return self._ctx.user_id

    @property
    def app_name(self) -> str:
        """The application running the agent."""
        return self._ctx.app_name

    @property
    def agent_name(self) -> str:
        """The name of the currently executing agent."""
        return self._ctx.agent_name

    @property
    def branch(self) -> str:
        """Dot-separated branch path for this agent in the tree."""
        return self._ctx.branch

    @property
    def state(self) -> MappingProxyType[str, Any]:
        """Read-only snapshot of the invocation state."""
        return MappingProxyType(self._ctx.state)


class CallbackContext(ReadonlyContext):
    """Mutable context passed to before/after callbacks.

    Extends ``ReadonlyContext`` with a writable state dict and a local
    ``EventActions`` instance. The agent inspects ``actions`` after the
    callback returns to apply any requested side-effects.

    Parameters
    ----------
    invocation_context : InvocationContext
        The underlying invocation context.

    Attributes
    ----------
    state : dict[str, Any]
        Mutable reference to the invocation state. Changes are immediately
        visible to all subsequent agents sharing the same context.
    actions : EventActions
        Side-effects the callback wants to signal: escalate, transfer,
        state_delta, etc.
    """

    def __init__(self, invocation_context: InvocationContext) -> None:
        super().__init__(invocation_context)
        self.actions: EventActions = EventActions()

    @property
    def state(self) -> dict[str, Any]:  # type: ignore[override]
        """Mutable reference to the invocation state."""
        return self._ctx.state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Merge ``value`` into the invocation state (in place).

        Parameters
        ----------
        value : dict[str, Any]
            Keys to merge into the existing state dict. Existing keys
            are overwritten; missing keys are preserved. The state
            dict itself is not reassigned — callers still observe
            their shared reference.
        """
        self._ctx.state.update(value)
