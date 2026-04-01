"""Agent context — propagated through agent call chains.

Carries runtime state for a single agent invocation: who is running,
which session it belongs to, what ephemeral state has accumulated,
and how the run is configured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.artifacts.base_artifact_service import BaseArtifactService


class InvocationContext(BaseModel):
    """Runtime context propagated through an agent's execution tree.

    Each agent invocation receives a context.  Child agents (sub-agents,
    AgentTool) receive a derived context via ``derive()`` with an updated
    branch and agent_name — enabling event attribution across the hierarchy.

    Attributes
    ----------
    invocation_id : str
        Unique ID for this specific invocation.
    session_id : str
        The session this invocation belongs to.
    user_id : str
        The user who initiated the session.
    app_name : str
        The application running the agent.
    agent_name : str
        The name of the currently executing agent.
    branch : str
        Dot-separated path for nested execution (e.g. ``"root.child"``).
        Set automatically by orchestration agents via ``derive()``.
        Events carry this field so consumers know which sub-agent
        produced each event as it bubbles up.
    state : dict[str, Any]
        Mutable key-value store for cross-agent state within one run.
        Agents may read/write this freely; changes are not persisted
        automatically — the Runner handles persistence.
    agent_states : dict[str, dict[str, Any]]
        Per-agent execution state.  Used by composite agents (e.g.
        ``LoopAgent``) to track which step a child agent was on so it
        can be reset between iterations.
    end_of_agents : dict[str, bool]
        Tracks which agents have finished their turn.  Unlike
        ``escalate``, this does not stop a ``LoopAgent``.
    end_invocation : bool
        Kill switch.  Set to ``True`` to force-stop the entire
        invocation immediately.  All agents — including composite
        parents — should check this flag and stop yielding events.
    is_resumable : bool
        When ``True``, enables pause/resume for long-running tools.
    long_running_tool_ids : set[str]
        Tool call IDs that are long-running.  When a tool call's ID
        appears here, ``should_pause_invocation()`` returns ``True``
        so the runner can persist state and resume later.
    input_content : str, optional
        The input that initiated this invocation — either the user's
        message or the output from a previous agent in a pipeline.
        Preserved for resumption and for agents that need to refer
        back to the original request.
    session : Session, optional
        The session this invocation belongs to.
    run_config : dict[str, Any]
        Per-run configuration (LangChain RunnableConfig / AgentConfig).
    current_agent : BaseAgent, optional
        Reference to the current agent instance.  Used by
        ``reset_sub_agent_states`` to walk the agent tree via
        ``find_agent()``.
    memory_service : Any, optional
        Optional memory service for long-term recall.
    event_callback : callable, optional
        Callback for real-time event streaming to parent agents.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    invocation_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str = ""
    app_name: str = ""
    agent_name: str
    branch: str = ""
    state: dict[str, Any] = Field(default_factory=dict)
    agent_states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    end_of_agents: dict[str, bool] = Field(default_factory=dict)
    end_invocation: bool = False
    is_resumable: bool = False
    long_running_tool_ids: set[str] = Field(default_factory=set)
    input_content: str | None = None
    session: Any | None = None
    run_config: dict[str, Any] = Field(default_factory=dict)
    current_agent: BaseAgent | None = None
    artifact_service: BaseArtifactService | None = None
    memory_service: Any | None = None
    event_callback: Callable[[Any], None] | None = None

    def derive(
        self,
        *,
        agent_name: str,
        branch_suffix: str = "",
    ) -> InvocationContext:
        """Create a child context for a sub-agent invocation.

        The child shares the same session, invocation ID, state reference,
        agent_states, and run_config, but gets an updated agent_name and
        branch for attribution and isolation.

        Parameters
        ----------
        agent_name : str
            The name of the child agent.
        branch_suffix : str, optional
            Optional suffix appended to the branch path.
            Defaults to agent_name.

        Returns
        -------
        Context
            A new Context scoped to the child agent.
        """
        suffix = branch_suffix or agent_name
        new_branch = f"{self.branch}.{suffix}" if self.branch else suffix
        return self.model_copy(
            update={
                "agent_name": agent_name,
                "branch": new_branch,
                "state": self.state,
                "agent_states": self.agent_states,
                "end_of_agents": self.end_of_agents,
                "long_running_tool_ids": self.long_running_tool_ids,
                "session": self.session,
                "run_config": self.run_config,
                "current_agent": self.current_agent,
                "artifact_service": self.artifact_service,
                "memory_service": self.memory_service,
                "event_callback": self.event_callback,
            }
        )

    def clear_session(self) -> InvocationContext:
        """Return a copy with an empty session (no conversation history).

        The new session keeps the same IDs (session_id, app_name,
        user_id) but has no events or prior state — giving the
        consumer a clean slate.  ``ctx.state`` is preserved.

        Returns
        -------
        Context
            A new Context with an empty session.
        """
        if self.session is None:
            return self.model_copy()

        from orxhestra.sessions.session import Session

        return self.model_copy(
            update={
                "session": Session(
                    id=self.session_id,
                    app_name=self.app_name,
                    user_id=self.user_id,
                ),
            }
        )

    def get_events(
        self,
        *,
        current_branch: bool = False,
        current_invocation: bool = False,
    ) -> list[Any]:
        """Return session events with optional filtering.

        Parameters
        ----------
        current_branch : bool
            If ``True``, only return events whose ``branch`` matches
            this context's branch exactly.
        current_invocation : bool
            If ``True``, only return events whose ``invocation_id``
            matches this context's invocation_id.

        Returns
        -------
        list[Event]
            Filtered list of session events.
        """
        if self.session is None:
            return []

        events = self.session.events

        if current_invocation:
            inv_id = self.invocation_id
            events = [e for e in events if e.invocation_id == inv_id]

        if current_branch:
            branch = self.branch
            events = [e for e in events if e.branch == branch]

        return events

    def set_agent_state(
        self,
        agent_name: str,
        *,
        agent_state: dict[str, Any] | None = None,
        end_of_agent: bool = False,
    ) -> None:
        """Set or clear the execution state for a named agent.

        Parameters
        ----------
        agent_name : str
            The agent whose state to set.
        agent_state : dict, optional
            The state to persist.  If ``None`` and ``end_of_agent`` is
            ``False``, the state is cleared entirely.
        end_of_agent : bool
            If ``True``, marks the agent as finished and clears its state.
        """
        if end_of_agent:
            self.end_of_agents[agent_name] = True
            self.agent_states.pop(agent_name, None)
        elif agent_state is not None:
            self.agent_states[agent_name] = agent_state
            self.end_of_agents[agent_name] = False
        else:
            # Clear both — allows the agent to re-run from scratch
            self.end_of_agents.pop(agent_name, None)
            self.agent_states.pop(agent_name, None)

    def reset_sub_agent_states(self, agent_name: str) -> None:
        """Recursively reset the state of all sub-agents of the named agent.

        Called by ``LoopAgent`` between iterations so that child agents
        start fresh each loop cycle.

        Parameters
        ----------
        agent_name : str
            The parent agent whose children should be reset.
        """
        if self.current_agent is None:
            return

        parent = self.current_agent.find_agent(agent_name)
        if parent is None:
            return

        for sub_agent in parent.sub_agents:
            self.set_agent_state(sub_agent.name)
            self.reset_sub_agent_states(sub_agent.name)

    def should_pause_invocation(self, event: Any) -> bool:
        """Return ``True`` if the invocation should pause after this event.

        Pausing is triggered when:
        1. ``is_resumable`` is ``True``
        2. The event contains function calls
        3. At least one function call ID is in ``long_running_tool_ids``

        The runner can then persist state and resume later.

        Parameters
        ----------
        event : Event
            The event to inspect.

        Returns
        -------
        bool
            ``True`` if the invocation should pause.
        """
        if not self.is_resumable:
            return False

        if not self.long_running_tool_ids:
            return False

        tool_calls = getattr(event, "tool_calls", None) or []
        for tc in tool_calls:
            tc_id = getattr(tc, "tool_call_id", None)
            if tc_id and tc_id in self.long_running_tool_ids:
                return True

        return False
