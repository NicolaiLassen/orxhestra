"""BasePlanner - abstract interface for agent planning strategies.

A planner hooks into the LlmAgent's tool-call loop at two points:

1. Before each LLM call: ``build_planning_instruction()`` returns an
   optional string that is appended to the system prompt. Use this to
   inject planning frameworks, task status, or chain-of-thought guidance.

2. After each LLM response: ``process_planning_response()`` may inspect or
   transform the LlmResponse before the agent acts on it. Return None to
   leave the response unchanged.

Attach a planner to an LlmAgent via the ``planner`` constructor argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orxhestra.agents.readonly_context import ReadonlyContext
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse


class BasePlanner(ABC):
    """Abstract base class for all planners.

    A planner allows planning logic to be attached to an
    :class:`LlmAgent` without embedding that logic in the agent
    itself. Subclass and implement :meth:`build_planning_instruction`
    at minimum.

    See Also
    --------
    TaskPlanner : Built-in planner backed by a shared ``TodoList``.
    PlanReActPlanner : Built-in Plan-ReAct style planner.
    LlmAgent : Pass the planner via the ``planner`` constructor arg.
    """

    @abstractmethod
    def build_planning_instruction(
        self,
        readonly_context: ReadonlyContext,
        llm_request: LlmRequest,
    ) -> str | None:
        """Return an instruction string to append to the system prompt.

        Called before every LLM invocation. The returned string is appended
        to the agent's resolved system instructions for that turn only.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            A read-only view of the current invocation context.
        llm_request : LlmRequest
            The LLM request being built for this turn. Inspect to make
            context-sensitive planning decisions.

        Returns
        -------
        str or None
            Planning instruction to append, or None if no instruction is
            needed for this turn.
        """

    def has_pending_tasks(self, readonly_context: ReadonlyContext) -> bool:
        """Check whether the planner still has incomplete tasks.

        Called by the agent after a no-tool-call response to decide whether
        to continue the loop or return. The default returns ``False``,
        meaning the agent always treats text-only responses as final.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            A read-only view of the current invocation context.

        Returns
        -------
        bool
            True if there are pending tasks that should keep the loop going.
        """
        return False

    def process_planning_response(
        self,
        readonly_context: ReadonlyContext,
        response: LlmResponse,
    ) -> LlmResponse | None:
        """Optionally transform the LLM response after it arrives.

        Called after every LLM invocation, before the agent acts on the
        response. Return None to leave the response unchanged.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            A read-only view of the current invocation context.
        response : LlmResponse
            The model response from this turn.

        Returns
        -------
        LlmResponse or None
            A replacement response, or None to leave the response unchanged.
        """
        return None
