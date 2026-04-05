"""PlannerAdapter — centralises all planner interactions for LlmAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage, HumanMessage

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse
    from orxhestra.planners.base_planner import BasePlanner


class PlannerAdapter:
    """Thin facade over a ``BasePlanner``.

    Consolidates the three planner touch-points that were previously
    scattered across ``LlmAgent`` into a single object:

    * Enriching the system prompt before each LLM call.
    * Post-processing the LLM response.
    * Deciding whether the agent loop should continue.

    Parameters
    ----------
    planner : BasePlanner
        The underlying planner strategy.
    """

    def __init__(self, planner: BasePlanner) -> None:
        self._planner = planner

    # -- Expose underlying planner for subclass access (e.g. ReActAgent) --

    @property
    def planner(self) -> BasePlanner:
        """Return the wrapped planner instance."""
        return self._planner

    # -- Public helpers used by LlmAgent ----------------------------------

    def enrich_prompt(
        self,
        ctx: InvocationContext,
        base_prompt: str,
        request: LlmRequest,
    ) -> str:
        """Append the planner's instruction to *base_prompt*.

        Parameters
        ----------
        ctx : InvocationContext
            Current invocation context.
        base_prompt : str
            The system prompt before planner enrichment.
        request : LlmRequest
            The request being built for the LLM.

        Returns
        -------
        str
            Enriched prompt, or *base_prompt* unchanged when the
            planner returns nothing.
        """
        from orxhestra.agents.readonly_context import ReadonlyContext

        readonly = ReadonlyContext(ctx)
        instruction = self._planner.build_planning_instruction(readonly, request)
        if instruction:
            return f"{base_prompt}\n\n{instruction}"
        return base_prompt

    def process_response(
        self,
        ctx: InvocationContext,
        response: LlmResponse,
    ) -> LlmResponse:
        """Let the planner optionally transform the LLM response.

        Parameters
        ----------
        ctx : InvocationContext
            Current invocation context.
        response : LlmResponse
            The raw LLM response.

        Returns
        -------
        LlmResponse
            The (possibly replaced) response.
        """
        from orxhestra.agents.readonly_context import ReadonlyContext

        readonly = ReadonlyContext(ctx)
        replacement = self._planner.process_planning_response(readonly, response)
        return replacement if replacement is not None else response

    def should_continue(
        self,
        ctx: InvocationContext,
        messages: list[BaseMessage],
    ) -> bool:
        """Check if the planner still has pending work.

        When ``True``, a continuation prompt is appended to *messages*
        so the LLM keeps working.

        Parameters
        ----------
        ctx : InvocationContext
            Current invocation context.
        messages : list[BaseMessage]
            Conversation messages (mutated in-place when continuing).

        Returns
        -------
        bool
            ``True`` if the loop should continue, ``False`` if the
            agent may return its final answer.
        """
        from orxhestra.agents.readonly_context import ReadonlyContext

        readonly = ReadonlyContext(ctx)
        if self._planner.has_pending_tasks(readonly):
            messages.append(
                HumanMessage(
                    content=(
                        "You still have pending tasks on the task board. "
                        "Continue working on the next task — call the "
                        "appropriate tools to make progress."
                    )
                )
            )
            return True
        return False
