"""PlanReActPlanner - structured chain-of-thought planning via tags.

Before each LLM call, appends a multi-section planning instruction that
requires the model to:
  1. Produce a plan before acting (/*PLANNING*/ tag).
  2. Reason between steps (/*REASONING*/ tag).
  3. Call tools under /*ACTION*/ tags.
  4. Deliver a final answer under /*FINAL_ANSWER*/.

The LlmAgent picks up the /*FINAL_ANSWER*/ section as the response text,
and reasoning/planning sections are treated as thought events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orxhestra.planners.base_planner import BasePlanner

if TYPE_CHECKING:
    from orxhestra.agents.readonly_context import ReadonlyContext
    from orxhestra.models.llm_request import LlmRequest


PLANNING_TAG = "/*PLANNING*/"
REPLANNING_TAG = "/*REPLANNING*/"
REASONING_TAG = "/*REASONING*/"
ACTION_TAG = "/*ACTION*/"
FINAL_ANSWER_TAG = "/*FINAL_ANSWER*/"


def _build_instruction() -> str:
    preamble = f"""\
When answering, use the available tools rather than memorised knowledge.

Follow this process:
1. Form a plan in natural language.
2. Execute the plan step by step, reasoning between tool calls.
3. Return a single final answer.

Use these tags to structure your response:
- Plan: {PLANNING_TAG}
- Reasoning between steps: {REASONING_TAG}
- Tool invocations: {ACTION_TAG}
- Final answer: {FINAL_ANSWER_TAG}
- If the initial plan fails, revise it under {REPLANNING_TAG}."""

    planning_req = f"""\
Planning requirements:
- The plan covers all aspects of the query and only references available tools.
- Each step in the plan maps to one or more tool calls.
- If the plan cannot be executed, revise it under {REPLANNING_TAG}."""

    reasoning_req = """\
Reasoning requirements:
- Summarise what you know so far and what remains.
- State the next concrete action based on the plan and previous tool outputs."""

    final_answer_req = """\
Final answer requirements:
- Be precise and follow any formatting requirements in the query.
- If the query cannot be answered with the available tools, explain why
  and ask for clarification."""

    return "\n\n".join([preamble, planning_req, reasoning_req, final_answer_req])


_INSTRUCTION = _build_instruction()


class PlanReActPlanner(BasePlanner):
    """Injects structured Plan-Re-Act instructions before each LLM call.

    Requires the model to produce a plan, then reason and act
    step-by-step, and finally deliver the answer under a
    ``/*FINAL_ANSWER*/`` tag. The tag structure is lightweight and
    works with any LangChain chat model.

    See Also
    --------
    BasePlanner : Interface this implements.
    TaskPlanner : Alternative planner reading from a shared TodoList.
    ReActAgent : Pairs well with this planner for structured reasoning.

    Examples
    --------
    >>> from orxhestra.planners import PlanReActPlanner
    >>> agent = LlmAgent(
    ...     name="analyst",
    ...     model=model,
    ...     tools=[search_tool],
    ...     planner=PlanReActPlanner(),
    ... )
    """

    def build_planning_instruction(
        self,
        readonly_context: ReadonlyContext,
        llm_request: LlmRequest,
    ) -> str | None:
        """Return the Plan-Re-Act structured instruction.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            The current invocation context (read-only).
        llm_request : LlmRequest
            The LLM request being built for this turn.

        Returns
        -------
        str
            The planning instruction string.
        """
        return _INSTRUCTION
