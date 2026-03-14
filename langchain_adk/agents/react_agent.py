"""ReActAgent - structured output Reason+Act loop.

A specialized variant of LlmAgent where the model explicitly outputs its
reasoning at each step. Uses LangChain's with_structured_output() to
force the LLM to produce either:
  - ReasonAndAct  - a thought + tool to call
  - FinalAnswer   - the loop terminates

This is the "thinking" agent for tasks that benefit from explicit
chain-of-thought before each action.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import (
    ActionEvent,
    ErrorEvent,
    Event,
    FinalAnswerEvent,
    ObservationEvent,
    ThoughtEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from langchain_adk.models.part import Content


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class ReActStep(BaseModel):
    """A single ReAct step — either reason+act or final answer.

    When ``action`` is set, the agent wants to call a tool.
    When ``answer`` is set, the loop terminates.

    Attributes
    ----------
    scratchpad : str
        Running notes accumulated across loop iterations.
    thought : str
        The agent's current reasoning about the problem.
    action : str, optional
        The name of the tool to call (None when giving final answer).
    action_input : str, optional
        The input to pass to the tool.
    answer : str, optional
        The final answer (None when calling a tool).
    """

    scratchpad: str = Field(description="Running notes and observations accumulated across steps")
    thought: str = Field(description="Current reasoning about the problem")
    action: Optional[str] = Field(default=None, description="Tool name to call, or null if giving final answer")
    action_input: Optional[str] = Field(default=None, description="Input for the tool, or null if giving final answer")
    answer: Optional[str] = Field(default=None, description="Final answer text, or null if calling a tool")

    @property
    def is_final(self) -> bool:
        return self.answer is not None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a reasoning agent. You solve problems step by step using the ReAct pattern.

On each step you MUST output a JSON object with:
- "scratchpad": your running notes across steps
- "thought": your current reasoning
- Either "action" + "action_input" to call a tool, OR "answer" to give a final answer.
- Set action/action_input to null when giving a final answer, and answer to null when calling a tool.

Available tools:
{tool_descriptions}

Rules:
- Always think before acting.
- Use your scratchpad to accumulate observations and notes across steps.
- Only produce a FinalAnswer when you are confident in the result.
- Never call a tool that is not listed above.
"""


class ReActAgent(BaseAgent):
    """Agent that uses explicit structured reasoning before each action.

    Uses LangChain with_structured_output() to enforce ReasonAndAct |
    FinalAnswer at every step. Ideal for tasks requiring transparent
    step-by-step reasoning.

    Attributes
    ----------
    llm : BaseChatModel
        The LangChain chat model to use.
    tools : list[BaseTool]
        Tools available to the agent.
    max_iterations : int
        Maximum ReAct loop iterations.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        *,
        description: str = "",
        max_iterations: int = 10,
    ) -> None:
        super().__init__(name=name, description=description)
        self._tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self._llm = llm.with_structured_output(
            schema=ReActStep,
            include_raw=False,
        )

    def _system_message(self) -> SystemMessage:
        """Build the system message with available tool descriptions."""
        tool_descriptions = "\n".join(
            f"  - {name}: {tool.description}"
            for name, tool in self._tools.items()
        ) or "  (no tools available)"
        return SystemMessage(
            content=_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        )

    async def run(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        """Run the ReAct loop with structured reasoning output.

        Parameters
        ----------
        input : str
            The user message or task.
        ctx : InvocationContext
            The invocation context.

        Yields
        ------
        Event
            ThoughtEvent, ActionEvent, ToolCallEvent, ToolResultEvent,
            ObservationEvent, FinalAnswerEvent, or ErrorEvent.
        """
        messages: list[BaseMessage] = [
            self._system_message(),
            HumanMessage(content=input),
        ]

        for _ in range(self.max_iterations):
            try:
                step: ReActStep = await self._llm.ainvoke(messages)
            except Exception as exc:
                yield ErrorEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
                return

            if step.is_final:
                yield FinalAnswerEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    content=Content.from_text(step.answer),
                    scratchpad=step.scratchpad,
                )
                return

            # --- Reason and Act branch ---
            yield ThoughtEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                content=Content.from_text(step.thought),
                scratchpad=step.scratchpad,
            )
            yield ActionEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                action=step.action,
                action_input=step.action_input,
            )

            tool = self._tools.get(step.action)
            if tool is None:
                observation = f"Error: tool '{step.action}' not found."
            else:
                yield ToolCallEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    tool_name=step.action,
                    tool_input=step.action_input,
                )
                try:
                    result = await tool.ainvoke(step.action_input)
                    observation = str(result)
                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=step.action,
                        content=Content.from_text(str(result)),
                    )
                except Exception as exc:
                    observation = f"Error: {exc}"
                    yield ToolResultEvent(
                        session_id=ctx.session_id,
                        agent_name=self.name,
                        tool_name=step.action,
                        error=str(exc),
                    )

            yield ObservationEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                content=Content.from_text(observation),
                tool_name=step.action,
            )

            messages.append(AIMessage(content=str(step.model_dump())))
            messages.append(HumanMessage(content=f"Observation: {observation}"))

        yield ErrorEvent(
            session_id=ctx.session_id,
            agent_name=self.name,
            message=f"Max iterations ({self.max_iterations}) reached without a final answer.",
        )
