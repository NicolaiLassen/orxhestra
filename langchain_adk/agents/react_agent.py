"""ReActAgent - structured Reason+Act loop.

Uses with_structured_output() to get a typed ReActStep per iteration.
SSE streaming = token-level partial ThoughtEvents. No streaming = complete events only.
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


class ReActStep(BaseModel):
    """A single ReAct step — either reason+act or final answer."""

    scratchpad: str = Field(description="Running notes and observations accumulated across steps")
    thought: str = Field(description="Current reasoning about the problem")
    action: Optional[str] = Field(default=None, description="Tool name to call, or null if giving final answer")
    action_input: Optional[str] = Field(default=None, description="Input for the tool, or null if giving final answer")
    answer: Optional[str] = Field(default=None, description="Final answer text, or null if calling a tool")

    @property
    def is_final(self) -> bool:
        return self.answer is not None


_SYSTEM_PROMPT = """\
You are a reasoning agent. You solve problems step by step using the ReAct pattern.

IMPORTANT: Output exactly ONE JSON object per response. Do NOT output multiple JSON objects.

Your JSON object MUST have these fields:
- "scratchpad": your running notes across steps
- "thought": your current reasoning about what to do next
- "action": the tool name to call (set to null if giving a final answer)
- "action_input": the input for the tool (set to null if giving a final answer)
- "answer": your final answer text (set to null if calling a tool)

You will be called multiple times. Each call, output ONE step only.

Available tools:
{tool_descriptions}

Rules:
- Output exactly ONE JSON object. Never output multiple steps.
- Always think before acting.
- Use your scratchpad to accumulate observations and notes across steps.
- Only set "answer" when you are confident in the result.
- Never call a tool that is not listed above.
"""


class ReActAgent(BaseAgent):
    """Agent that uses explicit structured reasoning before each action.

    Uses LangChain ``with_structured_output()`` to enforce a typed
    ``ReActStep`` at every iteration. SSE streaming emits token-level
    partial ``ThoughtEvent`` objects; without streaming only complete
    events are yielded.

    Attributes
    ----------
    llm : BaseChatModel
        The LangChain chat model to use.
    tools : list[BaseTool]
        Tools available to the agent.
    max_iterations : int
        Maximum ReAct loop iterations before yielding an error.
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
        try:
            self._llm = llm.with_structured_output(
                schema=ReActStep, include_raw=False, method="json_mode",
            )
        except (NotImplementedError, TypeError, ValueError):
            self._llm = llm.with_structured_output(
                schema=ReActStep, include_raw=False,
            )

    def _system_message(self) -> SystemMessage:
        tool_descriptions = "\n".join(
            f"  - {name}: {tool.description}"
            for name, tool in self._tools.items()
        ) or "  (no tools available)"
        return SystemMessage(
            content=_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)
        )

    async def _get_step(
        self,
        messages: list[BaseMessage],
        ctx: InvocationContext,
    ) -> AsyncIterator[ThoughtEvent | ReActStep]:
        """Get a ReActStep from the LLM. Yields partial ThoughtEvents if SSE streaming.

        Always yields the final ReActStep as the last item.
        """
        streaming = self.is_streaming(ctx)

        if not streaming:
            yield await self._llm.ainvoke(messages)
            return

        # SSE: stream partial thoughts as the JSON builds up
        last_thought = ""
        final_step: Optional[ReActStep] = None

        async for partial in self._llm.astream(messages):
            if not isinstance(partial, ReActStep):
                continue
            final_step = partial
            current_thought = partial.thought or ""
            if current_thought != last_thought:
                last_thought = current_thought
                yield ThoughtEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    content=Content.from_text(current_thought),
                    scratchpad=partial.scratchpad or "",
                    partial=True,
                )

        if final_step is None:
            raise RuntimeError("Streaming produced no output")
        yield final_step

    async def run(
        self,
        input: str,
        *,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]:
        messages: list[BaseMessage] = [
            self._system_message(),
            HumanMessage(content=input),
        ]

        for _ in range(self.max_iterations):
            # Get step — may yield partial ThoughtEvents before the final ReActStep
            step: Optional[ReActStep] = None
            try:
                async for item in self._get_step(messages, ctx):
                    if isinstance(item, ThoughtEvent):
                        yield item  # partial thought
                    else:
                        step = item  # final ReActStep
            except Exception as exc:
                yield ErrorEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                )
                return

            # Complete thought (partial=False)
            yield ThoughtEvent(
                session_id=ctx.session_id,
                agent_name=self.name,
                content=Content.from_text(step.thought),
                scratchpad=step.scratchpad,
                partial=False,
            )

            if step.is_final:
                yield FinalAnswerEvent(
                    session_id=ctx.session_id,
                    agent_name=self.name,
                    content=Content.from_text(step.answer),
                    scratchpad=step.scratchpad,
                )
                return

            # Tool call
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
