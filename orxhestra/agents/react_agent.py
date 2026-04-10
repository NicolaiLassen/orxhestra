"""ReActAgent - structured Reason+Act loop.

Uses with_structured_output() to get a typed ReActStep per iteration.
Extends LlmAgent, inheriting support for instructions, planners, skills,
callbacks, and all other LLM-agent features.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.agents.llm_agent import LlmAgent
from orxhestra.agents.tracing import trace
from orxhestra.events.event import Event, EventType
from orxhestra.models.part import Content, ToolCallPart, ToolResponsePart


class ReActStep(BaseModel):
    """A single ReAct step — either reason+act or final answer.

    Attributes
    ----------
    scratchpad : str
        Running notes and observations accumulated across steps.
    thought : str
        Current reasoning about the problem.
    action : str, optional
        Tool name to call, or ``None`` if giving final answer.
    action_input : str, optional
        Input for the tool, or ``None`` if giving final answer.
    answer : str, optional
        Final answer text, or ``None`` if calling a tool.
    """

    scratchpad: str = Field(
        description="Running notes and observations accumulated across steps",
    )
    thought: str = Field(description="Current reasoning about the problem")
    action: str | None = Field(
        default=None, description="Tool name to call, or null if giving final answer",
    )
    action_input: str | None = Field(
        default=None, description="Input for the tool, or null if giving final answer",
    )
    answer: str | None = Field(
        default=None, description="Final answer text, or null if calling a tool",
    )

    @property
    def is_final(self) -> bool:
        """Whether this step produced a final answer (no more tool calls)."""
        return self.answer is not None


_REACT_SYSTEM_PROMPT = """\
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


class ReActAgent(LlmAgent):
    """Agent that uses explicit structured reasoning before each action.

    Extends ``LlmAgent``, inheriting support for custom instructions,
    planners, skills, callbacks, and output schemas. Uses LangChain
    ``with_structured_output()`` to enforce a typed ``ReActStep`` at
    every iteration.

    Parameters
    ----------
    name : str
        Unique name identifying this agent.
    llm : BaseChatModel
        The LangChain chat model to use.
    tools : list[BaseTool], optional
        Tools available to the agent.
    description : str
        Short description used by LLMs for routing decisions.
    max_iterations : int
        Maximum ReAct loop iterations before yielding an error.
    **kwargs
        All other keyword arguments are passed to ``LlmAgent``, including
        ``instructions``, ``planner``, ``output_schema``, and callbacks.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        *,
        description: str = "",
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        # Default to empty instructions — the ReAct prompt is the base
        kwargs.setdefault("instructions", "")
        super().__init__(
            name=name,
            llm=llm,
            tools=tools,
            description=description,
            max_iterations=max_iterations,
            **kwargs,
        )
        try:
            self._structured_llm = llm.with_structured_output(
                schema=ReActStep, include_raw=False, method="json_mode",
            )
        except (NotImplementedError, TypeError, ValueError):
            self._structured_llm = llm.with_structured_output(
                schema=ReActStep, include_raw=False,
            )

    async def _build_react_system_prompt(self, ctx: Context) -> str:
        """Build the ReAct system prompt, incorporating custom instructions."""
        tool_descriptions = "\n".join(
            f"  - {name}: {tool.description}"
            for name, tool in self._tools.items()
        ) or "  (no tools available)"

        base = _REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

        # Append custom instructions if provided
        custom = await self._resolve_instructions(ctx)
        if custom:
            base = f"{base}\n\nAdditional instructions:\n{custom}"

        # Append planner instruction if a planner is attached
        if self._planner is not None:
            from orxhestra.agents.readonly_context import ReadonlyContext
            from orxhestra.models.llm_request import LlmRequest

            request = LlmRequest(
                model=getattr(self._llm, "model_name", None)
                or getattr(self._llm, "model", None),
                system_instruction=base,
                messages=[],
                tools=list(self._tools.values()),
                tools_dict=dict(self._tools),
            )
            readonly = ReadonlyContext(ctx)
            instruction = self._planner.build_planning_instruction(readonly, request)
            if instruction:
                base = f"{base}\n\n{instruction}"

        return base

    async def _invoke_step(
        self,
        messages: list[BaseMessage],
        ctx: Context,
    ) -> AsyncIterator[Event | ReActStep]:
        """Invoke the LLM for one ReAct step, yielding partial events.

        Yields partial events during streaming, then yields the final ReActStep.
        """
        last_thought = ""
        last_answer = ""
        step: ReActStep | None = None

        async for partial in self._structured_llm.astream(messages, config=ctx.run_config):
            if not isinstance(partial, ReActStep):
                continue
            step = partial

            current_thought = partial.thought or ""
            if current_thought != last_thought:
                last_thought = current_thought
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_text(current_thought),
                    metadata={
                        "react_step": "thought",
                        "scratchpad": partial.scratchpad or "",
                    },
                    partial=True,
                    turn_complete=False,
                )

            current_answer = partial.answer or ""
            if current_answer and current_answer != last_answer:
                last_answer = current_answer
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_text(current_answer),
                    metadata={"scratchpad": partial.scratchpad or ""},
                    partial=True,
                    turn_complete=False,
                )

        if step is None:
            raise RuntimeError("Streaming produced no output")

        # Complete thought event
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(step.thought),
            metadata={
                "react_step": "thought",
                "scratchpad": step.scratchpad,
            },
            partial=False,
        )

        yield step

    @trace("ReActAgent")
    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: Context | None = None,
    ) -> AsyncIterator[Event]:
        """Run the ReAct reasoning loop, yielding events.

        Parameters
        ----------
        input : str
            The user message or problem to solve.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : Context, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            Events emitted during execution, including thought steps,
            tool calls, observations, and the final answer.
        """
        system_prompt = await self._build_react_system_prompt(ctx)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=input),
        ]

        for _ in range(self.max_iterations):
            step: ReActStep | None = None
            try:
                async for item in self._invoke_step(messages, ctx):
                    if isinstance(item, ReActStep):
                        step = item
                    else:
                        yield item
            except Exception as exc:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_text(str(exc)),
                    metadata={
                        "error": True,
                        "exception_type": type(exc).__name__,
                    },
                )
                return

            # Final answer
            if step.is_final:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_text(step.answer),
                    metadata={"scratchpad": step.scratchpad},
                )
                return

            # Tool call
            yield self._emit_event(
                ctx,
                EventType.AGENT_MESSAGE,
                metadata={
                    "react_step": "action",
                    "action": step.action,
                    "action_input": step.action_input,
                },
            )

            tool = self._tools.get(step.action)
            if tool is None:
                observation = f"Error: tool '{step.action}' not found."
            else:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content(
                        parts=[
                            ToolCallPart(
                                tool_call_id=f"react_{step.action}",
                                tool_name=step.action,
                                args={"input": step.action_input},
                            )
                        ]
                    ),
                )
                try:
                    result = await tool.ainvoke(step.action_input, config=ctx.run_config)
                    observation = str(result)
                    yield self._emit_event(
                        ctx,
                        EventType.TOOL_RESPONSE,
                        content=Content(
                            parts=[
                                ToolResponsePart(
                                    tool_call_id=f"react_{step.action}",
                                    tool_name=step.action,
                                    result=observation,
                                )
                            ]
                        ),
                    )
                except Exception as exc:
                    observation = f"Error: {exc}"
                    yield self._emit_event(
                        ctx,
                        EventType.TOOL_RESPONSE,
                        content=Content(
                            parts=[
                                ToolResponsePart(
                                    tool_call_id=f"react_{step.action}",
                                    tool_name=step.action,
                                    error=str(exc),
                                )
                            ]
                        ),
                    )

            yield self._emit_event(
                ctx,
                EventType.AGENT_MESSAGE,
                content=Content.from_text(observation),
                metadata={"react_step": "observation", "tool_name": step.action},
            )

            messages.append(AIMessage(content=str(step.model_dump())))
            messages.append(HumanMessage(content=f"Observation: {observation}"))

        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(
                f"Max iterations ({self.max_iterations}) reached without a final answer."
            ),
            metadata={"error": True},
        )
