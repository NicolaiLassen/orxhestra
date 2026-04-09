"""LlmAgent — the primary orxhestra agent.

Implements a manual tool-call loop using LangChain's BaseChatModel.
No LangGraph — orchestration is pure Python async.

The loop:
  1. Build system prompt from instructions (or instruction provider).
  2. Load session history filtered by branch + invocation_id (like
     ``_get_events(current_invocation=True,
     current_branch=True)``).  Events are further processed by
     visibility filtering and compaction.
  3. If a planner is attached, append its planning instruction.
  4. Call ``llm.bind_tools(tools).astream(messages)``.
  5. If response has tool_calls → execute in parallel → append
     ToolMessages → loop.
  6. Yield typed events throughout (partial tokens, tool calls,
     tool responses, final answer).
  7. Repeat until no tool calls or ``max_iterations``.

Context management:
  - **Branch filtering** — each agent only sees its own history;
    sibling agents in a SequentialAgent don't contaminate each other.
  - **Invocation filtering** — isolates events across different runs
    and LoopAgent iterations (via unique branch suffixes).
  - **Visibility filtering** — drops empty, partial, framework
    lifecycle (AGENT_START/END), and error-only events.
  - **Compaction** — when an event carries ``actions.compaction``,
    all raw events in that timestamp range are replaced by the
    compaction summary in the LLM context.
  - **include_contents** — ``'default'`` loads full filtered history;
    ``'none'`` skips history entirely (sub-agents that don't need
    conversation context).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import reduce
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.callbacks import LlmAgentCallbacks
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.agents.message_builder import (
    InstructionProvider,
    MessageBuilder,
    _truncate_tool_message,
)
from orxhestra.agents.planner_adapter import PlannerAdapter
from orxhestra.agents.structured_output import StructuredOutputParser
from orxhestra.agents.tool_executor import ToolExecutor
from orxhestra.agents.tracing import trace
from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.models.content_parser import parse_content_blocks
from orxhestra.models.llm_request import LlmRequest
from orxhestra.models.llm_response import LlmResponse
from orxhestra.models.part import Content, DataPart, TextPart

if TYPE_CHECKING:
    from orxhestra.planners.base_planner import BasePlanner

logger: logging.Logger = logging.getLogger(__name__)


class LlmAgent(BaseAgent):
    """Agent with a manual tool-call loop.

    Uses any LangChain ``BaseChatModel`` as the LLM backend. Supports
    static or dynamic system instructions, arbitrary LangChain tools,
    before/after callbacks at the model and tool level, an optional planner
    for per-turn planning instructions, and token-level streaming.

    The heavy lifting is delegated to composable helpers:

    * ``MessageBuilder`` — instruction resolution and history assembly.
    * ``ToolExecutor`` — concurrent tool execution with event streaming.
    * ``PlannerAdapter`` — prompt enrichment, response processing, and
      continuation decisions.
    * ``StructuredOutputParser`` — Pydantic output schema parsing.
    * ``LlmAgentCallbacks`` — grouped lifecycle callbacks.

    Attributes
    ----------
    llm : BaseChatModel
        The LangChain chat model to use.
    tools : list[BaseTool]
        Tools available to the agent.
    instructions : str or callable
        System prompt string or callable returning one.
    planner : BasePlanner, optional
        Planner that injects planning instructions before each LLM call.
    output_key : str, optional
        When set, the agent's final text answer is automatically saved
        to ``ctx.state[output_key]``.
    output_schema : type, optional
        Optional Pydantic model for structured final output.
    include_contents : str
        Controls how much session history the agent sees:

        - ``'default'`` — full conversation history filtered by branch
          and invocation.
        - ``'none'`` — no prior history.
    max_iterations : int
        Maximum tool-call loop iterations before stopping.
    callbacks : LlmAgentCallbacks
        Grouped lifecycle callbacks for model and tool events.
    tool_response_max_chars : int
        Max characters kept per tool response in message history.
    context_max_chars : int
        Max characters per previous agent response in context.
    context_total_max_chars : int
        Total character budget for all previous agent responses.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        *,
        instructions: InstructionProvider | None = None,
        description: str = "",
        planner: BasePlanner | None = None,
        output_key: str | None = None,
        output_schema: type | None = None,
        include_contents: str = "default",
        max_iterations: int = 10,
        before_model_callback: (
            Callable[[InvocationContext, LlmRequest], Awaitable[None]] | None
        ) = None,
        after_model_callback: (
            Callable[[InvocationContext, LlmResponse], Awaitable[None]] | None
        ) = None,
        on_model_error_callback: (
            Callable[
                [InvocationContext, LlmRequest, Exception],
                Awaitable[LlmResponse | None],
            ]
            | None
        ) = None,
        before_tool_callback: (
            Callable[[InvocationContext, str, dict], Awaitable[None]] | None
        ) = None,
        after_tool_callback: (
            Callable[[InvocationContext, str, Any], Awaitable[None]] | None
        ) = None,
        tool_response_max_chars: int = 30_000,
        context_max_chars: int = 5_000,
        context_total_max_chars: int = 10_000,
        signing_key: Any | None = None,
        signing_did: str = "",
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            signing_key=signing_key,
            signing_did=signing_did,
        )

        self._llm = llm
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._instructions = instructions
        self._output_key = output_key
        self._output_schema = output_schema
        self.max_iterations = max_iterations

        # Expose limits for test assertions.
        self.tool_response_max_chars = tool_response_max_chars
        self.context_max_chars = context_max_chars
        self.context_total_max_chars = context_total_max_chars

        # Compose helpers.
        self._callbacks = LlmAgentCallbacks(
            before_model=before_model_callback,
            after_model=after_model_callback,
            on_model_error=on_model_error_callback,
            before_tool=before_tool_callback,
            after_tool=after_tool_callback,
        )
        # Use the default prompt when no instructions are provided.
        from orxhestra.agents.message_builder import _DEFAULT_INSTRUCTIONS

        effective_instructions: InstructionProvider = (
            _DEFAULT_INSTRUCTIONS if instructions is None else instructions
        )
        self._message_builder = MessageBuilder(
            instructions=effective_instructions,
            output_schema=output_schema,
            include_contents=include_contents,
            context_max_chars=context_max_chars,
            context_total_max_chars=context_total_max_chars,
            tool_response_max_chars=tool_response_max_chars,
        )
        self._tool_executor = ToolExecutor(
            tools=self._tools,
            callbacks=self._callbacks,
            emit_event=self._emit_event,
        )
        self._planner_adapter: PlannerAdapter | None = PlannerAdapter(planner) if planner else None
        # Keep raw reference for subclass access (e.g. ReActAgent).
        self._planner = planner
        self._structured_parser: StructuredOutputParser | None = (
            StructuredOutputParser(llm, output_schema) if output_schema else None
        )

    @property
    def before_model_callback(self) -> Callable[..., Any] | None:
        """Before-model callback."""
        return self._callbacks.before_model

    @property
    def after_model_callback(self) -> Callable[..., Any] | None:
        """After-model callback."""
        return self._callbacks.after_model

    @property
    def on_model_error_callback(self) -> Callable[..., Any] | None:
        """On-model-error callback."""
        return self._callbacks.on_model_error

    @property
    def before_tool_callback(self) -> Callable[..., Any] | None:
        """Before-tool callback."""
        return self._callbacks.before_tool

    @property
    def after_tool_callback(self) -> Callable[..., Any] | None:
        """After-tool callback."""
        return self._callbacks.after_tool

    async def _resolve_instructions(self, ctx: InvocationContext) -> str:
        """Resolve the system prompt.

        Delegates to ``MessageBuilder.resolve_instructions``.  Kept as
        a method so subclasses (e.g. ``ReActAgent``) can call it.

        Parameters
        ----------
        ctx : InvocationContext
            Current invocation context.

        Returns
        -------
        str
            The fully resolved system prompt.
        """
        return await self._message_builder.resolve_instructions(ctx)

    def _build_bound_llm(self) -> BaseChatModel:
        """Return the LLM with tools bound."""
        llm: BaseChatModel = self._llm
        if self._tools:
            llm = llm.bind_tools(list(self._tools.values()))
        return llm

    def _build_request(
        self,
        system_instruction: str,
        messages: list[BaseMessage],
    ) -> LlmRequest:
        """Package the current turn into an ``LlmRequest``."""
        return LlmRequest(
            model=getattr(self._llm, "model_name", None) or getattr(self._llm, "model", None),
            system_instruction=system_instruction,
            messages=list(messages),
            tools=list(self._tools.values()),
            tools_dict=dict(self._tools),
            output_schema=self._output_schema,
        )

    async def _call_llm(
        self,
        llm: BaseChatModel,
        messages: list[BaseMessage],
        ctx: InvocationContext,
    ) -> AsyncIterator[Event | AIMessage]:
        """Call the LLM, yielding partial events and the final ``AIMessage``.

        Yields partial ``AGENT_MESSAGE`` events for each text token as
        they arrive, then yields the final accumulated ``AIMessage`` as
        the last item.
        """
        chunks: list[AIMessageChunk] = []
        has_tool_calls: bool = False

        async for chunk in llm.astream(messages, config=ctx.run_config):
            chunks.append(chunk)

            if not has_tool_calls and (
                getattr(chunk, "tool_calls", None) or getattr(chunk, "tool_call_chunks", None)
            ):
                has_tool_calls = True

            if has_tool_calls:
                continue

            chunk_text, chunk_thinking = parse_content_blocks(chunk.content)

            if chunk_thinking:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_thinking(chunk_thinking),
                    partial=True,
                    turn_complete=False,
                )

            if chunk_text:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content.from_text(chunk_text),
                    partial=True,
                    turn_complete=False,
                )

        if not chunks:
            yield AIMessage(content="")
            return

        yield reduce(lambda x, y: x + y, chunks)

    async def _call_llm_with_recovery(
        self,
        llm: BaseChatModel,
        messages: list[BaseMessage],
        ctx: InvocationContext,
        request: LlmRequest,
    ) -> AsyncIterator[Event | AIMessage | None]:
        """Call the LLM with error recovery, yielding partial events.

        Yields partial ``Event`` objects during streaming, then yields
        the final ``AIMessage`` as the last item.  Yields ``None`` as
        the last item if an unrecoverable error occurred.
        """
        try:
            async for item in self._call_llm(llm, messages, ctx):
                yield item
        except Exception as exc:
            recovered: AIMessage | None = await self._handle_llm_error(ctx, request, exc)
            if recovered is not None:
                yield recovered
            else:
                yield None

    async def _handle_llm_error(
        self,
        ctx: InvocationContext,
        request: LlmRequest,
        exc: Exception,
    ) -> AIMessage | None:
        """Handle an LLM call error, returning a recovery message or ``None``."""
        if self._callbacks.on_model_error:
            recovery: LlmResponse | None = await self._callbacks.on_model_error(ctx, request, exc)
            if recovery is not None:
                return AIMessage(content=recovery.text or "")
        return None

    async def _handle_final_response(
        self,
        ctx: InvocationContext,
        messages: list[BaseMessage],
        llm_response: LlmResponse,
    ) -> Event | None:
        """Build the final response event, or ``None`` to continue the loop.

        Checks the planner for pending tasks (returns ``None`` to
        continue), parses structured output if configured, saves to
        ``output_key``, and builds the final event.

        Parameters
        ----------
        ctx : InvocationContext
            The current invocation context.
        messages : list[BaseMessage]
            Conversation messages (mutated if planner continues).
        llm_response : LlmResponse
            The LLM response with no tool calls.

        Returns
        -------
        Event or None
            The final response event, or ``None`` if the planner wants
            the loop to continue.
        """
        if self._planner_adapter is not None:
            if self._planner_adapter.should_continue(ctx, messages):
                return None

        answer_text: str | None = llm_response.text
        parts: list[TextPart | DataPart] = []
        if answer_text:
            parts.append(TextPart(text=answer_text))

        # Parse structured output if schema is set.
        if self._structured_parser is not None:
            structured = await self._structured_parser.parse(answer_text, messages, ctx)
            if structured is not None:
                parts.append(DataPart(data=structured.model_dump()))

        # Auto-save final answer to state when output_key is set.
        emit_kwargs: dict[str, Any] = {
            "content": Content(parts=parts),
            "partial": False,
            "llm_response": llm_response,
        }
        if self._output_key and answer_text:
            ctx.state[self._output_key] = answer_text
            emit_kwargs["actions"] = EventActions(
                state_delta={self._output_key: answer_text},
            )

        return self._emit_event(ctx, EventType.AGENT_MESSAGE, **emit_kwargs)

    @trace("LlmAgent")
    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Stream events from the LLM agent.

        Runs the tool-call loop, yielding events as they occur:
        partial tokens, tool calls, tool responses, and the final answer.

        Parameters
        ----------
        input : str
            The user message or task description.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : InvocationContext, optional
            Invocation context.  Auto-created if not provided.

        Yields
        ------
        Event
            Events emitted during execution, including partial streaming
            tokens, tool call/response events, and the final answer.
        """
        system_prompt, messages = await self._message_builder.build_conversation_history(ctx, input)
        llm: BaseChatModel = self._build_bound_llm()

        for _ in range(self.max_iterations):
            # External kill switch — stop immediately without yielding.
            if ctx.end_invocation:
                return

            # Build request and apply planner instruction.
            request: LlmRequest = self._build_request(system_prompt, messages)
            if self._planner_adapter is not None:
                effective_prompt: str = self._planner_adapter.enrich_prompt(
                    ctx, system_prompt, request
                )
                if effective_prompt != system_prompt:
                    if messages and isinstance(messages[0], SystemMessage):
                        messages[0] = SystemMessage(content=effective_prompt)

            if self._callbacks.before_model:
                await self._callbacks.before_model(ctx, request)

            # Log context size at debug level only.
            total_chars: int = sum(len(str(m.content)) for m in messages)
            if total_chars > 200_000:
                logger.debug(
                    "Message context is %d chars (~%dk tokens).",
                    total_chars,
                    total_chars // 4000,
                )

            # Call LLM with error recovery.
            raw_response: AIMessage | None = None
            async for item in self._call_llm_with_recovery(llm, messages, ctx, request):
                if isinstance(item, AIMessage):
                    raw_response = item
                elif item is None:
                    yield self._emit_event(
                        ctx,
                        EventType.AGENT_MESSAGE,
                        content=Content.from_text("LLM call failed."),
                        metadata={"error": True},
                    )
                    return
                else:
                    yield item  # partial event

            if raw_response is None:
                return

            # Post-process response.
            llm_response: LlmResponse = LlmResponse.from_ai_message(raw_response)
            if self._planner_adapter is not None:
                llm_response = self._planner_adapter.process_response(ctx, llm_response)

            if self._callbacks.after_model:
                await self._callbacks.after_model(ctx, llm_response)

            messages.append(raw_response)

            # No tool calls → final answer or planner continuation.
            if not llm_response.has_tool_calls:
                final_event: Event | None = await self._handle_final_response(
                    ctx, messages, llm_response
                )
                if final_event is None:
                    continue
                yield final_event
                return

            # Execute tool calls and collect ToolMessages.
            tool_messages: list[ToolMessage] = []
            async for item in self._tool_executor.execute(ctx, llm_response):
                if isinstance(item, tuple):
                    event, tool_msg = item
                    yield event
                    tool_messages.append(
                        _truncate_tool_message(tool_msg, self.tool_response_max_chars)
                    )
                else:
                    yield item

            messages.extend(tool_messages)

        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(
                f"Max iterations ({self.max_iterations}) reached without a final answer."
            ),
            metadata={"error": True},
        )
