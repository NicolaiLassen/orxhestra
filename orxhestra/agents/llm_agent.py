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

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import reduce
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.concurrency import gather_with_event_queue
from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.events.filters import apply_compaction, should_include_event
from orxhestra.models.llm_request import LlmRequest
from orxhestra.models.llm_response import LlmResponse
from orxhestra.models.part import (
    Content,
    DataPart,
    TextPart,
    ToolCallPart,
    ToolResponsePart,
)
from orxhestra.tools.exit_loop import EXIT_LOOP_SENTINEL
from orxhestra.tools.transfer_tool import TRANSFER_SENTINEL

if TYPE_CHECKING:
    from orxhestra.planners.base_planner import BasePlanner

logger: logging.Logger = logging.getLogger(__name__)

# Type alias for instruction providers - either a static string or a callable
# that receives the current Context and returns a string.
InstructionProvider = str | Callable[[InvocationContext], str | Awaitable[str]]

_DEFAULT_INSTRUCTIONS = """\
You are a helpful assistant. Answer the user's questions clearly and concisely.
When you have enough information to answer, provide a direct response.
Only use tools when necessary to complete the task.
"""

_PREV_CONTEXT_MAX_CHARS = 2000
_PREV_CONTEXT_TOTAL_MAX_CHARS = 6000

# Maximum characters per tool response kept in the message history.
_TOOL_RESPONSE_MAX_CHARS = 30_000


def _build_previous_context(
    events: list,
) -> list[HumanMessage]:
    """Build context messages from previous invocations' final responses.

    Deduplicates by agent name (keeps only the latest response per agent)
    and truncates long responses to prevent token explosion.

    Parameters
    ----------
    events : list[Event]
        Final response events from previous invocations
        (from ``ctx.get_previous_final_responses()``).

    Returns
    -------
    list[HumanMessage]
        Context messages formatted as ``[agent] said: ...``.
    """
    from orxhestra.tools.output import truncate_output

    if not events:
        return []

    # Deduplicate: keep only the LAST final response per agent
    latest_by_agent: dict[str, Any] = {}
    for event in events:
        agent = event.agent_name or "agent"
        latest_by_agent[agent] = event

    # Build messages, truncating each and respecting total budget
    messages: list[HumanMessage] = []
    total_chars = 0

    for agent, event in latest_by_agent.items():
        if total_chars >= _PREV_CONTEXT_TOTAL_MAX_CHARS:
            break

        remaining = _PREV_CONTEXT_TOTAL_MAX_CHARS - total_chars
        max_chars = min(_PREV_CONTEXT_MAX_CHARS, remaining)
        text = truncate_output(event.text, max_chars)

        content = f"[{agent}] said: {text}"
        messages.append(HumanMessage(content=content))
        total_chars += len(content)

    return messages


def _truncate_tool_message(msg: ToolMessage) -> ToolMessage:
    """Truncate a ToolMessage if its content exceeds the limit."""
    content = msg.content
    if isinstance(content, str) and len(content) > _TOOL_RESPONSE_MAX_CHARS:
        from orxhestra.tools.output import truncate_output

        content = truncate_output(content, _TOOL_RESPONSE_MAX_CHARS)
        return ToolMessage(
            content=content,
            tool_call_id=msg.tool_call_id,
        )
    return msg


class LlmAgent(BaseAgent):
    """Agent with a manual tool-call loop.

    Uses any LangChain ``BaseChatModel`` as the LLM backend. Supports
    static or dynamic system instructions, arbitrary LangChain tools,
    before/after callbacks at the model and tool level, an optional planner
    for per-turn planning instructions, and token-level streaming.

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
        to ``ctx.state[output_key]``.  This makes the output available
        to downstream agents via state (and ``{key}`` templating).
    output_schema : type, optional
        Optional Pydantic model for structured final output.
    include_contents : str
        Controls how much session history the agent sees:

        - ``'default'`` — full conversation history filtered by branch
          and invocation (drops sibling agents' events and events from
          other loop iterations).
        - ``'none'`` — no prior history; agent operates solely on the
          current input and its system instructions.  Ideal for
          sub-agents that don't need conversation context.
    max_iterations : int
        Maximum tool-call loop iterations before stopping.
    before_model_callback : callable, optional
        Called with ``(ctx, request: LlmRequest)`` before each LLM call.
    after_model_callback : callable, optional
        Called with ``(ctx, response: LlmResponse)`` after each LLM call.
    on_model_error_callback : callable, optional
        Called with ``(ctx, request: LlmRequest, exception)`` when an LLM
        call raises. Return a ``LlmResponse`` to recover, or ``None`` to
        push an error event.
    before_tool_callback : callable, optional
        Called with ``(ctx, tool_name, tool_args)`` before each tool execution.
    after_tool_callback : callable, optional
        Called with ``(ctx, tool_name, result)`` after each tool execution.
    """

    def __init__(
        self,
        name: str,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        *,
        instructions: InstructionProvider = _DEFAULT_INSTRUCTIONS,
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
    ) -> None:
        super().__init__(name=name, description=description)
        self._llm = llm
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._instructions = instructions
        self._planner = planner
        self._output_key = output_key
        self._output_schema = output_schema
        self._include_contents = include_contents
        self.max_iterations = max_iterations
        self.before_model_callback = before_model_callback
        self.after_model_callback = after_model_callback
        self.on_model_error_callback = on_model_error_callback
        self.before_tool_callback = before_tool_callback
        self.after_tool_callback = after_tool_callback

    # ── Prompt & history helpers ─────────────────────────────────

    async def _resolve_instructions(self, ctx: InvocationContext) -> str:
        """Resolve the system prompt from a string or instruction provider.

        Supports ``{key}`` template substitution from ``ctx.state``.
        Unknown keys are left as-is (no KeyError).
        """
        if callable(self._instructions):
            result = self._instructions(ctx)
            if asyncio.iscoroutine(result):
                prompt = await result
            else:
                prompt = result
        else:
            prompt = self._instructions

        # Template substitution: replace {key} with values from ctx.state.
        if ctx.state and "{" in prompt:

            class _DefaultDict(dict):
                def __missing__(self, key: str) -> str:
                    return "{" + key + "}"

            prompt = prompt.format_map(_DefaultDict(ctx.state))

        if self._output_schema is not None:
            parser = PydanticOutputParser(pydantic_object=self._output_schema)
            prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

        return prompt

    async def _build_conversation_history(
        self, ctx: InvocationContext, input: str,
    ) -> tuple[str, list[BaseMessage]]:
        """Build the system prompt and full message list for the LLM.

        Parameters
        ----------
        ctx : InvocationContext
            The current invocation context.
        input : str
            The user message or task description.

        Returns
        -------
        tuple[str, list[BaseMessage]]
            Resolved system prompt and the conversation messages.
        """
        system_prompt = await self._resolve_instructions(ctx)
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if self._include_contents != "none":
            prev_responses = ctx.get_previous_final_responses()
            prev_messages = _build_previous_context(prev_responses)
            messages.extend(prev_messages)

            filtered = ctx.get_events(
                current_branch=bool(ctx.branch),
                current_invocation=bool(ctx.branch),
            )
            inv_id = ctx.invocation_id
            filtered = [e for e in filtered if e.invocation_id != inv_id]
            if filtered:
                messages.extend(self._events_to_messages(filtered))

        messages.append(HumanMessage(content=input))
        return system_prompt, messages

    # ── LLM helpers ──────────────────────────────────────────────

    def _build_bound_llm(self) -> BaseChatModel:
        """Return the LLM with tools bound."""
        llm = self._llm
        if self._tools:
            llm = llm.bind_tools(list(self._tools.values()))
        return llm

    def _build_structured_llm(self) -> Any:
        """Return the LLM bound with structured output for the final answer."""
        if self._output_schema is None:
            return None
        for method in ("json_schema", "json_mode"):
            try:
                return self._llm.with_structured_output(
                    self._output_schema, method=method
                )
            except (NotImplementedError, TypeError, ValueError):
                continue
        return None

    def _build_request(
        self,
        system_instruction: str,
        messages: list[BaseMessage],
    ) -> LlmRequest:
        """Package the current turn into an LlmRequest."""
        return LlmRequest(
            model=getattr(self._llm, "model_name", None)
            or getattr(self._llm, "model", None),
            system_instruction=system_instruction,
            messages=list(messages),
            tools=list(self._tools.values()),
            tools_dict=dict(self._tools),
            output_schema=self._output_schema,
        )

    def _apply_planner_instruction(
        self,
        base_prompt: str,
        ctx: InvocationContext,
        request: LlmRequest,
    ) -> str:
        """Append the planner's instruction to the system prompt."""
        if self._planner is None:
            return base_prompt
        from orxhestra.agents.readonly_context import ReadonlyContext

        readonly = ReadonlyContext(ctx)
        instruction = self._planner.build_planning_instruction(readonly, request)
        if instruction:
            return f"{base_prompt}\n\n{instruction}"
        return base_prompt

    @staticmethod
    def _events_to_messages(events: list[Event]) -> list[BaseMessage]:
        """Convert session events to LangChain messages for multi-turn context.

        Applies visibility filtering, compaction processing, and drops
        tool call events whose calls lack a matching response (e.g. from
        interrupted sessions).

        Parameters
        ----------
        events : list[Event]
            Session events, already filtered by branch/invocation.

        Returns
        -------
        list[BaseMessage]
            LangChain messages ready for the LLM.
        """
        # 1. Apply compaction — replace raw events with summaries
        events = apply_compaction(events)

        # 2. Build set of responded tool call IDs
        responded_ids: set[str] = set()
        for event in events:
            if not event.partial and event.type == EventType.TOOL_RESPONSE:
                for tr in event.content.tool_responses:
                    if tr.tool_call_id:
                        responded_ids.add(tr.tool_call_id)

        # 3. Convert to messages with visibility filtering
        messages: list[BaseMessage] = []
        for event in events:
            if not should_include_event(event):
                continue
            if event.type == EventType.USER_MESSAGE:
                messages.append(event.to_langchain_message())
            elif event.type == EventType.AGENT_MESSAGE:
                if event.has_tool_calls:
                    paired = [
                        {"id": tc.tool_call_id, "name": tc.tool_name, "args": tc.args}
                        for tc in event.tool_calls
                        if tc.tool_call_id in responded_ids
                    ]
                    if paired:
                        messages.append(AIMessage(content="", tool_calls=paired))
                elif event.text:
                    messages.append(event.to_langchain_message())
            elif event.type == EventType.TOOL_RESPONSE:
                messages.append(event.to_langchain_message())

        return messages

    # ── LLM call with streaming ──────────────────────────────────

    async def _call_llm(
        self,
        llm: BaseChatModel,
        messages: list[BaseMessage],
        ctx: InvocationContext,
    ) -> AsyncIterator[Event | AIMessage]:
        """Call the LLM, yielding partial events and the final AIMessage.

        Yields partial AGENT_MESSAGE events for each text token as they
        arrive, then yields the final accumulated AIMessage as the last item.
        """
        rc = ctx.run_config

        chunks: list[AIMessageChunk] = []
        has_tool_calls = False

        async for chunk in llm.astream(messages, config=rc):
            chunks.append(chunk)

            if not has_tool_calls and (
                getattr(chunk, "tool_calls", None)
                or getattr(chunk, "tool_call_chunks", None)
            ):
                has_tool_calls = True

            if has_tool_calls:
                continue

            chunk_text = ""
            if isinstance(chunk.content, str):
                chunk_text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict):
                        chunk_text += part.get("text", "")

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

        Yields partial ``Event`` objects during streaming, then yields the
        final ``AIMessage`` as the last item. Yields ``None`` as the last
        item if an unrecoverable error occurred (caller should return).
        """
        try:
            async for item in self._call_llm(llm, messages, ctx):
                yield item
        except Exception as exc:
            recovered = await self._handle_llm_error(ctx, request, exc)
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
        """Handle an LLM call error, returning a recovery message or None."""
        if self.on_model_error_callback:
            recovery = await self.on_model_error_callback(ctx, request, exc)
            if recovery is not None:
                return AIMessage(content=recovery.text or "")
        return None

    # ── Final response handling ──────────────────────────────────

    async def _handle_final_response(
        self,
        ctx: InvocationContext,
        messages: list[BaseMessage],
        llm_response: LlmResponse,
    ) -> Event | None:
        """Build the final response event, or None to continue the loop.

        Checks the planner for pending tasks (returns None to continue),
        parses structured output if configured, saves to output_key,
        and builds the final event.

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
            The final response event, or None if the planner wants
            the loop to continue.
        """
        # Check if planner has pending tasks — continue the loop.
        if self._planner is not None:
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
                return None

        answer_text = llm_response.text
        parts: list[TextPart | DataPart] = []
        if answer_text:
            parts.append(TextPart(text=answer_text))

        # Parse structured output if schema is set.
        if self._output_schema is not None:
            structured = await self._parse_structured_output(
                answer_text, messages, ctx
            )
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

    async def _parse_structured_output(
        self,
        answer_text: str | None,
        messages: list[BaseMessage],
        ctx: InvocationContext,
    ) -> Any:
        """Try to parse structured output from the answer text."""
        parser = PydanticOutputParser(pydantic_object=self._output_schema)
        if answer_text:
            try:
                return parser.parse(answer_text)
            except Exception:
                pass
        structured_llm = self._build_structured_llm()
        if structured_llm is not None:
            try:
                return await structured_llm.ainvoke(
                    messages, config=ctx.run_config,
                )
            except Exception:
                pass
        return None

    # ── Tool execution ───────────────────────────────────────────

    async def _execute_tool_calls(
        self,
        ctx: InvocationContext,
        llm_response: LlmResponse,
    ) -> AsyncIterator[Event | tuple[Event, ToolMessage]]:
        """Execute tool calls in parallel, yielding events as they complete.

        Yields
        ------
        Event
            The initial tool call event and intermediate child events.
        tuple[Event, ToolMessage]
            Tool response event paired with its ToolMessage for history.
        """
        event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

        tool_ctx = ctx.model_copy(
            update={"event_callback": event_queue.put_nowait}
        )

        # Yield the tool call event.
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content(parts=[
                ToolCallPart(
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    args=tc["args"],
                    metadata={"interactive": True}
                    if getattr(
                        self._tools.get(tc["name"]), "interactive", False
                    )
                    else {},
                )
                for tc in llm_response.tool_calls
            ]),
            llm_response=llm_response,
        )

        async def _execute_one(tool_call: dict) -> tuple[Event, ToolMessage]:
            """Execute a single tool call and return response event + message."""
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call["id"]

            tool = self._tools.get(t_name)
            if tool is None:
                return self._tool_response(
                    ctx, t_id, t_name,
                    error=f"Tool '{t_name}' not found. "
                    f"Available: {list(self._tools)}",
                )

            if hasattr(tool, "inject_context"):
                tool.inject_context(tool_ctx)
            if self.before_tool_callback:
                await self.before_tool_callback(ctx, t_name, t_args)

            try:
                result = await tool.ainvoke(t_args, config=ctx.run_config)
                if self.after_tool_callback:
                    await self.after_tool_callback(ctx, t_name, result)
                return self._tool_response(ctx, t_id, t_name, result=str(result))
            except Exception as exc:
                if self.after_tool_callback:
                    await self.after_tool_callback(ctx, t_name, None)
                return self._tool_response(ctx, t_id, t_name, error=str(exc))

        # Run all tool calls concurrently, streaming intermediate events.
        async for item in gather_with_event_queue(
            [_execute_one(tc) for tc in llm_response.tool_calls],
            event_queue,
        ):
            if isinstance(item, Event):
                yield item
            else:
                yield item  # tuple[Event, ToolMessage]

    def _tool_response(
        self,
        ctx: InvocationContext,
        t_id: str,
        t_name: str,
        *,
        result: str | None = None,
        error: str | None = None,
    ) -> tuple[Event, ToolMessage]:
        """Build a tool response event and ToolMessage pair."""
        part = ToolResponsePart(
            tool_call_id=t_id,
            tool_name=t_name,
            result=result or "",
            error=error,
        )
        actions = EventActions()
        content_str = result or error or ""
        if result and result.startswith(TRANSFER_SENTINEL):
            actions = EventActions(
                transfer_to_agent=result.removeprefix(TRANSFER_SENTINEL).strip()
            )
        elif result == EXIT_LOOP_SENTINEL:
            actions = EventActions(escalate=True)
        event = self._emit_event(
            ctx,
            EventType.TOOL_RESPONSE,
            content=Content(parts=[part]),
            actions=actions,
        )
        msg_kwargs = {"status": "error"} if error else {}
        msg = ToolMessage(
            content=content_str,
            tool_call_id=t_id,
            **msg_kwargs,
        )
        return (event, msg)

    # ── Main loop ────────────────────────────────────────────────

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
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            Events emitted during execution, including partial streaming
            tokens, tool call/response events, and the final answer.
        """
        ctx = self._ensure_ctx(config, ctx)
        system_prompt, messages = await self._build_conversation_history(
            ctx, input
        )
        llm = self._build_bound_llm()

        for _ in range(self.max_iterations):
            if ctx.end_invocation:
                return

            # Build request and apply planner instruction.
            request = self._build_request(system_prompt, messages)
            effective_prompt = self._apply_planner_instruction(
                system_prompt, ctx, request
            )
            if effective_prompt != system_prompt:
                if messages and isinstance(messages[0], SystemMessage):
                    messages[0] = SystemMessage(content=effective_prompt)

            if self.before_model_callback:
                await self.before_model_callback(ctx, request)

            # Call LLM with error recovery.
            raw_response: AIMessage | None = None
            async for item in self._call_llm_with_recovery(
                llm, messages, ctx, request
            ):
                if isinstance(item, AIMessage):
                    raw_response = item
                elif item is None:
                    # Unrecoverable error — emit error event and stop.
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
            llm_response = LlmResponse.from_ai_message(raw_response)
            if self._planner is not None:
                from orxhestra.agents.readonly_context import ReadonlyContext

                readonly = ReadonlyContext(ctx)
                replacement = self._planner.process_planning_response(
                    readonly, llm_response
                )
                if replacement is not None:
                    llm_response = replacement

            if self.after_model_callback:
                await self.after_model_callback(ctx, llm_response)

            messages.append(raw_response)

            # No tool calls → final answer or planner continuation.
            if not llm_response.has_tool_calls:
                final_event = await self._handle_final_response(
                    ctx, messages, llm_response
                )
                if final_event is None:
                    continue
                yield final_event
                return

            # Execute tool calls and collect ToolMessages.
            tool_messages: list[ToolMessage] = []
            async for item in self._execute_tool_calls(ctx, llm_response):
                if isinstance(item, tuple):
                    event, tool_msg = item
                    yield event
                    tool_messages.append(_truncate_tool_message(tool_msg))
                else:
                    yield item

            messages.extend(tool_messages)

        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(
                f"Max iterations ({self.max_iterations}) reached "
                f"without a final answer."
            ),
            metadata={"error": True},
        )
