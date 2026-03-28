"""LlmAgent - the primary orxhestra agent.

Implements a manual tool-call loop using LangChain's BaseChatModel.
No LangGraph - orchestration is pure Python async.

The loop:
  1. Build system prompt from instructions (or instruction provider)
  2. If a planner is attached, append its planning instruction
  3. Build an LlmRequest and call llm.bind_tools(tools).astream(messages)
  4. Wrap the AIMessage in LlmResponse
  5. If response has tool_calls -> execute each in parallel -> append ToolMessages -> loop
  6. Yield typed events throughout
  7. Repeat until no tool calls or max_iterations
"""

from __future__ import annotations

import asyncio
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
from orxhestra.agents.context import Context
from orxhestra.concurrency import gather_with_event_queue
from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
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

# Type alias for instruction providers - either a static string or a callable
# that receives the current Context and returns a string.
InstructionProvider = str | Callable[[Context], str | Awaitable[str]]

_DEFAULT_INSTRUCTIONS = """\
You are a helpful assistant. Answer the user's questions clearly and concisely.
When you have enough information to answer, provide a direct response.
Only use tools when necessary to complete the task.
"""


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
    output_schema : type, optional
        Optional Pydantic model for structured final output.
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
        output_schema: type | None = None,
        max_iterations: int = 10,
        before_model_callback: (
            Callable[[Context, LlmRequest], Awaitable[None]] | None
        ) = None,
        after_model_callback: (
            Callable[[Context, LlmResponse], Awaitable[None]] | None
        ) = None,
        on_model_error_callback: (
            Callable[
                [Context, LlmRequest, Exception],
                Awaitable[LlmResponse | None],
            ]
            | None
        ) = None,
        before_tool_callback: (
            Callable[[Context, str, dict], Awaitable[None]] | None
        ) = None,
        after_tool_callback: (
            Callable[[Context, str, Any], Awaitable[None]] | None
        ) = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self._llm = llm
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._instructions = instructions
        self._planner = planner
        self._output_schema = output_schema
        self.max_iterations = max_iterations
        self.before_model_callback = before_model_callback
        self.after_model_callback = after_model_callback
        self.on_model_error_callback = on_model_error_callback
        self.before_tool_callback = before_tool_callback
        self.after_tool_callback = after_tool_callback

    async def _resolve_instructions(self, ctx: Context) -> str:
        """Resolve the system prompt from a string or instruction provider."""
        if callable(self._instructions):
            result = self._instructions(ctx)
            if asyncio.iscoroutine(result):
                prompt = await result
            else:
                prompt = result
        else:
            prompt = self._instructions

        if self._output_schema is not None:
            parser = PydanticOutputParser(pydantic_object=self._output_schema)
            prompt = f"{prompt}\n\n{parser.get_format_instructions()}"

        return prompt

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
        ctx: Context,
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

        Drops tool call AIMessages that lack matching ToolMessages to
        prevent ``tool_call_id`` pairing errors from the LLM provider.
        """
        # Collect tool_call_ids that have a response
        responded_ids: set[str] = set()
        for event in events:
            if not event.partial and event.type == EventType.TOOL_RESPONSE:
                for tr in event.content.tool_responses:
                    if tr.tool_call_id:
                        responded_ids.add(tr.tool_call_id)

        messages: list[BaseMessage] = []
        for event in events:
            if event.partial:
                continue
            if event.type == EventType.USER_MESSAGE:
                messages.append(event.to_langchain_message())
            elif event.type == EventType.AGENT_MESSAGE:
                if event.has_tool_calls:
                    ids = [tc.tool_call_id for tc in event.tool_calls]
                    if ids and all(tid in responded_ids for tid in ids):
                        messages.append(event.to_langchain_message())
                elif event.text:
                    messages.append(event.to_langchain_message())
            elif event.type == EventType.TOOL_RESPONSE:
                messages.append(event.to_langchain_message())
        return messages

    async def _call_llm(
        self,
        llm: BaseChatModel,
        messages: list[BaseMessage],
        ctx: Context,
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

    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: Context | None = None,
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
        ctx : Context, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            Events emitted during execution, including partial streaming
            tokens, tool call/response events, and the final answer.
        """
        ctx = self._ensure_ctx(config, ctx)

        system_prompt = await self._resolve_instructions(ctx)
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # Rebuild conversation history from session events (multi-turn)
        if ctx.session and ctx.session.events:
            messages.extend(self._events_to_messages(ctx.session.events))

        messages.append(HumanMessage(content=input))

        llm = self._build_bound_llm()

        for _ in range(self.max_iterations):
            request = self._build_request(system_prompt, messages)

            # Apply planner instruction
            effective_prompt = self._apply_planner_instruction(
                system_prompt, ctx, request
            )
            prompt_changed = effective_prompt != system_prompt
            if prompt_changed and messages and isinstance(messages[0], SystemMessage):
                messages[0] = SystemMessage(content=effective_prompt)

            if self.before_model_callback:
                await self.before_model_callback(ctx, request)

            # Call LLM — yields partial events + final AIMessage
            raw_response: AIMessage | None = None
            try:
                async for item in self._call_llm(llm, messages, ctx):
                    if isinstance(item, AIMessage):
                        raw_response = item
                    else:
                        yield item  # partial event
            except Exception as exc:
                if self.on_model_error_callback:
                    recovery = await self.on_model_error_callback(ctx, request, exc)
                    if recovery is not None:
                        raw_response = AIMessage(content=recovery.text or "")
                    else:
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
                else:
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

            llm_response = LlmResponse.from_ai_message(raw_response)

            # Let the planner post-process the response
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

            # No tool calls -> final answer
            if not llm_response.has_tool_calls:
                answer_text = llm_response.text
                parts: list[TextPart | DataPart] = []
                if answer_text:
                    parts.append(TextPart(text=answer_text))

                # Parse structured output if schema is set
                if self._output_schema is not None:
                    structured_output = None
                    parser = PydanticOutputParser(pydantic_object=self._output_schema)
                    if answer_text:
                        try:
                            structured_output = parser.parse(answer_text)
                        except Exception:
                            pass
                    if structured_output is None:
                        structured_llm = self._build_structured_llm()
                        if structured_llm is not None:
                            try:
                                structured_output = await structured_llm.ainvoke(
                                    messages,
                                    config=ctx.run_config,
                                )
                            except Exception:
                                pass
                    if structured_output is not None:
                        parts.append(DataPart(data=structured_output.model_dump()))

                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content(parts=parts),
                    partial=False,
                    llm_response=llm_response,
                )
                return

            # Execute all tool calls in parallel
            tool_messages: list[ToolMessage] = []
            event_queue: asyncio.Queue[Event | None] = asyncio.Queue()

            # Inject event_callback so tools (e.g. AgentTool) can push
            # events in real-time while they run.
            tool_ctx = ctx.model_copy(
                update={"event_callback": event_queue.put_nowait}
            )

            def _tool_response(
                t_id: str, t_name: str, *, result: str | None = None, error: str | None = None,
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
                        transfer_to_agent=result.removeprefix(
                            TRANSFER_SENTINEL
                        ).strip()
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

            async def _execute_tool(tool_call: dict) -> tuple[Event, ToolMessage]:
                """Execute a single tool call and return response event + message."""
                t_name, t_args, t_id = tool_call["name"], tool_call["args"], tool_call["id"]

                tool = self._tools.get(t_name)
                if tool is None:
                    return _tool_response(
                        t_id,
                        t_name,
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
                    return _tool_response(t_id, t_name, result=str(result))
                except Exception as exc:
                    if self.after_tool_callback:
                        await self.after_tool_callback(ctx, t_name, None)
                    return _tool_response(t_id, t_name, error=str(exc))

            # Yield tool call events
            for tool_call in llm_response.tool_calls:
                yield self._emit_event(
                    ctx,
                    EventType.AGENT_MESSAGE,
                    content=Content(parts=[ToolCallPart(
                        tool_call_id=tool_call["id"],
                        tool_name=tool_call["name"],
                        args=tool_call["args"],
                    )]),
                    llm_response=llm_response,
                )

            # Run all tool calls concurrently, streaming intermediate
            # events (e.g. from AgentTool) in real time via the queue.
            async for item in gather_with_event_queue(
                [_execute_tool(tc) for tc in llm_response.tool_calls],
                event_queue,
            ):
                if isinstance(item, Event):
                    yield item  # intermediate event from child agent
                else:
                    response_event, tool_msg = item
                    yield response_event
                    tool_messages.append(tool_msg)

            messages.extend(tool_messages)

        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(
                f"Max iterations ({self.max_iterations}) reached without a final answer."
            ),
            metadata={"error": True},
        )
