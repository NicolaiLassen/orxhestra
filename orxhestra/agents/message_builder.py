"""MessageBuilder — builds LangChain messages from session state.

Responsible for resolving instructions (static or dynamic), converting
session events into LangChain messages, and assembling the full
conversation history sent to the LLM on each turn.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser

from orxhestra.events.event import EventType
from orxhestra.events.filters import apply_compaction, should_include_event

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.events.event import Event

# Type alias for instruction providers — uses a string forward reference
# because InvocationContext is behind TYPE_CHECKING.
InstructionProvider = str | Callable[..., str | Awaitable[str]]

_DEFAULT_INSTRUCTIONS = """\
You are a helpful assistant. Answer the user's questions clearly and concisely.
When you have enough information to answer, provide a direct response.
Only use tools when necessary to complete the task.
"""

_PREV_CONTEXT_MAX_CHARS = 5000
_PREV_CONTEXT_TOTAL_MAX_CHARS = 10_000

# Maximum characters per tool response kept in the message history.
_TOOL_RESPONSE_MAX_CHARS = 30_000


def _build_previous_context(
    events: list[Event],
    max_chars: int = _PREV_CONTEXT_MAX_CHARS,
    total_max_chars: int = _PREV_CONTEXT_TOTAL_MAX_CHARS,
) -> list[HumanMessage]:
    """Build context messages from previous invocations' final responses.

    Deduplicates by agent name (keeps only the latest response per agent)
    and truncates long responses to prevent token explosion.

    Parameters
    ----------
    events : list[Event]
        Final response events from previous invocations
        (from ``ctx.get_previous_final_responses()``).
    max_chars : int
        Maximum characters per individual agent response.
    total_max_chars : int
        Total character budget across all agent responses.

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
        if total_chars >= total_max_chars:
            break

        remaining = total_max_chars - total_chars
        effective_max = min(max_chars, remaining)
        text = truncate_output(event.text, effective_max)

        content = f"[{agent}] said: {text}"
        messages.append(HumanMessage(content=content))
        total_chars += len(content)

    return messages


def _truncate_tool_message(
    msg: ToolMessage, max_chars: int = _TOOL_RESPONSE_MAX_CHARS,
) -> ToolMessage:
    """Truncate a ToolMessage if its content exceeds the limit.

    Parameters
    ----------
    msg : ToolMessage
        The tool message to potentially truncate.
    max_chars : int
        Maximum character count for the content.

    Returns
    -------
    ToolMessage
        Original message if within limit, or a new truncated copy.
    """
    content = msg.content
    if isinstance(content, str) and len(content) > max_chars:
        from orxhestra.tools.output import truncate_output

        content = truncate_output(content, max_chars)
        return ToolMessage(
            content=content,
            tool_call_id=msg.tool_call_id,
        )
    return msg


class MessageBuilder:
    """Builds LangChain messages for an LlmAgent turn.

    Handles instruction resolution (static strings, callables, template
    substitution), session-event-to-message conversion (with visibility
    filtering and compaction), and previous-invocation context injection.

    Parameters
    ----------
    instructions : InstructionProvider
        System prompt string or callable returning one.
    output_schema : type, optional
        Pydantic model whose format instructions are appended to the
        system prompt.
    include_contents : str
        ``'default'`` loads full filtered history; ``'none'`` skips
        history entirely.
    context_max_chars : int
        Max characters per previous agent response in context.
    context_total_max_chars : int
        Total character budget for all previous agent responses.
    tool_response_max_chars : int
        Max characters kept per tool response in message history.
    """

    def __init__(
        self,
        instructions: InstructionProvider = _DEFAULT_INSTRUCTIONS,
        output_schema: type | None = None,
        include_contents: str = "default",
        context_max_chars: int = _PREV_CONTEXT_MAX_CHARS,
        context_total_max_chars: int = _PREV_CONTEXT_TOTAL_MAX_CHARS,
        tool_response_max_chars: int = _TOOL_RESPONSE_MAX_CHARS,
    ) -> None:
        self._instructions = instructions
        self._output_schema = output_schema
        self._include_contents = include_contents
        self.context_max_chars = context_max_chars
        self.context_total_max_chars = context_total_max_chars
        self.tool_response_max_chars = tool_response_max_chars

    async def resolve_instructions(self, ctx: InvocationContext) -> str:
        """Resolve the system prompt from a string or instruction provider.

        Supports ``{key}`` template substitution from ``ctx.state``.
        Unknown keys are left as-is (no ``KeyError``).

        Parameters
        ----------
        ctx : InvocationContext
            Current invocation context (used for state templating and
            as the argument to callable instruction providers).

        Returns
        -------
        str
            The fully resolved system prompt.
        """
        if callable(self._instructions):
            result = self._instructions(ctx)
            if asyncio.iscoroutine(result):
                prompt: str = await result
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

    async def build_conversation_history(
        self, ctx: InvocationContext, input_text: str,
    ) -> tuple[str, list[BaseMessage]]:
        """Build the system prompt and full message list for the LLM.

        Parameters
        ----------
        ctx : InvocationContext
            The current invocation context.
        input_text : str
            The user message or task description.

        Returns
        -------
        tuple[str, list[BaseMessage]]
            Resolved system prompt and the conversation messages.
        """
        system_prompt: str = await self.resolve_instructions(ctx)
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if self._include_contents != "none":
            prev_responses = ctx.get_previous_final_responses()
            prev_messages = _build_previous_context(
                prev_responses,
                max_chars=self.context_max_chars,
                total_max_chars=self.context_total_max_chars,
            )
            messages.extend(prev_messages)

            filtered = ctx.get_events(
                current_branch=bool(ctx.branch),
                current_invocation=bool(ctx.branch),
            )
            inv_id = ctx.invocation_id
            filtered = [e for e in filtered if e.invocation_id != inv_id]
            if filtered:
                messages.extend(self.events_to_messages(filtered))

        messages.append(HumanMessage(content=input_text))
        return system_prompt, messages

    def events_to_messages(self, events: list[Event]) -> list[BaseMessage]:
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
                msg = event.to_langchain_message()
                messages.append(
                    _truncate_tool_message(msg, self.tool_response_max_chars)
                )

        return messages
