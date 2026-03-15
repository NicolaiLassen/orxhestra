"""Tests for LlmAgent: tool loop, streaming, sentinels."""

import pytest
from typing import Any, List, Optional
from unittest.mock import AsyncMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_adk.agents.llm_agent import LlmAgent
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event, EventType
from langchain_adk.tools.exit_loop import EXIT_LOOP_SENTINEL
from langchain_adk.tools.transfer_tool import TRANSFER_SENTINEL


class FakeChatModel(BaseChatModel):
    """Minimal fake chat model for testing."""

    responses: list[AIMessage]
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        """No-op bind_tools — just return self."""
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        msg = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, **kwargs)


def _ctx(**kwargs) -> Context:
    defaults = {"session_id": "test", "agent_name": "agent"}
    defaults.update(kwargs)
    return Context(**defaults)


def _llm(*texts: str) -> FakeChatModel:
    """Create a FakeChatModel that returns the given text responses in order."""
    return FakeChatModel(responses=[AIMessage(content=t) for t in texts])


@pytest.mark.asyncio
async def test_simple_text_response():
    agent = LlmAgent(name="agent", llm=_llm("Hello world"))
    events = [e async for e in agent.astream("hi", ctx=_ctx())]
    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "Hello world"


@pytest.mark.asyncio
async def test_tool_call_and_result():
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # First response: tool call, second response: final text
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "add", "args": {"a": 2, "b": 3}}],
    )
    final_msg = AIMessage(content="The answer is 5")
    llm = FakeChatModel(responses=[tool_msg, final_msg])

    agent = LlmAgent(name="agent", llm=llm, tools=[add])
    events = [e async for e in agent.astream("add 2+3", ctx=_ctx())]

    tool_calls = [e for e in events if e.has_tool_calls]
    tool_results = [e for e in events if e.type == EventType.TOOL_RESPONSE]
    finals = [e for e in events if e.is_final_response()]

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_calls[0].tool_name == "add"
    assert len(tool_results) == 1
    assert "5" in tool_results[0].content.tool_responses[0].result
    assert len(finals) == 1
    assert "5" in finals[0].text


@pytest.mark.asyncio
async def test_tool_not_found():
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "nonexistent", "args": {}}],
    )
    final_msg = AIMessage(content="ok")
    llm = FakeChatModel(responses=[tool_msg, final_msg])

    agent = LlmAgent(name="agent", llm=llm)
    events = [e async for e in agent.astream("call bad tool", ctx=_ctx())]

    results = [e for e in events if e.type == EventType.TOOL_RESPONSE]
    assert len(results) == 1
    assert "not found" in results[0].content.tool_responses[0].error


@pytest.mark.asyncio
async def test_max_iterations_error():
    # LLM always returns a tool call → hits max iterations
    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "nonexistent", "args": {}}],
    )
    llm = FakeChatModel(responses=[tool_msg])

    agent = LlmAgent(name="agent", llm=llm, max_iterations=2)
    events = [e async for e in agent.astream("loop", ctx=_ctx())]

    errors = [e for e in events if e.metadata.get("error")]
    assert len(errors) == 1
    assert "Max iterations" in errors[0].text


@pytest.mark.asyncio
async def test_transfer_sentinel():
    from langchain_core.tools import tool

    @tool
    def transfer(agent_name: str) -> str:
        """Transfer to agent."""
        return f"{TRANSFER_SENTINEL}{agent_name}"

    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "transfer", "args": {"agent_name": "TargetAgent"}}],
    )
    final_msg = AIMessage(content="transferred")
    llm = FakeChatModel(responses=[tool_msg, final_msg])

    agent = LlmAgent(name="agent", llm=llm, tools=[transfer])
    events = [e async for e in agent.astream("transfer", ctx=_ctx())]

    results = [e for e in events if e.type == EventType.TOOL_RESPONSE]
    assert len(results) == 1
    assert results[0].actions.transfer_to_agent == "TargetAgent"


@pytest.mark.asyncio
async def test_exit_loop_sentinel():
    from langchain_core.tools import tool

    @tool
    def exit_loop() -> str:
        """Exit loop."""
        return EXIT_LOOP_SENTINEL

    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "exit_loop", "args": {}}],
    )
    final_msg = AIMessage(content="done")
    llm = FakeChatModel(responses=[tool_msg, final_msg])

    agent = LlmAgent(name="agent", llm=llm, tools=[exit_loop])
    events = [e async for e in agent.astream("exit", ctx=_ctx())]

    results = [e for e in events if e.type == EventType.TOOL_RESPONSE]
    assert len(results) == 1
    assert results[0].actions.escalate is True


@pytest.mark.asyncio
async def test_before_after_model_callbacks():
    calls = []

    async def before(ctx, req):
        calls.append("before_model")

    async def after(ctx, resp):
        calls.append("after_model")

    agent = LlmAgent(
        name="agent",
        llm=_llm("done"),
        before_model_callback=before,
        after_model_callback=after,
    )
    events = [e async for e in agent.astream("hi", ctx=_ctx())]
    assert "before_model" in calls
    assert "after_model" in calls


@pytest.mark.asyncio
async def test_custom_instructions():
    agent = LlmAgent(
        name="agent",
        llm=_llm("response"),
        instructions="You are a pirate.",
    )
    events = [e async for e in agent.astream("hi", ctx=_ctx())]
    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1


@pytest.mark.asyncio
async def test_callable_instructions():
    def get_instructions(ctx):
        return f"You are agent {ctx.agent_name}"

    agent = LlmAgent(
        name="agent",
        llm=_llm("response"),
        instructions=get_instructions,
    )
    events = [e async for e in agent.astream("hi", ctx=_ctx())]
    assert any(e.is_final_response() for e in events)


@pytest.mark.asyncio
async def test_multi_turn_history_from_session():
    """LlmAgent should rebuild conversation history from session events."""
    from langchain_adk.events.event import Event, EventType
    from langchain_adk.models.part import Content
    from langchain_adk.sessions.session import Session

    # Create a session with previous conversation events
    session = Session(app_name="test", user_id="user1")
    session.events.append(Event(
        type=EventType.USER_MESSAGE,
        author="user",
        session_id=session.id,
        content=Content.from_text("My name is Alice"),
    ))
    session.events.append(Event(
        type=EventType.AGENT_MESSAGE,
        session_id=session.id,
        agent_name="agent",
        content=Content.from_text("Hello Alice!"),
        partial=False,
    ))

    # Track what messages the LLM receives
    received_messages: list[BaseMessage] = []

    class SpyLlm(FakeChatModel):
        def _generate(self, messages, stop=None, **kwargs):
            received_messages.extend(messages)
            return super()._generate(messages, stop, **kwargs)

    llm = SpyLlm(responses=[AIMessage(content="Your name is Alice")])
    agent = LlmAgent(name="agent", llm=llm)

    ctx = _ctx(session=session)
    events = [e async for e in agent.astream("What is my name?", ctx=ctx)]

    # Should have: SystemMessage, HumanMessage("My name is Alice"),
    # AIMessage("Hello Alice!"), HumanMessage("What is my name?")
    human_msgs = [m for m in received_messages if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in received_messages if isinstance(m, AIMessage)]
    assert len(human_msgs) == 2  # previous + current
    assert human_msgs[0].content == "My name is Alice"
    assert human_msgs[1].content == "What is my name?"
    assert len(ai_msgs) == 1
    assert ai_msgs[0].content == "Hello Alice!"


@pytest.mark.asyncio
async def test_tool_call_id_in_parts():
    """Tool call events should carry tool_call_id in ToolCallPart/ToolResponsePart."""
    from langchain_core.tools import tool

    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hi {name}"

    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_123", "name": "greet", "args": {"name": "Bob"}}],
    )
    final_msg = AIMessage(content="Done")
    llm = FakeChatModel(responses=[tool_msg, final_msg])

    agent = LlmAgent(name="agent", llm=llm, tools=[greet])
    events = [e async for e in agent.astream("greet Bob", ctx=_ctx())]

    tool_calls = [e for e in events if e.has_tool_calls]
    tool_results = [e for e in events if e.type == EventType.TOOL_RESPONSE]

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_calls[0].tool_call_id == "call_123"
    assert len(tool_results) == 1
    assert tool_results[0].content.tool_responses[0].tool_call_id == "call_123"
