"""Tests for LlmAgent context limits and truncation."""

import logging

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.agents.llm_agent import LlmAgent
from orxhestra.agents.message_builder import (
    _build_previous_context,
    _truncate_tool_message,
)
from orxhestra.events.event import EventType


class FakeChatModel(BaseChatModel):
    """Minimal fake chat model for testing."""

    responses: list[AIMessage]
    call_count: int = 0
    received_messages: list[BaseMessage] = []

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs,
    ) -> ChatResult:
        self.received_messages = list(messages)
        msg = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)


def _ctx(**kwargs) -> Context:
    defaults = {"session_id": "test", "agent_name": "agent"}
    defaults.update(kwargs)
    return Context(**defaults)


# ── _truncate_tool_message ───────────────────────────────────


def test_truncate_tool_message_under_limit():
    """Messages under the limit are returned unchanged."""
    msg = ToolMessage(content="short", tool_call_id="id1")
    result = _truncate_tool_message(msg, max_chars=100)
    assert result.content == "short"
    assert result.tool_call_id == "id1"


def test_truncate_tool_message_over_limit():
    """Messages over the limit are truncated."""
    content = "x" * 50_000
    msg = ToolMessage(content=content, tool_call_id="id1")
    result = _truncate_tool_message(msg, max_chars=1000)
    assert len(result.content) <= 1100  # small overhead from suffix
    assert result.tool_call_id == "id1"


def test_truncate_tool_message_custom_limit():
    """Custom max_chars is respected."""
    content = "a" * 500
    msg = ToolMessage(content=content, tool_call_id="id1")
    result = _truncate_tool_message(msg, max_chars=200)
    assert len(result.content) < 500


def test_truncate_tool_message_non_string_content():
    """Non-string content is returned unchanged."""
    msg = ToolMessage(content=[{"type": "text"}], tool_call_id="id1")
    result = _truncate_tool_message(msg, max_chars=10)
    assert result.content == [{"type": "text"}]


# ── _build_previous_context ─────────────────────────────────


def test_build_previous_context_empty():
    """Empty events list returns empty messages."""
    assert _build_previous_context([]) == []


def test_build_previous_context_truncates_long_responses():
    """Long responses are truncated to max_chars."""
    from orxhestra.events.event import Event
    from orxhestra.models.part import Content

    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id="s1",
        agent_name="researcher",
        content=Content.from_text("x" * 10_000),
        partial=False,
    )

    messages = _build_previous_context(
        [event], max_chars=100, total_max_chars=500,
    )
    assert len(messages) == 1
    assert len(messages[0].content) < 200  # [researcher] said: + truncated


def test_build_previous_context_respects_total_budget():
    """Total budget caps how many agent responses are included."""
    from orxhestra.events.event import Event
    from orxhestra.models.part import Content

    events = [
        Event(
            type=EventType.AGENT_MESSAGE,
            session_id="s1",
            agent_name=f"agent_{i}",
            content=Content.from_text("data " * 100),
            partial=False,
        )
        for i in range(10)
    ]

    messages = _build_previous_context(
        events, max_chars=200, total_max_chars=500,
    )
    total_chars = sum(len(m.content) for m in messages)
    assert total_chars <= 700  # some overhead allowed


def test_build_previous_context_deduplicates_by_agent():
    """Only the latest response per agent is kept."""
    from orxhestra.events.event import Event
    from orxhestra.models.part import Content

    events = [
        Event(
            type=EventType.AGENT_MESSAGE,
            session_id="s1",
            agent_name="researcher",
            content=Content.from_text("old response"),
            partial=False,
        ),
        Event(
            type=EventType.AGENT_MESSAGE,
            session_id="s1",
            agent_name="researcher",
            content=Content.from_text("new response"),
            partial=False,
        ),
    ]

    messages = _build_previous_context(events, max_chars=5000, total_max_chars=10_000)
    assert len(messages) == 1
    assert "new response" in messages[0].content


# ── Configurable limits on LlmAgent ─────────────────────────


def test_default_limits():
    """Default limits match expected values."""
    llm = FakeChatModel(responses=[AIMessage(content="hi")])
    agent = LlmAgent(name="agent", llm=llm)
    assert agent.tool_response_max_chars == 30_000
    assert agent.context_max_chars == 5000
    assert agent.context_total_max_chars == 10_000


def test_custom_limits():
    """Custom limits are stored on the agent."""
    llm = FakeChatModel(responses=[AIMessage(content="hi")])
    agent = LlmAgent(
        name="agent",
        llm=llm,
        tool_response_max_chars=50_000,
        context_max_chars=8000,
        context_total_max_chars=20_000,
    )
    assert agent.tool_response_max_chars == 50_000
    assert agent.context_max_chars == 8000
    assert agent.context_total_max_chars == 20_000


@pytest.mark.asyncio
async def test_tool_response_truncated_in_history():
    """Large tool responses are truncated before being added to message history."""
    from langchain_core.tools import tool

    @tool
    def big_search(query: str) -> str:
        """Return a huge result."""
        return "x" * 100_000

    tool_msg = AIMessage(
        content="",
        tool_calls=[{"id": "tc1", "name": "big_search", "args": {"query": "test"}}],
    )
    final_msg = AIMessage(content="Summary")
    llm = FakeChatModel(responses=[tool_msg, final_msg])

    agent = LlmAgent(
        name="agent",
        llm=llm,
        tools=[big_search],
        tool_response_max_chars=5000,
    )
    _ = [e async for e in agent.astream("search", ctx=_ctx())]

    # The LLM should have received truncated tool output in its messages
    tool_msgs = [m for m in llm.received_messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert len(tool_msgs[0].content) <= 5100  # small overhead


@pytest.mark.asyncio
async def test_context_budget_warning(caplog):
    """Warning is logged when message context exceeds 200k chars."""
    llm = FakeChatModel(responses=[AIMessage(content="ok")])
    agent = LlmAgent(
        name="agent",
        llm=llm,
        instructions="x" * 250_000,
    )

    with caplog.at_level(logging.DEBUG, logger="orxhestra.agents.llm_agent"):
        _ = [e async for e in agent.astream("hi", ctx=_ctx())]

    assert any("Message context is" in r.message for r in caplog.records)


# ── Planner continuation ────────────────────────────────────


@pytest.mark.asyncio
async def test_planner_pending_tasks_continues_loop():
    """Agent continues the loop when planner has pending tasks."""
    from orxhestra.planners.base_planner import BasePlanner

    class StubPlanner(BasePlanner):
        call_count: int = 0

        def build_planning_instruction(self, ctx, req):
            return "Keep working."

        def has_pending_tasks(self, ctx):
            self.call_count += 1
            return self.call_count <= 1  # pending first time, done second

        def process_planning_response(self, ctx, resp):
            return None

    planner = StubPlanner()

    # First response: text only (no tools) → planner says pending → continue
    # Second response: text only → planner says done → final
    llm = FakeChatModel(responses=[
        AIMessage(content="Working on it..."),
        AIMessage(content="All done!"),
    ])

    agent = LlmAgent(name="agent", llm=llm, planner=planner)
    events = [e async for e in agent.astream("do tasks", ctx=_ctx())]

    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "All done!"
    assert llm.call_count == 2  # called twice because planner continued


@pytest.mark.asyncio
async def test_planner_no_pending_returns_immediately():
    """Agent returns immediately when planner has no pending tasks."""
    from orxhestra.planners.base_planner import BasePlanner

    class DonePlanner(BasePlanner):
        def build_planning_instruction(self, ctx, req):
            return None

        def has_pending_tasks(self, ctx):
            return False

    llm = FakeChatModel(responses=[AIMessage(content="Done")])
    agent = LlmAgent(name="agent", llm=llm, planner=DonePlanner())
    events = [e async for e in agent.astream("hi", ctx=_ctx())]

    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "Done"
    assert llm.call_count == 1
