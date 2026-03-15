"""Tests for BaseAgent: ainvoke, astream, find_agent, callbacks."""

import pytest

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event, EventType
from langchain_adk.models.part import Content


class StubAgent(BaseAgent):
    """Minimal agent that yields a fixed answer."""

    def __init__(self, name: str = "stub", answer: str = "hello", **kwargs):
        super().__init__(name=name, **kwargs)
        self._answer = answer

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield Event(
            type=EventType.AGENT_MESSAGE,
            session_id=ctx.session_id,
            agent_name=self.name,
            content=Content.from_text(self._answer),
        )


class NoAnswerAgent(BaseAgent):
    """Agent that yields no final answer event."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield Event(
            type=EventType.AGENT_START,
            session_id=ctx.session_id,
            agent_name=self.name,
        )


@pytest.mark.asyncio
async def test_ainvoke_returns_final_answer():
    agent = StubAgent(answer="42")
    result = await agent.ainvoke("question")
    assert isinstance(result, Event)
    assert result.text == "42"
    assert result.is_final_response()


@pytest.mark.asyncio
async def test_ainvoke_raises_on_no_answer():
    agent = NoAnswerAgent(name="empty")
    with pytest.raises(RuntimeError, match="no final answer"):
        await agent.ainvoke("question")


@pytest.mark.asyncio
async def test_astream_yields_all_events():
    agent = StubAgent(answer="streamed")
    events = [e async for e in agent.astream("q")]
    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "streamed"


def test_find_agent_self():
    agent = StubAgent(name="root")
    assert agent.find_agent("root") is agent


def test_find_agent_child():
    parent = StubAgent(name="parent")
    child = StubAgent(name="child")
    parent.register_sub_agent(child)
    assert parent.find_agent("child") is child
    assert child.parent_agent is parent


def test_find_agent_nested():
    root = StubAgent(name="root")
    mid = StubAgent(name="mid")
    leaf = StubAgent(name="leaf")
    root.register_sub_agent(mid)
    mid.register_sub_agent(leaf)
    assert root.find_agent("leaf") is leaf


def test_find_agent_not_found():
    agent = StubAgent(name="root")
    assert agent.find_agent("nonexistent") is None


def test_root_agent():
    root = StubAgent(name="root")
    child = StubAgent(name="child")
    grandchild = StubAgent(name="grandchild")
    root.register_sub_agent(child)
    child.register_sub_agent(grandchild)
    assert grandchild.root_agent is root
    assert child.root_agent is root
    assert root.root_agent is root


@pytest.mark.asyncio
async def test_astream_with_ctx():
    """Test that providing an explicit ctx works."""
    agent = StubAgent(answer="with-ctx")
    ctx = Context(session_id="test", agent_name="stub")
    events = [e async for e in agent.astream("q", ctx=ctx)]
    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "with-ctx"


@pytest.mark.asyncio
async def test_before_after_callbacks():
    """Callbacks are called via run_async (queue-based wrapper)."""
    agent = StubAgent(answer="cb")
    calls = []

    async def before(ctx):
        calls.append("before")

    async def after(ctx):
        calls.append("after")

    agent.before_agent_callback = before
    agent.after_agent_callback = after

    # Callbacks are triggered by run_async, not astream directly
    ctx = Context(session_id="test", agent_name="stub")
    import asyncio
    task = asyncio.create_task(agent.run_async("q", ctx=ctx))
    events = []
    while True:
        event = await ctx.events.get()
        if event is None:
            break
        events.append(event)
    await task

    assert "before" in calls
    assert "after" in calls
    types = [e.type for e in events]
    assert EventType.AGENT_START in types
    assert EventType.AGENT_END in types


def test_repr():
    agent = StubAgent(name="test_agent")
    assert "StubAgent" in repr(agent)
    assert "test_agent" in repr(agent)
