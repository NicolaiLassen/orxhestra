"""Tests for BaseAgent: ainvoke, astream, find_agent, _emit_event."""

from __future__ import annotations

import pytest

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.events.event import Event, EventType
from orxhestra.models.part import Content


class StubAgent(BaseAgent):
    """Minimal agent that yields a fixed answer."""

    def __init__(self, name: str = "stub", answer: str = "hello", **kwargs):
        super().__init__(name=name, **kwargs)
        self._answer = answer

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(self._answer),
        )


class NoAnswerAgent(BaseAgent):
    """Agent that yields a non-final event (partial)."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("thinking..."),
            partial=True,
            turn_complete=False,
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
async def test_emit_event_sets_branch():
    """_emit_event includes branch from ctx."""
    agent = StubAgent(answer="branched")
    ctx = Context(session_id="s1", agent_name="root")
    child_ctx = ctx.derive(agent_name="stub")
    events = [e async for e in agent.astream("q", ctx=child_ctx)]
    assert events[0].branch == "stub"


def test_repr():
    agent = StubAgent(name="test_agent")
    assert "StubAgent" in repr(agent)
    assert "test_agent" in repr(agent)
