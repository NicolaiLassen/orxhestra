"""Tests for branch attribution through orchestrator agents."""

import pytest

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.loop_agent import LoopAgent
from langchain_adk.agents.parallel_agent import ParallelAgent
from langchain_adk.agents.sequential_agent import SequentialAgent
from langchain_adk.events.event import EventType
from langchain_adk.events.event_actions import EventActions
from langchain_adk.models.part import Content


class EchoAgent(BaseAgent):
    """Agent that echoes its input, carrying branch info from ctx."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(f"echo:{input}"),
        )


class EscalatingAgent(BaseAgent):
    """Agent that emits one event then escalates."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("done"),
            actions=EventActions(escalate=True),
        )


@pytest.mark.asyncio
async def test_sequential_branch_attribution():
    """Each sub-agent in a SequentialAgent gets its own branch."""
    a = EchoAgent(name="alpha")
    b = EchoAgent(name="beta")
    seq = SequentialAgent(name="pipeline", agents=[a, b])

    events = [e async for e in seq.astream("hello")]
    assert len(events) == 2
    assert events[0].branch == "alpha"
    assert events[0].agent_name == "alpha"
    assert events[1].branch == "beta"
    assert events[1].agent_name == "beta"


@pytest.mark.asyncio
async def test_parallel_branch_attribution():
    """Each sub-agent in a ParallelAgent gets its own branch."""
    a = EchoAgent(name="left")
    b = EchoAgent(name="right")
    par = ParallelAgent(name="fan-out", agents=[a, b])

    events = [e async for e in par.astream("hello")]
    assert len(events) == 2
    branches = {e.branch for e in events}
    assert branches == {"left", "right"}


@pytest.mark.asyncio
async def test_loop_branch_attribution():
    """Sub-agents in a LoopAgent get branch attribution each iteration."""
    agent = EscalatingAgent(name="worker")
    loop = LoopAgent(name="loop", agents=[agent], max_iterations=5)

    events = [e async for e in loop.astream("go")]
    assert len(events) == 1
    assert events[0].branch == "worker"


@pytest.mark.asyncio
async def test_nested_branch_attribution():
    """Nested orchestrators build dot-separated branch paths."""
    inner_a = EchoAgent(name="inner-a")
    inner_b = EchoAgent(name="inner-b")
    inner_seq = SequentialAgent(name="inner", agents=[inner_a, inner_b])
    outer = SequentialAgent(name="outer", agents=[inner_seq])

    events = [e async for e in outer.astream("hello")]
    # inner_seq gets branch "inner", then inner_a gets "inner.inner-a"
    assert events[0].branch == "inner.inner-a"
    assert events[1].branch == "inner.inner-b"
