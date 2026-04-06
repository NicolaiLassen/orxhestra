"""Tests for hierarchical tracing via LangChain's AsyncCallbackManager."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from langchain_core.callbacks import AsyncCallbackHandler

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.agents.sequential_agent import SequentialAgent
from orxhestra.agents.tracing import end_agent_span, error_agent_span, start_agent_span
from orxhestra.events.event import EventType
from orxhestra.models.part import Content
from orxhestra.runner import Runner
from orxhestra.sessions.in_memory_session_service import InMemorySessionService


class SpanRecorder(AsyncCallbackHandler):
    """Records on_chain_start/on_chain_end calls with parent relationships."""

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self._open: dict[UUID, dict[str, Any]] = {}

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        record = {
            "name": serialized.get("name", ""),
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "type": "chain_start",
        }
        self._open[run_id] = record
        self.spans.append(record)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.spans.append({
            "run_id": run_id,
            "type": "chain_end",
        })

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.spans.append({
            "run_id": run_id,
            "type": "chain_error",
        })

    def started_names(self) -> list[str]:
        return [s["name"] for s in self.spans if s["type"] == "chain_start"]

    def parent_of(self, child_name: str) -> str | None:
        for s in self.spans:
            if s["type"] == "chain_start" and s["name"] == child_name:
                pid = s["parent_run_id"]
                if pid is None:
                    return None
                for p in self.spans:
                    if p["type"] == "chain_start" and p["run_id"] == pid:
                        return p["name"]
        return None


class StubAgent(BaseAgent):
    """Agent that yields a fixed answer."""

    def __init__(self, name: str = "stub", answer: str = "hello") -> None:
        super().__init__(name=name)
        self._answer = answer

    async def astream(self, input: str, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(self._answer),
        )


class TracedStubAgent(BaseAgent):
    """Stub agent that emits its own trace span (like LlmAgent would)."""

    def __init__(self, name: str = "traced_stub", answer: str = "ok") -> None:
        super().__init__(name=name)
        self._answer = answer

    async def astream(self, input: str, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        ctx, run_mgr = await start_agent_span(
            ctx, self.name, "TracedStubAgent", {"input": input},
        )
        try:
            yield self._emit_event(
                ctx,
                EventType.AGENT_MESSAGE,
                content=Content.from_text(self._answer),
            )
        except BaseException as exc:
            await error_agent_span(run_mgr, exc)
            raise
        else:
            await end_agent_span(run_mgr)


# ── Unit tests for start/end/error_agent_span ────────────────


@pytest.mark.asyncio
async def test_start_agent_span_no_callbacks() -> None:
    """When no callbacks are configured, ctx is returned unchanged."""
    ctx = InvocationContext(session_id="s1", agent_name="test")
    new_ctx, run_mgr = await start_agent_span(
        ctx, "test", "TestAgent", {"input": "hi"},
    )
    assert run_mgr is None
    assert new_ctx is ctx


@pytest.mark.asyncio
async def test_start_agent_span_empty_callbacks() -> None:
    """Empty callbacks list is treated as no callbacks."""
    ctx = InvocationContext(
        session_id="s1", agent_name="test", run_config={"callbacks": []},
    )
    new_ctx, run_mgr = await start_agent_span(
        ctx, "test", "TestAgent", {"input": "hi"},
    )
    assert run_mgr is None


@pytest.mark.asyncio
async def test_start_agent_span_creates_child_manager() -> None:
    """With callbacks, a child manager replaces run_config callbacks."""
    recorder = SpanRecorder()
    ctx = InvocationContext(
        session_id="s1",
        agent_name="test",
        run_config={"callbacks": [recorder]},
    )
    new_ctx, run_mgr = await start_agent_span(
        ctx, "MyAgent", "LlmAgent", {"input": "hi"},
    )
    assert run_mgr is not None
    assert new_ctx is not ctx
    # The new config's callbacks should be a child manager, not the original list
    assert new_ctx.run_config["callbacks"] is not ctx.run_config["callbacks"]

    await end_agent_span(run_mgr)

    assert recorder.started_names() == ["MyAgent"]


@pytest.mark.asyncio
async def test_error_agent_span_records_error() -> None:
    """Error span fires on_chain_error."""
    recorder = SpanRecorder()
    ctx = InvocationContext(
        session_id="s1",
        agent_name="test",
        run_config={"callbacks": [recorder]},
    )
    _, run_mgr = await start_agent_span(
        ctx, "FailAgent", "LlmAgent", {"input": "hi"},
    )
    await error_agent_span(run_mgr, RuntimeError("boom"))

    error_events = [s for s in recorder.spans if s["type"] == "chain_error"]
    assert len(error_events) == 1


# ── Integration: Runner root span ────────────────────────────


@pytest.mark.asyncio
async def test_runner_creates_root_span() -> None:
    """Runner wraps agent execution in a root trace span."""
    recorder = SpanRecorder()
    runner = Runner(
        agent=StubAgent(answer="hi"),
        app_name="test-app",
        session_service=InMemorySessionService(),
    )

    events = [
        e async for e in runner.astream(
            user_id="u1",
            session_id="s1",
            new_message="hello",
            config={"callbacks": [recorder]},
        )
    ]

    assert len(events) == 1
    assert events[0].text == "hi"

    names = recorder.started_names()
    assert any("Runner:" in n for n in names)

    end_events = [s for s in recorder.spans if s["type"] == "chain_end"]
    assert len(end_events) >= 1


# ── Integration: nested spans via SequentialAgent ────────────


@pytest.mark.asyncio
async def test_sequential_agent_nested_spans() -> None:
    """SequentialAgent with traced children creates nested spans."""
    recorder = SpanRecorder()

    agent1 = TracedStubAgent(name="agent_a", answer="result_a")
    agent2 = TracedStubAgent(name="agent_b", answer="result_b")
    pipeline = SequentialAgent(
        name="pipeline", agents=[agent1, agent2],
    )

    runner = Runner(
        agent=pipeline,
        app_name="test-app",
        session_service=InMemorySessionService(),
    )

    events = [
        e async for e in runner.astream(
            user_id="u1",
            session_id="s1",
            new_message="go",
            config={"callbacks": [recorder]},
        )
    ]

    assert len(events) == 2

    names = recorder.started_names()
    # Expect: Runner:pipeline, pipeline, agent_a, agent_b
    assert "pipeline" in names
    assert "agent_a" in names
    assert "agent_b" in names

    # Verify nesting: agent_a's parent should be pipeline
    assert recorder.parent_of("agent_a") == "pipeline"
    assert recorder.parent_of("agent_b") == "pipeline"

    # pipeline's parent should be the Runner span
    pipeline_parent = recorder.parent_of("pipeline")
    assert pipeline_parent is not None
    assert "Runner:" in pipeline_parent


# ── No-callbacks path has zero overhead ──────────────────────


@pytest.mark.asyncio
async def test_no_callbacks_no_overhead() -> None:
    """Without callbacks, agents produce events normally."""
    runner = Runner(
        agent=TracedStubAgent(answer="ok"),
        app_name="app",
        session_service=InMemorySessionService(),
    )

    events = [
        e async for e in runner.astream(
            user_id="u1", session_id="s1", new_message="hi",
        )
    ]

    assert len(events) == 1
    assert events[0].text == "ok"
