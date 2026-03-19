"""Tests for Runner - session-managed agent execution."""

import pytest

from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.events.event import EventType
from langchain_adk.events.event_actions import EventActions
from langchain_adk.models.part import Content
from langchain_adk.runner import Runner
from langchain_adk.sessions.in_memory_session_service import InMemorySessionService


class StubAgent(BaseAgent):
    """Agent that yields a fixed answer."""

    def __init__(self, name: str = "stub", answer: str = "hello"):
        super().__init__(name=name)
        self._answer = answer

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text(self._answer),
        )


class StateDeltaAgent(BaseAgent):
    """Agent that emits a state delta."""

    async def astream(self, input, config=None, *, ctx=None):
        ctx = self._ensure_ctx(config, ctx)
        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            content=Content.from_text("updated"),
            actions=EventActions(state_delta={"key": "value"}),
        )


@pytest.mark.asyncio
async def test_runner_creates_session():
    runner = Runner(
        agent=StubAgent(answer="hi"),
        app_name="test-app",
        session_service=InMemorySessionService(),
    )
    events = [
        e async for e in runner.astream(
            user_id="user-1", session_id="s1", new_message="hello"
        )
    ]
    assert len(events) == 1
    assert events[0].text == "hi"
    assert events[0].is_final_response()


@pytest.mark.asyncio
async def test_runner_persists_user_message():
    svc = InMemorySessionService()
    runner = Runner(agent=StubAgent(), app_name="app", session_service=svc)

    async for _ in runner.astream(
        user_id="u1", session_id="s1", new_message="hello"
    ):
        pass

    session = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert session is not None
    # First event is user message, second is agent response
    assert len(session.events) == 2
    assert session.events[0].type == EventType.USER_MESSAGE
    assert session.events[0].text == "hello"
    assert session.events[1].type == EventType.AGENT_MESSAGE


@pytest.mark.asyncio
async def test_runner_persists_agent_events():
    svc = InMemorySessionService()
    runner = Runner(agent=StubAgent(answer="world"), app_name="app", session_service=svc)

    async for _ in runner.astream(
        user_id="u1", session_id="s1", new_message="hi"
    ):
        pass

    session = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    agent_events = [e for e in session.events if e.type == EventType.AGENT_MESSAGE]
    assert len(agent_events) == 1
    assert agent_events[0].text == "world"


@pytest.mark.asyncio
async def test_runner_applies_state_delta():
    svc = InMemorySessionService()
    runner = Runner(agent=StateDeltaAgent(name="delta"), app_name="app", session_service=svc)

    async for _ in runner.astream(
        user_id="u1", session_id="s1", new_message="go"
    ):
        pass

    session = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    assert session.state["key"] == "value"


@pytest.mark.asyncio
async def test_runner_reuses_existing_session():
    svc = InMemorySessionService()
    runner = Runner(agent=StubAgent(answer="turn1"), app_name="app", session_service=svc)

    # First turn
    async for _ in runner.astream(
        user_id="u1", session_id="s1", new_message="first"
    ):
        pass

    # Second turn — same session
    runner2 = Runner(agent=StubAgent(answer="turn2"), app_name="app", session_service=svc)
    async for _ in runner2.astream(
        user_id="u1", session_id="s1", new_message="second"
    ):
        pass

    session = await svc.get_session(app_name="app", user_id="u1", session_id="s1")
    # 2 user messages + 2 agent responses = 4 events
    assert len(session.events) == 4


@pytest.mark.asyncio
async def test_runner_get_or_create_session():
    svc = InMemorySessionService()
    runner = Runner(agent=StubAgent(), app_name="app", session_service=svc)

    # Creates new session
    session = await runner.get_or_create_session(user_id="u1", session_id="s1")
    assert session.id == "s1"
    assert session.user_id == "u1"

    # Returns existing session
    session2 = await runner.get_or_create_session(user_id="u1", session_id="s1")
    assert session2.id == session.id


@pytest.mark.asyncio
async def test_runner_passes_config_to_agent():
    """Config dict reaches the agent context via run_config."""
    received_config = {}

    class ConfigCapture(BaseAgent):
        async def astream(self, input, config=None, *, ctx=None):
            ctx = self._ensure_ctx(config, ctx)
            received_config.update(ctx.run_config)
            yield self._emit_event(
                ctx, EventType.AGENT_MESSAGE, content=Content.from_text("ok")
            )

    svc = InMemorySessionService()
    runner = Runner(agent=ConfigCapture(name="cfg"), app_name="app", session_service=svc)

    async for _ in runner.astream(
        user_id="u1",
        session_id="s1",
        new_message="hi",
        config={"tags": ["test"]},
    ):
        pass

    assert received_config.get("tags") == ["test"]
