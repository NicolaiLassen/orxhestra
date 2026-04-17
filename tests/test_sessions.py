"""Tests for Session and InMemorySessionService."""

from __future__ import annotations

import pytest

from orxhestra.events.event import Event, EventType
from orxhestra.events.event_actions import EventActions
from orxhestra.models.part import Content
from orxhestra.sessions.in_memory_session_service import InMemorySessionService


@pytest.fixture
def service():
    return InMemorySessionService()


@pytest.mark.asyncio
async def test_create_session(service):
    session = await service.create_session(app_name="app", user_id="u1")
    assert session.app_name == "app"
    assert session.user_id == "u1"
    assert session.state == {}
    assert session.events == []
    assert session.id is not None


@pytest.mark.asyncio
async def test_create_session_with_state(service):
    session = await service.create_session(
        app_name="app", user_id="u1", state={"key": "val"}
    )
    assert session.state["key"] == "val"


@pytest.mark.asyncio
async def test_create_session_with_explicit_id(service):
    session = await service.create_session(
        app_name="app", user_id="u1", session_id="fixed-id"
    )
    assert session.id == "fixed-id"


@pytest.mark.asyncio
async def test_get_session(service):
    created = await service.create_session(app_name="app", user_id="u1")
    fetched = await service.get_session(
        app_name="app", user_id="u1", session_id=created.id
    )
    assert fetched is not None
    assert fetched.id == created.id


@pytest.mark.asyncio
async def test_get_session_not_found(service):
    result = await service.get_session(
        app_name="app", user_id="u1", session_id="nonexistent"
    )
    assert result is None


@pytest.mark.asyncio
async def test_delete_session(service):
    session = await service.create_session(app_name="app", user_id="u1")
    await service.delete_session(session.id)
    result = await service.get_session(
        app_name="app", user_id="u1", session_id=session.id
    )
    assert result is None


@pytest.mark.asyncio
async def test_list_sessions(service):
    await service.create_session(app_name="app", user_id="u1")
    await service.create_session(app_name="app", user_id="u1")
    await service.create_session(app_name="app", user_id="u2")

    sessions = await service.list_sessions(app_name="app", user_id="u1")
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_append_event_applies_state_delta(service):
    session = await service.create_session(app_name="app", user_id="u1")
    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id=session.id,
        agent_name="agent",
        content=Content.from_text("done"),
        actions=EventActions(state_delta={"result": "done"}),
    )
    await service.append_event(session, event)
    assert session.state["result"] == "done"
    assert len(session.events) == 1
    assert session.last_update_time > 0


@pytest.mark.asyncio
async def test_update_session_state(service):
    session = await service.create_session(app_name="app", user_id="u1")
    updated = await service.update_session(session.id, state={"foo": "bar"})
    assert updated.state["foo"] == "bar"


@pytest.mark.asyncio
async def test_append_user_message_event(service):
    """USER_MESSAGE events should be persisted in session.events."""
    session = await service.create_session(app_name="app", user_id="u1")
    user_event = Event(
        type=EventType.USER_MESSAGE,
        author="user",
        session_id=session.id,
        content=Content.from_text("Hello agent"),
    )
    await service.append_event(session, user_event)
    assert len(session.events) == 1
    assert session.events[0].type == EventType.USER_MESSAGE
    assert session.events[0].text == "Hello agent"
