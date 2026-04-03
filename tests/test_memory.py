"""Tests for Memory, BaseMemoryService, and InMemoryMemoryService."""

import pytest

from orxhestra.events.event import Event, EventType
from orxhestra.memory.base_memory_service import SearchMemoryResponse
from orxhestra.memory.in_memory_service import InMemoryMemoryService
from orxhestra.memory.memory import Memory
from orxhestra.models.part import Content
from orxhestra.sessions.session import Session


def _make_session_with_events():
    session = Session(app_name="app", user_id="u1")
    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id=session.id,
        agent_name="agent",
        content=Content.from_text("The capital of France is Paris."),
    )
    session.events.append(event)
    return session


def test_memory_defaults():
    m = Memory(content="test content")
    assert m.content == "test content"
    assert m.author is None
    assert m.metadata == {}


@pytest.mark.asyncio
async def test_add_session_to_memory():
    service = InMemoryMemoryService()
    session = _make_session_with_events()
    await service.add_session_to_memory(session)

    response = await service.search_memory(
        app_name="app", user_id="u1", query="Paris"
    )
    assert len(response.memories) >= 1
    assert any("Paris" in m.content for m in response.memories)


@pytest.mark.asyncio
async def test_search_memory_no_match():
    service = InMemoryMemoryService()
    session = _make_session_with_events()
    await service.add_session_to_memory(session)

    response = await service.search_memory(
        app_name="app", user_id="u1", query="Berlin"
    )
    assert len(response.memories) == 0


@pytest.mark.asyncio
async def test_search_memory_wrong_user():
    service = InMemoryMemoryService()
    session = _make_session_with_events()
    await service.add_session_to_memory(session)

    response = await service.search_memory(
        app_name="app", user_id="other_user", query="Paris"
    )
    assert len(response.memories) == 0


@pytest.mark.asyncio
async def test_search_memory_response_type():
    service = InMemoryMemoryService()
    response = await service.search_memory(
        app_name="app", user_id="u1", query="anything"
    )
    assert isinstance(response, SearchMemoryResponse)
    assert isinstance(response.memories, list)
