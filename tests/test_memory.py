"""Tests for Memory, MemoryStore, and InMemoryMemoryStore."""

import pytest

from langchain_adk.events.event import Event, EventType
from langchain_adk.memory.in_memory_store import InMemoryMemoryStore
from langchain_adk.memory.memory import Memory
from langchain_adk.memory.memory_store import SearchMemoryResponse
from langchain_adk.models.part import Content
from langchain_adk.sessions.session import Session


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
    store = InMemoryMemoryStore()
    session = _make_session_with_events()
    await store.add_session_to_memory(session)

    response = await store.search_memory(
        app_name="app", user_id="u1", query="Paris"
    )
    assert len(response.memories) >= 1
    assert any("Paris" in m.content for m in response.memories)


@pytest.mark.asyncio
async def test_search_memory_no_match():
    store = InMemoryMemoryStore()
    session = _make_session_with_events()
    await store.add_session_to_memory(session)

    response = await store.search_memory(
        app_name="app", user_id="u1", query="Berlin"
    )
    assert len(response.memories) == 0


@pytest.mark.asyncio
async def test_search_memory_wrong_user():
    store = InMemoryMemoryStore()
    session = _make_session_with_events()
    await store.add_session_to_memory(session)

    response = await store.search_memory(
        app_name="app", user_id="other_user", query="Paris"
    )
    assert len(response.memories) == 0


@pytest.mark.asyncio
async def test_search_memory_response_type():
    store = InMemoryMemoryStore()
    response = await store.search_memory(
        app_name="app", user_id="u1", query="anything"
    )
    assert isinstance(response, SearchMemoryResponse)
    assert isinstance(response.memories, list)
