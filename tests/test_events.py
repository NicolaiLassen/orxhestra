"""Tests for unified event model."""

from langchain_adk.events.event import Event, EventType
from langchain_adk.events.event_actions import EventActions
from langchain_adk.models.part import Content, ToolCallPart, ToolResponsePart


def test_event_actions_defaults():
    actions = EventActions()
    assert actions.state_delta == {}
    assert actions.transfer_to_agent is None
    assert actions.escalate is None
    assert actions.skip_summarization is None


def test_event_actions_escalate():
    actions = EventActions(escalate=True)
    assert actions.escalate is True


def test_event_actions_state_delta():
    actions = EventActions(state_delta={"key": "value"})
    assert actions.state_delta["key"] == "value"


def test_event_base_fields():
    event = Event(type=EventType.AGENT_START, session_id="s1", agent_name="agent")
    assert event.type == EventType.AGENT_START
    assert event.session_id == "s1"
    assert event.agent_name == "agent"
    assert event.id is not None
    assert event.timestamp is not None


def test_agent_message_final_answer():
    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id="s1",
        agent_name="agent",
        content=Content.from_text("42"),
    )
    assert event.text == "42"
    assert event.is_final_response() is True


def test_agent_message_with_tool_calls():
    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id="s1",
        agent_name="agent",
        content=Content(parts=[
            ToolCallPart(tool_call_id="tc1", tool_name="search", args={"q": "test"})
        ]),
    )
    assert event.has_tool_calls is True
    assert event.tool_calls[0].tool_name == "search"
    assert event.is_final_response() is False


def test_tool_response_event():
    event = Event(
        type=EventType.TOOL_RESPONSE,
        session_id="s1",
        agent_name="agent",
        content=Content(parts=[
            ToolResponsePart(tool_call_id="tc1", tool_name="search", error="timeout")
        ]),
    )
    assert event.type == EventType.TOOL_RESPONSE
    assert event.content.tool_responses[0].error == "timeout"
    assert event.is_final_response() is False


def test_error_in_metadata():
    event = Event(
        type=EventType.AGENT_MESSAGE,
        session_id="s1",
        agent_name="agent",
        content=Content.from_text("something failed"),
        metadata={"error": True, "exception_type": "RuntimeError"},
    )
    assert event.metadata["error"] is True
    assert event.text == "something failed"


def test_event_unique_ids():
    e1 = Event(type=EventType.AGENT_START)
    e2 = Event(type=EventType.AGENT_START)
    assert e1.id != e2.id


def test_partial_is_not_final():
    event = Event(
        type=EventType.AGENT_MESSAGE,
        content=Content.from_text("partial text"),
        partial=True,
    )
    assert event.is_final_response() is False


def test_turn_complete_default():
    event = Event(type=EventType.AGENT_MESSAGE, content=Content.from_text("done"))
    assert event.turn_complete is True


def test_to_langchain_message_user():
    from langchain_core.messages import HumanMessage

    event = Event(type=EventType.USER_MESSAGE, content=Content.from_text("hello"))
    msg = event.to_langchain_message()
    assert isinstance(msg, HumanMessage)
    assert msg.content == "hello"


def test_to_langchain_message_agent():
    from langchain_core.messages import AIMessage

    event = Event(type=EventType.AGENT_MESSAGE, content=Content.from_text("answer"))
    msg = event.to_langchain_message()
    assert isinstance(msg, AIMessage)
    assert msg.content == "answer"


def test_to_langchain_message_tool_call():
    from langchain_core.messages import AIMessage

    event = Event(
        type=EventType.AGENT_MESSAGE,
        content=Content(parts=[
            ToolCallPart(tool_call_id="tc1", tool_name="search", args={"q": "test"})
        ]),
    )
    msg = event.to_langchain_message()
    assert isinstance(msg, AIMessage)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["name"] == "search"


def test_to_langchain_message_tool_response():
    from langchain_core.messages import ToolMessage

    event = Event(
        type=EventType.TOOL_RESPONSE,
        content=Content(parts=[
            ToolResponsePart(tool_call_id="tc1", tool_name="search", result="found it")
        ]),
    )
    msg = event.to_langchain_message()
    assert isinstance(msg, ToolMessage)
    assert msg.content == "found it"
    assert msg.tool_call_id == "tc1"


def test_from_langchain_message_roundtrip():
    from langchain_core.messages import AIMessage, HumanMessage

    # HumanMessage roundtrip
    event = Event.from_langchain_message(HumanMessage(content="hello"))
    assert event.type == EventType.USER_MESSAGE
    assert event.text == "hello"

    # AIMessage roundtrip
    event = Event.from_langchain_message(AIMessage(content="world"))
    assert event.type == EventType.AGENT_MESSAGE
    assert event.text == "world"
