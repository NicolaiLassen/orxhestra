"""Tests for LlmRequest and LlmResponse models."""

from langchain_core.messages import AIMessage

from langchain_adk.models.llm_request import LlmRequest
from langchain_adk.models.llm_response import LlmResponse


def test_llm_request_defaults():
    req = LlmRequest(system_instruction="You are helpful.", messages=[])
    assert req.system_instruction == "You are helpful."
    assert req.messages == []
    assert req.tools == []
    assert req.tools_dict == {}
    assert req.output_schema is None
    assert req.model is None


def test_llm_request_has_tools():
    req = LlmRequest(system_instruction="", messages=[], tools=[])
    assert req.has_tools() is False


def test_llm_response_from_text_message():
    msg = AIMessage(content="Hello world")
    resp = LlmResponse.from_ai_message(msg)
    assert resp.text == "Hello world"
    assert resp.has_tool_calls is False
    assert resp.tool_calls == []


def test_llm_response_from_tool_call_message():
    msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "tc1", "name": "search", "args": {"q": "test"}},
        ],
    )
    resp = LlmResponse.from_ai_message(msg)
    assert resp.has_tool_calls is True
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "search"


def test_llm_response_preserves_raw_message():
    msg = AIMessage(content="raw")
    resp = LlmResponse.from_ai_message(msg)
    assert resp.raw_message is msg


def test_llm_response_token_counts():
    msg = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    resp = LlmResponse.from_ai_message(msg)
    assert resp.input_tokens == 10
    assert resp.output_tokens == 5
