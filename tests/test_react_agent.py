"""Tests for ReActAgent: reasoning loop, tool execution, max iterations."""

import pytest
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from langchain_adk.agents.react_agent import ReActAgent, ReActStep
from langchain_adk.agents.context import Context
from langchain_adk.events.event import Event, EventType


class FakeStructuredModel(BaseChatModel):
    """Fake model that returns ReActStep objects from with_structured_output."""

    steps: list[ReActStep]
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake_structured"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        step = self.steps[min(self.call_count, len(self.steps) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(step.model_dump())))])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    def with_structured_output(self, schema=None, include_raw=False, method=None, **kwargs):
        """Return a wrapper that returns ReActStep objects."""
        parent = self

        class StructuredWrapper(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake_structured_wrapper"

            def _generate(self, messages, stop=None, **kw):
                step = parent.steps[min(parent.call_count, len(parent.steps) - 1)]
                parent.call_count += 1
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

            async def _agenerate(self, messages, stop=None, **kw):
                return self._generate(messages, stop, **kw)

            async def ainvoke(self, messages, **kw):
                step = parent.steps[min(parent.call_count, len(parent.steps) - 1)]
                parent.call_count += 1
                return step

        return StructuredWrapper()


def _ctx(**kwargs) -> Context:
    defaults = {"session_id": "test", "agent_name": "react"}
    defaults.update(kwargs)
    return Context(**defaults)


@pytest.mark.asyncio
async def test_react_direct_answer():
    step = ReActStep(
        scratchpad="",
        thought="I know the answer",
        answer="42",
    )
    llm = FakeStructuredModel(steps=[step])
    agent = ReActAgent(name="react", llm=llm)

    events = [e async for e in agent.astream("what is 6*7?", ctx=_ctx())]

    thoughts = [
        e for e in events
        if e.type == EventType.AGENT_MESSAGE
        and e.metadata.get("react_step") == "thought"
        and not e.partial
    ]
    finals = [e for e in events if e.is_final_response()]

    assert len(thoughts) == 1
    assert "know the answer" in thoughts[0].text
    assert len(finals) == 1
    assert finals[0].text == "42"


@pytest.mark.asyncio
async def test_react_tool_call_then_answer():
    @tool
    def lookup(key: str) -> str:
        """Look up a value."""
        return f"value_for_{key}"

    step1 = ReActStep(
        scratchpad="need to look up data",
        thought="I should use the lookup tool",
        action="lookup",
        action_input="test_key",
    )
    step2 = ReActStep(
        scratchpad="got value_for_test_key",
        thought="Now I have the answer",
        answer="The value is value_for_test_key",
    )
    llm = FakeStructuredModel(steps=[step1, step2])
    agent = ReActAgent(name="react", llm=llm, tools=[lookup])

    events = [e async for e in agent.astream("look up test_key", ctx=_ctx())]

    tool_calls = [e for e in events if e.has_tool_calls]
    tool_results = [e for e in events if e.type == EventType.TOOL_RESPONSE]
    observations = [
        e for e in events
        if e.metadata.get("react_step") == "observation"
    ]
    finals = [e for e in events if e.is_final_response()]

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_calls[0].tool_name == "lookup"
    assert len(tool_results) == 1
    assert "value_for_test_key" in tool_results[0].content.tool_responses[0].result
    assert len(finals) == 1


@pytest.mark.asyncio
async def test_react_tool_not_found():
    step1 = ReActStep(
        scratchpad="",
        thought="Try a tool",
        action="nonexistent",
        action_input="test",
    )
    step2 = ReActStep(
        scratchpad="tool not found",
        thought="No such tool",
        answer="Cannot find tool",
    )
    llm = FakeStructuredModel(steps=[step1, step2])
    agent = ReActAgent(name="react", llm=llm)

    events = [e async for e in agent.astream("test", ctx=_ctx())]

    observations = [
        e for e in events
        if e.metadata.get("react_step") == "observation"
    ]
    assert len(observations) == 1
    assert "not found" in observations[0].text


@pytest.mark.asyncio
async def test_react_max_iterations():
    step = ReActStep(
        scratchpad="",
        thought="Keep going",
        action="nonexistent",
        action_input="test",
    )
    llm = FakeStructuredModel(steps=[step])
    agent = ReActAgent(name="react", llm=llm, max_iterations=2)

    events = [e async for e in agent.astream("loop forever", ctx=_ctx())]

    errors = [e for e in events if e.metadata.get("error")]
    assert len(errors) == 1
    assert "Max iterations" in errors[0].text
