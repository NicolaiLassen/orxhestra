"""Tests for ReActAgent: reasoning loop, tool execution, max iterations, inheritance."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.agents.llm_agent import LlmAgent
from orxhestra.agents.react_agent import ReActAgent, ReActStep
from orxhestra.events.event import EventType


class FakeStructuredModel(BaseChatModel):
    """Fake model that returns ReActStep objects from with_structured_output."""

    steps: list[ReActStep]
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake_structured"

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs,
    ) -> ChatResult:
        step = self.steps[min(self.call_count, len(self.steps) - 1)]
        self.call_count += 1
        content = str(step.model_dump())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs,
    ) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    def with_structured_output(self, schema=None, include_raw=False, method=None, **kwargs):
        """Return a wrapper that returns ReActStep objects."""
        parent = self

        class StructuredWrapper(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "fake_structured_wrapper"

            def _generate(self, messages, stop=None, **kw):
                _step = parent.steps[min(parent.call_count, len(parent.steps) - 1)]  # noqa: F841
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
    model = FakeStructuredModel(steps=[step])
    agent = ReActAgent(name="react", model=model)

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
    model = FakeStructuredModel(steps=[step1, step2])
    agent = ReActAgent(name="react", model=model, tools=[lookup])

    events = [e async for e in agent.astream("look up test_key", ctx=_ctx())]

    tool_calls = [e for e in events if e.has_tool_calls]
    tool_results = [e for e in events if e.type == EventType.TOOL_RESPONSE]
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
    model = FakeStructuredModel(steps=[step1, step2])
    agent = ReActAgent(name="react", model=model)

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
    model = FakeStructuredModel(steps=[step])
    agent = ReActAgent(name="react", model=model, max_iterations=2)

    events = [e async for e in agent.astream("loop forever", ctx=_ctx())]

    errors = [e for e in events if e.metadata.get("error")]
    assert len(errors) == 1
    assert "Max iterations" in errors[0].text


# ---------------------------------------------------------------------------
# Inheritance tests — ReActAgent extends LlmAgent
# ---------------------------------------------------------------------------


def test_react_is_subclass_of_llm_agent():
    """ReActAgent should be a subclass of LlmAgent."""
    assert issubclass(ReActAgent, LlmAgent)


def test_react_inherits_llm_agent_attributes():
    """ReActAgent should accept all LlmAgent keyword arguments."""
    model = FakeStructuredModel(steps=[])
    agent = ReActAgent(
        name="react",
        model=model,
        instructions="Custom instructions here.",
        description="A react agent",
        max_iterations=5,
    )
    assert agent.name == "react"
    assert agent.description == "A react agent"
    assert agent.max_iterations == 5
    assert agent._instructions == "Custom instructions here."


def test_react_default_instructions_empty():
    """ReActAgent defaults to empty instructions (ReAct prompt is the base)."""
    model = FakeStructuredModel(steps=[])
    agent = ReActAgent(name="react", model=model)
    assert agent._instructions == ""


def test_react_accepts_planner():
    """ReActAgent should accept a planner kwarg via LlmAgent inheritance."""
    from unittest.mock import MagicMock

    model = FakeStructuredModel(steps=[])
    mock_planner = MagicMock()
    agent = ReActAgent(name="react", model=model, planner=mock_planner)
    assert agent._planner is mock_planner


def test_react_accepts_callbacks():
    """ReActAgent should accept callback kwargs via LlmAgent inheritance."""
    from unittest.mock import AsyncMock

    model = FakeStructuredModel(steps=[])
    before_cb = AsyncMock()
    after_cb = AsyncMock()
    agent = ReActAgent(
        name="react",
        model=model,
        before_model_callback=before_cb,
        after_model_callback=after_cb,
    )
    assert agent.before_model_callback is before_cb
    assert agent.after_model_callback is after_cb


@pytest.mark.asyncio
async def test_react_custom_instructions_in_system_prompt():
    """Custom instructions should appear in the ReAct system prompt."""
    model = FakeStructuredModel(steps=[])
    agent = ReActAgent(
        name="react",
        model=model,
        instructions="Always respond in French.",
    )
    ctx = _ctx()
    prompt = await agent._build_react_system_prompt(ctx)
    assert "ReAct pattern" in prompt  # base prompt
    assert "Always respond in French." in prompt  # custom instructions


@pytest.mark.asyncio
async def test_react_no_custom_instructions_clean_prompt():
    """Without custom instructions, the prompt should not have 'Additional instructions'."""
    model = FakeStructuredModel(steps=[])
    agent = ReActAgent(name="react", model=model)
    ctx = _ctx()
    prompt = await agent._build_react_system_prompt(ctx)
    assert "ReAct pattern" in prompt
    assert "Additional instructions" not in prompt


@pytest.mark.asyncio
async def test_react_with_instructions_and_tools():
    """Custom instructions + tools should both appear in the system prompt."""
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))

    model = FakeStructuredModel(steps=[
        ReActStep(scratchpad="", thought="I know", answer="done"),
    ])
    agent = ReActAgent(
        name="react",
        model=model,
        tools=[calculator],
        instructions="Show your work step by step.",
    )
    ctx = _ctx()
    prompt = await agent._build_react_system_prompt(ctx)
    assert "calculator" in prompt
    assert "Show your work step by step." in prompt
