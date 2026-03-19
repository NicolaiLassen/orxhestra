"""Tests for tools: function_tool, ToolRegistry, ToolContext, exit_loop."""

import pytest

from langchain_adk.agents.context import Context
from langchain_adk.tools.exit_loop import EXIT_LOOP_SENTINEL, exit_loop_tool
from langchain_adk.tools.function_tool import function_tool
from langchain_adk.tools.tool_context import ToolContext
from langchain_adk.tools.tool_registry import ToolRegistry


def _make_ctx():
    return Context(session_id="s1", agent_name="agent")


def test_function_tool_name_from_function():
    async def my_tool(x: str) -> str:
        """A test tool."""
        return x

    tool = function_tool(my_tool)
    assert tool.name == "my_tool"


def test_function_tool_description_from_docstring():
    async def my_tool(x: str) -> str:
        """Does something useful."""
        return x

    tool = function_tool(my_tool)
    assert "Does something useful" in tool.description


def test_function_tool_custom_name():
    async def my_tool(x: str) -> str:
        return x

    tool = function_tool(my_tool, name="custom_name")
    assert tool.name == "custom_name"


@pytest.mark.asyncio
async def test_function_tool_invocation():
    async def double(x: str) -> str:
        """Doubles a string."""
        return x + x

    tool = function_tool(double)
    result = await tool.ainvoke({"x": "ab"})
    assert result == "abab"


def test_tool_registry_register_and_get():
    registry = ToolRegistry()

    async def my_tool(x: str) -> str:
        """A tool."""
        return x

    tool = function_tool(my_tool)
    registry.register(tool)
    assert registry.get("my_tool") is tool


def test_tool_registry_duplicate_raises():
    registry = ToolRegistry()

    async def my_tool(x: str) -> str:
        """A tool."""
        return x

    tool = function_tool(my_tool)
    registry.register(tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_tool_registry_list():
    registry = ToolRegistry()

    async def tool_a(x: str) -> str:
        """Tool A."""
        return x

    async def tool_b(x: str) -> str:
        """Tool B."""
        return x

    registry.register(function_tool(tool_a))
    registry.register(function_tool(tool_b))
    assert len(registry.list_tools()) == 2


def test_tool_registry_deregister():
    registry = ToolRegistry()

    async def my_tool(x: str) -> str:
        """A tool."""
        return x

    tool = function_tool(my_tool)
    registry.register(tool)
    registry.deregister("my_tool")
    assert registry.get("my_tool") is None


def test_tool_context_exposes_state():
    ctx = _make_ctx()
    ctx.state["x"] = 1
    tool_ctx = ToolContext(ctx)
    assert tool_ctx.state["x"] == 1
    assert tool_ctx.agent_name == "agent"
    assert tool_ctx.session_id == "s1"


def test_tool_context_actions_default():
    ctx = _make_ctx()
    tool_ctx = ToolContext(ctx)
    assert tool_ctx.actions.escalate is None
    assert tool_ctx.actions.transfer_to_agent is None


def test_tool_context_state_is_shared():
    ctx = _make_ctx()
    tool_ctx = ToolContext(ctx)
    tool_ctx.state["new_key"] = "new_val"
    assert ctx.state["new_key"] == "new_val"


@pytest.mark.asyncio
async def test_exit_loop_tool_returns_sentinel():
    result = await exit_loop_tool.ainvoke({})
    assert result == EXIT_LOOP_SENTINEL
