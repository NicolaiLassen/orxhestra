"""Tests for ReadonlyContext and CallbackContext."""

from types import MappingProxyType

import pytest

from langchain_adk.agents.context import Context
from langchain_adk.agents.readonly_context import CallbackContext, ReadonlyContext


def test_readonly_context_exposes_properties():
    ctx = Context(
        session_id="s1",
        agent_name="agent",
        user_id="user-1",
        app_name="app",
    )
    ro = ReadonlyContext(ctx)

    assert ro.session_id == "s1"
    assert ro.agent_name == "agent"
    assert ro.user_id == "user-1"
    assert ro.app_name == "app"
    assert ro.invocation_id == ctx.invocation_id


def test_readonly_context_state_is_immutable():
    ctx = Context(session_id="s1", agent_name="agent", state={"key": "val"})
    ro = ReadonlyContext(ctx)

    assert isinstance(ro.state, MappingProxyType)
    assert ro.state["key"] == "val"

    with pytest.raises(TypeError):
        ro.state["new_key"] = "nope"


def test_readonly_context_branch():
    ctx = Context(session_id="s1", agent_name="root")
    child = ctx.derive(agent_name="child")
    ro = ReadonlyContext(child)

    assert ro.branch == "child"


def test_callback_context_state_is_mutable():
    ctx = Context(session_id="s1", agent_name="agent", state={"key": "val"})
    cb = CallbackContext(ctx)

    # State is mutable dict, not MappingProxyType
    assert isinstance(cb.state, dict)
    cb.state["new_key"] = "new_val"
    assert ctx.state["new_key"] == "new_val"  # changes visible on parent


def test_callback_context_has_actions():
    ctx = Context(session_id="s1", agent_name="agent")
    cb = CallbackContext(ctx)

    assert cb.actions.escalate is None
    assert cb.actions.transfer_to_agent is None
    assert cb.actions.state_delta == {}

    cb.actions.escalate = True
    assert cb.actions.escalate is True


def test_callback_context_inherits_readonly_properties():
    ctx = Context(
        session_id="s1",
        agent_name="agent",
        user_id="u1",
        app_name="myapp",
    )
    cb = CallbackContext(ctx)

    assert cb.session_id == "s1"
    assert cb.agent_name == "agent"
    assert cb.user_id == "u1"
    assert cb.app_name == "myapp"
