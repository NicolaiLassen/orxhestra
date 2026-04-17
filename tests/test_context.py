"""Tests for Context."""

from __future__ import annotations

from orxhestra.agents.base_agent import BaseAgent  # noqa: F401
from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.artifacts.base_artifact_service import BaseArtifactService  # noqa: F401

Context.model_rebuild()


def test_context_defaults():
    ctx = Context(session_id="s1", agent_name="root")
    assert ctx.session_id == "s1"
    assert ctx.agent_name == "root"
    assert ctx.branch == ""
    assert ctx.state == {}
    assert ctx.user_id == ""
    assert ctx.app_name == ""


def test_context_derive_updates_agent_name():
    ctx = Context(session_id="s1", agent_name="root")
    child = ctx.derive(agent_name="child")
    assert child.agent_name == "child"
    assert child.session_id == "s1"
    assert child.invocation_id == ctx.invocation_id


def test_context_derive_builds_branch():
    ctx = Context(session_id="s1", agent_name="root")
    child = ctx.derive(agent_name="child")
    assert child.branch == "child"

    grandchild = child.derive(agent_name="grandchild")
    assert grandchild.branch == "child.grandchild"


def test_context_derive_custom_branch_suffix():
    ctx = Context(session_id="s1", agent_name="root")
    child = ctx.derive(agent_name="child", branch_suffix="parallel.child")
    assert child.branch == "parallel.child"


def test_context_state_shared_reference():
    ctx = Context(session_id="s1", agent_name="root")
    child = ctx.derive(agent_name="child")

    # Mutations on child's state are visible on parent (shared reference)
    child.state["key"] = "value"
    assert ctx.state["key"] == "value"


def test_context_invocation_id_auto_generated():
    ctx1 = Context(session_id="s1", agent_name="root")
    ctx2 = Context(session_id="s1", agent_name="root")
    assert ctx1.invocation_id != ctx2.invocation_id
