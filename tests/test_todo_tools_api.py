"""Tests for the public todo tool factories.

Covers ``make_todo_tool``, ``make_read_todos_tool``, ``make_todo_tools``,
and the ``TodoList`` container.
"""

from __future__ import annotations

import json

import pytest

from orxhestra.tools.todo_tool import (
    TodoList,
    make_read_todos_tool,
    make_todo_tool,
    make_todo_tools,
)

# ── TodoList ─────────────────────────────────────────────────────────


def test_todo_list_starts_empty():
    tl = TodoList()
    assert tl.todos == []
    assert tl.version == 0
    assert not tl.has_pending()
    assert tl.get_active_task() is None


def test_todo_list_update_assigns_ids_and_bumps_version():
    tl = TodoList()
    tl.update([
        {"content": "first", "status": "pending"},
        {"content": "second", "status": "in_progress"},
    ])
    assert tl.version == 1
    assert tl.todos[0]["id"] == "t1"
    assert tl.todos[1]["id"] == "t2"


def test_todo_list_actor_recorded():
    tl = TodoList()
    tl.update([{"content": "x", "status": "pending"}], actor="agent_alice")
    assert tl.todos[0]["updated_by"] == "agent_alice"


def test_todo_list_has_pending_true_when_not_all_complete():
    tl = TodoList()
    tl.update([
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "pending"},
    ])
    assert tl.has_pending()


def test_todo_list_has_pending_false_when_all_complete():
    tl = TodoList()
    tl.update([
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "completed"},
    ])
    assert not tl.has_pending()


def test_todo_list_active_task_returns_in_progress():
    tl = TodoList()
    tl.update([
        {"content": "a", "status": "pending"},
        {"content": "active thing", "status": "in_progress"},
        {"content": "c", "status": "pending"},
    ])
    assert tl.get_active_task() == "active thing"


def test_todo_list_build_status_text_includes_summary():
    tl = TodoList()
    tl.update([
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "pending"},
    ])
    text = tl.build_status_text()
    assert text is not None
    assert "1/2 completed" in text


def test_todo_list_render_empty_returns_empty_string():
    tl = TodoList()
    assert tl.render() == ""


# ── make_todo_tool ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_todos_tool_creates_tasks():
    tl = TodoList()
    tool = make_todo_tool(tl, agent_name="tester")
    result = await tool.ainvoke({
        "todos": json.dumps([
            {"content": "step1", "status": "pending"},
            {"content": "step2", "status": "in_progress"},
        ]),
    })
    assert "Updated todo list v1" in result
    assert tl.version == 1
    assert len(tl.todos) == 2
    assert tl.todos[0]["updated_by"] == "tester"


@pytest.mark.asyncio
async def test_write_todos_rejects_invalid_json():
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke({"todos": "not-json"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_todos_rejects_missing_fields():
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke({"todos": json.dumps([{"content": "only"}])})
    assert "content" in result and "status" in result


@pytest.mark.asyncio
async def test_write_todos_rejects_unknown_status():
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke({
        "todos": json.dumps([{"content": "x", "status": "weird"}]),
    })
    assert "status must be one of" in result


@pytest.mark.asyncio
async def test_write_todos_tracks_status_transitions():
    tl = TodoList()
    tool = make_todo_tool(tl)
    await tool.ainvoke({
        "todos": json.dumps([{"content": "work", "status": "pending"}]),
    })
    result = await tool.ainvoke({
        "todos": json.dumps([{"content": "work", "status": "completed"}]),
    })
    assert "pending -> completed" in result


@pytest.mark.asyncio
async def test_write_todos_detects_removed_tasks():
    tl = TodoList()
    tool = make_todo_tool(tl)
    await tool.ainvoke({
        "todos": json.dumps([
            {"content": "a", "status": "pending"},
            {"content": "b", "status": "pending"},
        ]),
    })
    result = await tool.ainvoke({
        "todos": json.dumps([{"content": "a", "status": "pending"}]),
    })
    assert "- b" in result


@pytest.mark.asyncio
async def test_write_todos_adds_verification_nudge_on_completion():
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke({
        "todos": json.dumps([{"content": "ship it", "status": "completed"}]),
    })
    assert "verify" in result.lower()


@pytest.mark.asyncio
async def test_write_todos_skips_nudge_if_verify_step_exists():
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke({
        "todos": json.dumps([
            {"content": "ship", "status": "completed"},
            {"content": "verify all", "status": "completed"},
        ]),
    })
    # Verification step present → no nudge.
    assert "Before responding, verify" not in result


# ── make_read_todos_tool ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_todos_returns_current_list_as_json():
    tl = TodoList()
    tl.update([{"content": "task", "status": "pending"}])
    tool = make_read_todos_tool(tl)
    result = await tool.ainvoke({})
    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["content"] == "task"
    assert parsed[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_read_todos_empty_returns_empty_list():
    tl = TodoList()
    tool = make_read_todos_tool(tl)
    result = await tool.ainvoke({})
    assert json.loads(result) == []


@pytest.mark.asyncio
async def test_read_todos_reflects_updates_from_write_tool():
    tl = TodoList()
    write = make_todo_tool(tl)
    read = make_read_todos_tool(tl)
    await write.ainvoke({
        "todos": json.dumps([{"content": "x", "status": "in_progress"}]),
    })
    result = await read.ainvoke({})
    parsed = json.loads(result)
    assert parsed[0]["status"] == "in_progress"


# ── make_todo_tools (convenience bundle) ─────────────────────────────


@pytest.mark.asyncio
async def test_make_todo_tools_returns_both_tools():
    tools = make_todo_tools()
    names = {t.name for t in tools}
    assert names == {"write_todos", "read_todos"}


@pytest.mark.asyncio
async def test_make_todo_tools_shares_state_between_tools():
    tools = make_todo_tools(agent_name="bundle_agent")
    write = next(t for t in tools if t.name == "write_todos")
    read = next(t for t in tools if t.name == "read_todos")
    await write.ainvoke({
        "todos": json.dumps([{"content": "shared", "status": "pending"}]),
    })
    result = await read.ainvoke({})
    parsed = json.loads(result)
    assert parsed[0]["content"] == "shared"
    assert parsed[0]["updated_by"] == "bundle_agent"


@pytest.mark.asyncio
async def test_make_todo_tools_accepts_preexisting_list():
    tl = TodoList()
    tl.update([{"content": "seeded", "status": "pending"}])
    tools = make_todo_tools(tl)
    read = next(t for t in tools if t.name == "read_todos")
    result = await read.ainvoke({})
    parsed = json.loads(result)
    assert parsed[0]["content"] == "seeded"


def test_make_todo_tools_without_arg_creates_new_list():
    tools_a = make_todo_tools()
    tools_b = make_todo_tools()
    # Each bundle owns its own TodoList — no leaking.
    assert tools_a is not tools_b


# ── Public export surface ────────────────────────────────────────────


def test_todo_tools_exported_from_tools_package():
    from orxhestra.tools import (
        TodoList as ExportedTodoList,
    )
    from orxhestra.tools import (
        make_read_todos_tool as exported_read,
    )
    from orxhestra.tools import (
        make_todo_tool as exported_write,
    )
    from orxhestra.tools import (
        make_todo_tools as exported_bundle,
    )

    assert ExportedTodoList is TodoList
    assert exported_write is make_todo_tool
    assert exported_read is make_read_todos_tool
    assert exported_bundle is make_todo_tools
