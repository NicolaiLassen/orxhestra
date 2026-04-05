"""Tests for unified TodoList + TaskPlanner system."""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.agents.llm_agent import LlmAgent
from orxhestra.planners.task_planner import TaskPlanner
from orxhestra.tools.todo_tool import TodoList, make_todo_tool

# ── TodoList unit tests ──────────────────────────────────────


def test_todolist_empty():
    """Empty todo list has no pending tasks."""
    tl = TodoList()
    assert not tl.has_pending()
    assert tl.build_status_text() is None
    assert tl.render() == ""


def test_todolist_has_pending():
    """has_pending() returns True when tasks are incomplete."""
    tl = TodoList()
    tl.update([
        {"content": "Task A", "status": "completed"},
        {"content": "Task B", "status": "in_progress"},
    ])
    assert tl.has_pending()


def test_todolist_all_completed():
    """has_pending() returns False when all tasks are completed."""
    tl = TodoList()
    tl.update([
        {"content": "Task A", "status": "completed"},
        {"content": "Task B", "status": "completed"},
    ])
    assert not tl.has_pending()


def test_todolist_build_status_text():
    """build_status_text() returns formatted task status."""
    tl = TodoList()
    tl.update([
        {"content": "Read files", "status": "completed"},
        {"content": "Fix bug", "status": "in_progress"},
        {"content": "Write tests", "status": "pending"},
    ])
    text = tl.build_status_text()
    assert text is not None
    assert "Read files" in text
    assert "Fix bug" in text
    assert "Write tests" in text
    assert "[done]" in text
    assert "[in_progress]" in text
    assert "1/3 completed" in text


def test_todolist_render():
    """render() returns Rich-formatted string."""
    tl = TodoList()
    tl.update([{"content": "Task", "status": "pending"}])
    rendered = tl.render()
    assert "Task" in rendered
    assert "\u25cb" in rendered  # ○ for pending


def test_todolist_update_replaces():
    """update() replaces the entire list."""
    tl = TodoList()
    tl.update([{"content": "A", "status": "pending"}])
    tl.update([{"content": "B", "status": "completed"}])
    assert len(tl.todos) == 1
    assert tl.todos[0]["content"] == "B"


# ── write_todos tool tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_write_todos_tool():
    """write_todos tool updates the shared TodoList."""
    tl = TodoList()
    tool = make_todo_tool(tl)

    result = await tool.ainvoke(json.dumps([
        {"content": "Step 1", "status": "completed"},
        {"content": "Step 2", "status": "pending"},
    ]))
    assert "1/2 completed" in result
    assert len(tl.todos) == 2
    assert tl.has_pending()


@pytest.mark.asyncio
async def test_write_todos_invalid_json():
    """write_todos rejects invalid JSON."""
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke("not json")
    assert "Error" in result


@pytest.mark.asyncio
async def test_write_todos_invalid_status():
    """write_todos rejects invalid status values."""
    tl = TodoList()
    tool = make_todo_tool(tl)
    result = await tool.ainvoke(json.dumps([
        {"content": "Task", "status": "invalid"},
    ]))
    assert "Error" in result


# ── TaskPlanner unit tests ───────────────────────────────────


def test_task_planner_no_todolist():
    """TaskPlanner without a TodoList is inert."""
    planner = TaskPlanner()
    assert planner.build_planning_instruction(None, None) is None
    assert not planner.has_pending_tasks(None)


def test_task_planner_reads_todolist():
    """TaskPlanner reads from the shared TodoList."""
    tl = TodoList()
    tl.update([
        {"content": "Research", "status": "completed"},
        {"content": "Write code", "status": "pending"},
    ])
    planner = TaskPlanner(todo_list=tl)

    instruction = planner.build_planning_instruction(None, None)
    assert instruction is not None
    assert "Research" in instruction
    assert "Write code" in instruction
    assert "1/2 completed" in instruction
    assert planner.has_pending_tasks(None)


def test_task_planner_no_pending():
    """TaskPlanner reports no pending when all done."""
    tl = TodoList()
    tl.update([{"content": "Done", "status": "completed"}])
    planner = TaskPlanner(todo_list=tl)

    assert not planner.has_pending_tasks(None)


def test_task_planner_seeds_initial_tasks():
    """TaskPlanner seeds initial tasks on first access."""
    tl = TodoList()
    planner = TaskPlanner(
        todo_list=tl,
        tasks=[
            {"title": "Step 1"},
            {"title": "Step 2", "status": "in_progress"},
        ],
    )

    instruction = planner.build_planning_instruction(None, None)
    assert instruction is not None
    assert "Step 1" in instruction
    assert "Step 2" in instruction
    assert len(tl.todos) == 2
    assert tl.todos[0]["status"] == "pending"
    assert tl.todos[1]["status"] == "in_progress"


def test_task_planner_does_not_reseed():
    """TaskPlanner doesn't overwrite existing todos with seeds."""
    tl = TodoList()
    tl.update([{"content": "Already here", "status": "pending"}])
    planner = TaskPlanner(
        todo_list=tl,
        tasks=[{"title": "Seed task"}],
    )

    planner.build_planning_instruction(None, None)
    assert len(tl.todos) == 1
    assert tl.todos[0]["content"] == "Already here"


def test_task_planner_set_todo_list():
    """set_todo_list() binds a TodoList after construction."""
    tl = TodoList()
    tl.update([{"content": "Task", "status": "pending"}])
    planner = TaskPlanner()

    assert not planner.has_pending_tasks(None)
    planner.set_todo_list(tl)
    assert planner.has_pending_tasks(None)


# ── Integration: TaskPlanner + LlmAgent ──────────────────────


class FakeChatModel(BaseChatModel):
    """Minimal fake chat model."""

    responses: list[AIMessage]
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs):
        msg = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(self, messages: list[BaseMessage], stop=None, **kwargs):
        return self._generate(messages, stop, **kwargs)


@pytest.mark.asyncio
async def test_planner_continues_until_todos_done():
    """Agent loops when TaskPlanner sees pending todos."""
    tl = TodoList()
    planner = TaskPlanner(todo_list=tl)

    # Turn 1: agent creates todos (text only, no tool calls)
    # Turn 2: agent says "all done" (planner sees pending → continues)
    # Turn 3: final answer (planner sees all completed → returns)
    tl.update([
        {"content": "Step 1", "status": "pending"},
        {"content": "Step 2", "status": "pending"},
    ])

    call_count = 0

    class ProgressLlm(FakeChatModel):
        def _generate(self, messages, stop=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: text only, planner sees pending → continue
                return ChatResult(
                    generations=[ChatGeneration(
                        message=AIMessage(content="Working on step 1...")
                    )]
                )
            # Second call: mark all done, return final answer
            tl.update([
                {"content": "Step 1", "status": "completed"},
                {"content": "Step 2", "status": "completed"},
            ])
            return ChatResult(
                generations=[ChatGeneration(
                    message=AIMessage(content="All done!")
                )]
            )

    llm = ProgressLlm(responses=[AIMessage(content="x")])
    agent = LlmAgent(
        name="agent",
        llm=llm,
        planner=planner,
        tools=[make_todo_tool(tl)],
    )

    ctx = Context(session_id="test", agent_name="agent")
    events = [e async for e in agent.astream("do tasks", ctx=ctx)]

    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "All done!"
    assert call_count == 2
