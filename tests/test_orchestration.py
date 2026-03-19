"""Tests for task board and ManageTasksTool."""

from langchain_adk.planners.constants import TaskStatus
from langchain_adk.planners.task_board import (
    apply_task_action,
    has_unresolved_tasks,
    initialize_task_board,
    list_task_items,
    normalize_task_board,
)

SAMPLE_TASKS = [
    {"title": "Research", "description": "Look things up"},
    {"title": "Write", "description": "Write the report"},
]


def test_initialize_task_board():
    board = initialize_task_board(SAMPLE_TASKS)
    assert board["version"] == 1
    assert len(board["tasks_by_id"]) == 2
    assert board["summary"]["total"] == 2
    assert board["summary"]["completed"] == 0


def test_list_task_items_ordered():
    board = initialize_task_board(SAMPLE_TASKS)
    items = list_task_items(board)
    assert len(items) == 2
    assert items[0]["title"] == "Research"
    assert items[1]["title"] == "Write"


def test_has_unresolved_tasks_true():
    board = initialize_task_board(SAMPLE_TASKS)
    assert has_unresolved_tasks(board) is True


def test_has_unresolved_tasks_false():
    board = initialize_task_board(SAMPLE_TASKS)
    for task_id in board["tasks_by_id"]:
        board["tasks_by_id"][task_id]["status"] = TaskStatus.COMPLETED
    assert has_unresolved_tasks(board) is False


def test_apply_task_action_complete():
    board = initialize_task_board(SAMPLE_TASKS)
    task_id = board["order"][0]
    updated, task, msg = apply_task_action(
        board, action="complete", actor="agent", task_id=task_id
    )
    assert task["status"] == TaskStatus.COMPLETED
    assert "completed" in msg.lower()
    assert updated["summary"]["completed"] == 1


def test_apply_task_action_create():
    board = initialize_task_board(SAMPLE_TASKS)
    updated, task, msg = apply_task_action(
        board, action="create", actor="agent", title="New task"
    )
    assert task["title"] == "New task"
    assert updated["summary"]["total"] == 3


def test_apply_task_action_update():
    board = initialize_task_board(SAMPLE_TASKS)
    task_id = board["order"][0]
    updated, task, msg = apply_task_action(
        board, action="update", actor="agent",
        task_id=task_id, status="in_progress"
    )
    assert task["status"] == TaskStatus.IN_PROGRESS


def test_apply_task_action_remove():
    board = initialize_task_board(SAMPLE_TASKS)
    task_id = board["order"][0]
    updated, task, msg = apply_task_action(
        board, action="remove", actor="agent", task_id=task_id
    )
    assert task_id not in updated["tasks_by_id"]
    assert updated["summary"]["total"] == 1


def test_apply_task_action_list():
    board = initialize_task_board(SAMPLE_TASKS)
    updated, task, msg = apply_task_action(
        board, action="list", actor="agent"
    )
    assert task is None
    assert "Showing" in msg


def test_apply_task_action_unknown():
    board = initialize_task_board(SAMPLE_TASKS)
    updated, task, msg = apply_task_action(
        board, action="invalid_action", actor="agent"
    )
    assert "Unknown" in msg


def test_normalize_task_board_empty():
    board = normalize_task_board({})
    assert board["tasks_by_id"] == {}
    assert board["order"] == []
    assert board["summary"]["total"] == 0


def test_version_increments_on_mutation():
    board = initialize_task_board(SAMPLE_TASKS)
    assert board["version"] == 1
    task_id = board["order"][0]
    updated, _, _ = apply_task_action(
        board, action="complete", actor="agent", task_id=task_id
    )
    assert updated["version"] == 2
