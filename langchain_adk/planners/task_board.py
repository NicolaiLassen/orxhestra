"""Task board - pure functions for managing structured agent tasks.

The task board is a plain dict stored in
``Context.state[StateKey.TASK_BOARD]``. Agents read and write
it via the manage_tasks tool exposed by ``TaskPlanner``.

Schema::

    {
        "version": int,
        "tasks_by_id": {
            "t1": {
                "id": str, "title": str, "description": str | None,
                "status": "pending" | "in_progress" | "completed" | "blocked",
                "required": bool,
                "created_at": str, "updated_at": str,
                "completed_at": str | None, "updated_by": str | None,
            }
        },
        "order": ["t1", "t2", ...],
        "summary": {"total": int, "completed": int, "open": int, "blocked": int}
    }
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from langchain_adk.planners.constants import TaskAction, TaskStatus, normalize_action


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()


def _next_task_id(tasks_by_id: dict[str, Any]) -> str:
    """Generate the next sequential task ID (t1, t2, ...)."""
    max_n = 0
    for tid in tasks_by_id:
        if tid.startswith("t") and tid[1:].isdigit():
            max_n = max(max_n, int(tid[1:]))
    return f"t{max_n + 1}"


def _normalize_status(status: Any) -> str:
    """Coerce a raw status value to a canonical TaskStatus string."""
    normalized = str(status or TaskStatus.PENDING).strip().lower()
    return normalized if normalized in TaskStatus._ALL else TaskStatus.PENDING


def _normalize_task(raw: dict[str, Any], tasks_by_id: dict[str, Any]) -> dict[str, Any]:
    """Normalise a raw task dict into the canonical task schema."""
    now = _now_iso()
    status = _normalize_status(raw.get("status"))
    updated_at = str(raw.get("updated_at") or raw.get("created_at") or now)
    created_at = str(raw.get("created_at") or updated_at)
    completed_at = raw.get("completed_at")
    if status == TaskStatus.COMPLETED and not completed_at:
        completed_at = updated_at
    return {
        "id": str(raw.get("id") or _next_task_id(tasks_by_id)),
        "title": str(raw.get("title") or "Untitled task"),
        "description": raw.get("description"),
        "status": status,
        "required": bool(raw.get("required", False)),
        "created_at": created_at,
        "updated_at": updated_at,
        "completed_at": completed_at if status == TaskStatus.COMPLETED else None,
        "updated_by": raw.get("updated_by"),
    }


def _recompute_summary(board: dict[str, Any]) -> None:
    """Recompute and write the summary block on the board in place."""
    tasks_by_id = board.get("tasks_by_id", {})
    total = len(tasks_by_id)
    completed = sum(1 for t in tasks_by_id.values() if t.get("status") == TaskStatus.COMPLETED)
    blocked = sum(1 for t in tasks_by_id.values() if t.get("status") == TaskStatus.BLOCKED)
    board["summary"] = {
        "total": total,
        "completed": completed,
        "open": total - completed,
        "blocked": blocked,
    }


def normalize_task_board(raw: Any) -> dict[str, Any]:
    """Normalize any input into a canonical task board dict.

    Parameters
    ----------
    raw : Any
        Raw input - may be a dict, a Pydantic model, or None.

    Returns
    -------
    dict[str, Any]
        A canonical task board with version, tasks_by_id, order, summary.
    """
    if not isinstance(raw, dict):
        raw = {}

    tasks_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    raw_tasks_by_id = raw.get("tasks_by_id")
    if isinstance(raw_tasks_by_id, dict):
        for task_id, task in raw_tasks_by_id.items():
            task_dict = task.model_dump() if hasattr(task, "model_dump") else task
            if isinstance(task_dict, dict):
                normalized = _normalize_task({**task_dict, "id": task_id}, tasks_by_id)
                tasks_by_id[normalized["id"]] = normalized

    raw_tasks_list = raw.get("tasks")
    if isinstance(raw_tasks_list, list):
        for task in raw_tasks_list:
            task_dict = task.model_dump() if hasattr(task, "model_dump") else task
            if isinstance(task_dict, dict):
                normalized = _normalize_task(task_dict, tasks_by_id)
                tasks_by_id[normalized["id"]] = normalized

    raw_order = raw.get("order") or []
    if isinstance(raw_order, list):
        for task_id in raw_order:
            tid = str(task_id)
            if tid in tasks_by_id and tid not in order:
                order.append(tid)
    for task_id in tasks_by_id:
        if task_id not in order:
            order.append(task_id)

    board: dict[str, Any] = {
        "version": int(raw.get("version") or 0),
        "tasks_by_id": tasks_by_id,
        "order": order,
    }
    _recompute_summary(board)
    return board


def initialize_task_board(
    tasks: list[Any],
    existing: Any = None,
) -> dict[str, Any]:
    """Build a fresh task board from a list of task dicts.

    Parameters
    ----------
    tasks : list[Any]
        Initial tasks to populate the board with.
    existing : Any, optional
        Existing board data to merge into.

    Returns
    -------
    dict[str, Any]
        A fully initialised canonical task board.
    """
    board = normalize_task_board(existing)
    if board["version"] == 0:
        board["version"] = 1
    for task in tasks:
        task_dict = task.model_dump() if hasattr(task, "model_dump") else task
        if isinstance(task_dict, dict):
            normalized = _normalize_task(task_dict, board["tasks_by_id"])
            if normalized["id"] not in board["tasks_by_id"]:
                board["tasks_by_id"][normalized["id"]] = normalized
                if normalized["id"] not in board["order"]:
                    board["order"].append(normalized["id"])
    _recompute_summary(board)
    return board


def list_task_items(board_like: Any) -> list[dict[str, Any]]:
    """Return ordered task list.

    Parameters
    ----------
    board_like : Any
        A raw or canonical task board dict.

    Returns
    -------
    list[dict[str, Any]]
        Tasks in their defined display order.
    """
    board = normalize_task_board(board_like)
    return [
        deepcopy(board["tasks_by_id"][tid])
        for tid in board["order"]
        if tid in board["tasks_by_id"]
    ]


def has_unresolved_tasks(board_like: Any) -> bool:
    """Return True if any task is not completed.

    Parameters
    ----------
    board_like : Any
        A raw or canonical task board dict.

    Returns
    -------
    bool
        True if at least one task has a non-completed status.
    """
    board = normalize_task_board(board_like)
    return any(
        t.get("status") != TaskStatus.COMPLETED
        for t in board["tasks_by_id"].values()
    )


def _resolve_task(
    board: dict[str, Any], task_id: str | None, title: str | None
) -> dict[str, Any] | None:
    """Look up a task by ID or title within a board."""
    if task_id and task_id in board["tasks_by_id"]:
        return board["tasks_by_id"][task_id]
    if title:
        lookup = title.strip().lower()
        for task in board["tasks_by_id"].values():
            if str(task.get("title", "")).strip().lower() == lookup:
                return task
    return None


def apply_task_action(
    board_like: Any,
    *,
    action: str,
    actor: str,
    task_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Apply a task action and return (updated_board, affected_task, message).

    Parameters
    ----------
    board_like : Any
        The current task board (raw or canonical).
    action : str
        The action to perform (create, update, complete, remove, list).
    actor : str
        The name of the agent performing the action.
    task_id : str, optional
        Task ID for targeted operations (update, complete, remove).
    title : str, optional
        Task title for create or name-based lookup.
    description : str, optional
        Task description for create or update.
    status : str, optional
        New status for update operations.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any] or None, str]
        A 3-tuple of (updated_board, affected_task_or_None, message).
    """
    board = normalize_task_board(board_like)
    resolved = normalize_action(action)

    if resolved is None:
        msg = f"Unknown action '{action}'. Use: list, create, update, complete, remove."
        return board, None, msg

    if resolved == TaskAction.LIST:
        return board, None, "Showing current tasks."

    now = _now_iso()
    board["version"] = board.get("version", 0) + 1

    if resolved == TaskAction.COMPLETE:
        task = _resolve_task(board, task_id, title)
        if task is None:
            return board, None, "Task not found."
        if task.get("status") == TaskStatus.COMPLETED:
            return board, deepcopy(task), f"Task '{task['title']}' is already completed."
        task.update(status=TaskStatus.COMPLETED, updated_at=now, completed_at=now, updated_by=actor)
        _recompute_summary(board)
        return board, deepcopy(task), f"Task '{task['title']}' marked as completed."

    if resolved == TaskAction.CREATE:
        created = _normalize_task(
            {"title": title or "New task", "description": description,
             "status": status or TaskStatus.PENDING, "updated_by": actor,
             "created_at": now, "updated_at": now},
            board["tasks_by_id"],
        )
        board["tasks_by_id"][created["id"]] = created
        board["order"].append(created["id"])
        _recompute_summary(board)
        return board, deepcopy(created), f"Task '{created['title']}' created."

    if resolved == TaskAction.UPDATE:
        task = _resolve_task(board, task_id, title)
        if task is None:
            return board, None, "Task not found."
        if title:
            task["title"] = title
        if description is not None:
            task["description"] = description
        if status:
            ns = _normalize_status(status)
            task["status"] = ns
            task["completed_at"] = now if ns == TaskStatus.COMPLETED else None
        task.update(updated_at=now, updated_by=actor)
        _recompute_summary(board)
        return board, deepcopy(task), f"Task '{task['title']}' updated."

    if resolved == TaskAction.REMOVE:
        task = _resolve_task(board, task_id, title)
        if task is None:
            return board, None, "Task not found."
        if task.get("required"):
            return board, None, f"Task '{task['title']}' is required and cannot be removed."
        del board["tasks_by_id"][task["id"]]
        board["order"] = [t for t in board["order"] if t != task["id"]]
        _recompute_summary(board)
        return board, deepcopy(task), f"Task '{task['title']}' removed."

    return board, None, f"Unknown action '{action}'."
