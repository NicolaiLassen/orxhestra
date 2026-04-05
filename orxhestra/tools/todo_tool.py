"""Todo/planning tool — structured task tracking with metadata.

Provides a ``write_todos`` tool that the agent calls to create and
update a task list. Each task carries metadata: ID, description,
required flag, blocked status, version tracking, and who last edited.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

_VALID_STATUSES: frozenset[str] = frozenset({
    "pending", "in_progress", "completed", "blocked",
})


class TodoList:
    """Mutable todo list shared between the tool, CLI, and planner.

    Attributes
    ----------
    todos : list[dict[str, Any]]
        Task items. Each has at minimum ``content`` and ``status``.
        Optional: ``id``, ``description``, ``required``, ``updated_by``.
    version : int
        Incremented on each update.
    """

    def __init__(self) -> None:
        self.todos: list[dict[str, Any]] = []
        self.version: int = 0

    def update(
        self,
        todos: list[dict[str, Any]],
        *,
        actor: str = "",
    ) -> None:
        """Replace the entire todo list and bump version.

        Parameters
        ----------
        todos : list[dict[str, Any]]
            New task items.
        actor : str
            Name of the agent performing the update.
        """
        now: str = time.strftime("%H:%M:%S")
        for i, t in enumerate(todos):
            t.setdefault("id", f"t{i + 1}")
            t.setdefault("required", False)
            if actor:
                t["updated_by"] = actor
            t["updated_at"] = now
        self.todos = todos
        self.version += 1

    def has_pending(self) -> bool:
        """Return True if any todo is not yet completed."""
        return any(t.get("status") != "completed" for t in self.todos)

    def get_active_task(self) -> str | None:
        """Return the content of the first in_progress task, or None."""
        for t in self.todos:
            if t.get("status") == "in_progress":
                return t.get("content")
        return None

    def build_status_text(self) -> str | None:
        """Build a task status block for injection into the system prompt.

        Returns
        -------
        str or None
            Formatted task status, or None if no todos exist.
        """
        if not self.todos:
            return None
        lines: list[str] = [f"Current tasks (v{self.version}):"]
        for t in self.todos:
            tid: str = t.get("id", "?")
            status: str = t.get("status", "pending")
            content: str = t.get("content", "")
            tag: str = "[done]" if status == "completed" else f"[{status}]"
            req: str = " (required)" if t.get("required") else ""
            desc: str = f" — {t['description']}" if t.get("description") else ""
            by: str = f" @{t['updated_by']}" if t.get("updated_by") else ""
            lines.append(f"  {tid} {tag} {content}{req}{desc}{by}")
        completed: int = sum(
            1 for t in self.todos if t.get("status") == "completed"
        )
        blocked: int = sum(
            1 for t in self.todos if t.get("status") == "blocked"
        )
        summary: str = f"Progress: {completed}/{len(self.todos)} completed"
        if blocked:
            summary += f", {blocked} blocked"
        lines.append(f"\n{summary}.")
        return "\n".join(lines)

    def render(self) -> str:
        """Render the todo list as a Rich-formatted string."""
        if not self.todos:
            return ""
        lines: list[str] = []
        for todo in self.todos:
            status: str = todo.get("status", "pending")
            content: str = todo.get("content", "")
            if status == "completed":
                icon = "[orx.success]\u2713[/orx.success]"
            elif status == "in_progress":
                icon = "[orx.warning]\u25b6[/orx.warning]"
            elif status == "blocked":
                icon = "[orx.error]\u2717[/orx.error]"
            else:
                icon = "[orx.muted]\u25cb[/orx.muted]"
            req: str = " [orx.muted](required)[/orx.muted]" if todo.get("required") else ""
            lines.append(f"  {icon} {content}{req}")
        return "\n".join(lines)


def make_todo_tool(
    todo_list: TodoList,
    agent_name: str = "",
) -> BaseTool:
    """Create the write_todos tool backed by a shared TodoList.

    Parameters
    ----------
    todo_list : TodoList
        Shared mutable todo list instance.
    agent_name : str
        Agent name recorded as ``updated_by`` on each task.

    Returns
    -------
    BaseTool
        A ``write_todos`` structured tool.
    """

    async def write_todos(todos: str) -> str:
        """Update the task list for the current work.

        Args:
            todos: JSON array of task objects. Required fields:
                - content: what needs to be done (imperative form)
                - status: "pending", "in_progress", "completed", or "blocked"
                Optional fields:
                - description: longer explanation of the task
                - required: true if the task must be completed (default false)
        """
        try:
            parsed: list[dict[str, Any]] = json.loads(todos)
        except json.JSONDecodeError:
            return "Error: todos must be a valid JSON array"

        for item in parsed:
            if "content" not in item or "status" not in item:
                return "Error: each todo must have 'content' and 'status'"
            if item["status"] not in _VALID_STATUSES:
                return (
                    f"Error: status must be one of {sorted(_VALID_STATUSES)}"
                )

        old_todos: list[dict[str, Any]] = list(todo_list.todos)
        todo_list.update(parsed, actor=agent_name)

        completed: int = sum(
            1 for t in parsed if t["status"] == "completed"
        )
        total: int = len(parsed)

        lines: list[str] = [
            f"Updated todo list v{todo_list.version}: "
            f"{completed}/{total} completed"
        ]

        old_statuses: dict[str, str] = {
            t.get("content", ""): t.get("status", "") for t in old_todos
        }
        old_contents: set[str] = set(old_statuses)

        for t in parsed:
            content: str = t["content"]
            status: str = t["status"]
            old_status: str | None = old_statuses.get(content)
            if content not in old_contents:
                lines.append(f"  + {content} [{status}]")
            elif old_status and old_status != status:
                lines.append(f"  ~ {content} [{old_status} -> {status}]")

        removed: list[str] = [
            c for c in old_contents
            if c not in {t["content"] for t in parsed}
        ]
        for c in removed:
            lines.append(f"  - {c}")

        # Verification nudge: when all tasks are completed, remind the
        # agent to verify its work before responding to the user.
        if total > 0 and completed == total:
            has_verify_step: bool = any(
                "verify" in t["content"].lower()
                or "test" in t["content"].lower()
                or "check" in t["content"].lower()
                for t in parsed
            )
            if not has_verify_step:
                lines.append(
                    "\nAll tasks completed. Before responding, verify "
                    "your work — run tests, check files, or review "
                    "changes to confirm everything is correct."
                )

        return "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=write_todos,
        name="write_todos",
        description=(
            "Create or update a structured task list. "
            "Pass a JSON array with 'content' (what to do), "
            "'status' (pending/in_progress/completed/blocked), and "
            "optionally 'description' and 'required' (bool). "
            "Sends the full list each time (replaces previous).\n\n"
            "WHEN TO USE: complex multi-step tasks (3+ steps), "
            "user provides multiple tasks, non-trivial work.\n"
            "WHEN TO SKIP: single trivial task, purely conversational, "
            "can be done in fewer than 3 steps.\n"
            "RULES: mark tasks in_progress BEFORE starting work. "
            "Only one task should be in_progress at a time. "
            "Mark completed IMMEDIATELY after finishing."
        ),
    )
