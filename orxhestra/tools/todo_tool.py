"""Todo/planning tool - structured task tracking visible in the CLI.

Provides a `write_todos` tool that the agent calls to create and update
a task list. The CLI renders the todo list in the terminal.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool, StructuredTool


class TodoList:
    """Mutable todo list shared between the tool and the CLI renderer.

    Attributes
    ----------
    todos : list[dict[str, str]]
        List of todo items, each with ``"content"`` and ``"status"`` keys.
    """

    def __init__(self) -> None:
        self.todos: list[dict[str, str]] = []

    def update(self, todos: list[dict[str, str]]) -> None:
        """Replace the entire todo list.

        Parameters
        ----------
        todos : list[dict[str, str]]
            New list of todo items.
        """
        self.todos = todos

    def has_pending(self) -> bool:
        """Return True if any todo is not yet completed."""
        return any(t.get("status") != "completed" for t in self.todos)

    def build_status_text(self) -> str | None:
        """Build a task status block for injection into the system prompt.

        Returns
        -------
        str or None
            Formatted task status, or None if no todos exist.
        """
        if not self.todos:
            return None
        lines: list[str] = ["Current tasks:"]
        for i, t in enumerate(self.todos, 1):
            status: str = t.get("status", "pending")
            tag: str = "[done]" if status == "completed" else f"[{status}]"
            content: str = t.get("content", "")
            lines.append(f"  t{i} {tag} {content}")
        completed: int = sum(
            1 for t in self.todos if t.get("status") == "completed"
        )
        lines.append(f"\nProgress: {completed}/{len(self.todos)} completed.")
        return "\n".join(lines)

    def render(self) -> str:
        """Render the todo list as a formatted string.

        Returns
        -------
        str
            Rich-markup formatted string, or empty string if no todos.
        """
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
            else:
                icon = "[orx.muted]\u25cb[/orx.muted]"
            lines.append(f"  {icon} {content}")
        return "\n".join(lines)


def make_todo_tool(todo_list: TodoList) -> BaseTool:
    """Create the write_todos tool backed by a shared TodoList instance.

    Parameters
    ----------
    todo_list : TodoList
        Shared mutable todo list instance.

    Returns
    -------
    BaseTool
        A ``write_todos`` structured tool.
    """

    async def write_todos(todos: str) -> str:
        """Update the task list for the current work.

        Args:
            todos: JSON array of task objects, each with:
                - content: what needs to be done (imperative form)
                - status: "pending", "in_progress", or "completed"
        """
        try:
            parsed: list[dict[str, str]] = json.loads(todos)
        except json.JSONDecodeError:
            return "Error: todos must be a valid JSON array"

        for item in parsed:
            if "content" not in item or "status" not in item:
                return "Error: each todo must have 'content' and 'status' fields"
            if item["status"] not in ("pending", "in_progress", "completed"):
                return "Error: status must be 'pending', 'in_progress', or 'completed'"

        old_todos: list[dict[str, str]] = list(todo_list.todos)
        todo_list.update(parsed)

        completed: int = sum(1 for t in parsed if t["status"] == "completed")
        total: int = len(parsed)

        # Build a diff so the LLM sees what changed.
        lines: list[str] = [f"Updated todo list: {completed}/{total} completed"]

        old_contents: set[str] = {t.get("content", "") for t in old_todos}
        old_statuses: dict[str, str] = {
            t.get("content", ""): t.get("status", "") for t in old_todos
        }
        for t in parsed:
            content: str = t["content"]
            status: str = t["status"]
            old_status: str | None = old_statuses.get(content)
            if content not in old_contents:
                lines.append(f"  + {content} [{status}]")
            elif old_status and old_status != status:
                lines.append(f"  ~ {content} [{old_status} -> {status}]")

        removed: list[str] = [
            c for c in old_contents if c not in {t["content"] for t in parsed}
        ]
        for c in removed:
            lines.append(f"  - {c}")

        return "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=write_todos,
        name="write_todos",
        description=(
            "Create or update a structured task list to track your progress. "
            "Pass a JSON array of objects with 'content' (what to do) and "
            "'status' ('pending', 'in_progress', or 'completed'). "
            "Use this to break complex tasks into steps and show progress."
        ),
    )
