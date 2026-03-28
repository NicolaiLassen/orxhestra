"""Todo/planning tool - structured task tracking visible in the CLI.

Provides a `write_todos` tool that the agent calls to create and update
a task list. The CLI renders the todo list in the terminal.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


class TodoList:
    """Mutable todo list shared between the tool and the CLI renderer."""

    def __init__(self) -> None:
        self.todos: list[dict[str, str]] = []

    def update(self, todos: list[dict[str, str]]) -> None:
        """Replace the entire todo list."""
        self.todos = todos

    def render(self) -> str:
        """Render the todo list as a formatted string."""
        if not self.todos:
            return ""
        lines: list[str] = []
        for todo in self.todos:
            status: str = todo.get("status", "pending")
            content: str = todo.get("content", "")
            if status == "completed":
                icon = "[green]\u2713[/green]"
            elif status == "in_progress":
                icon = "[yellow]\u25b6[/yellow]"
            else:
                icon = "[dim]\u25cb[/dim]"
            lines.append(f"  {icon} {content}")
        return "\n".join(lines)


def make_todo_tool(todo_list: TodoList) -> BaseTool:
    """Create the write_todos tool backed by a shared TodoList instance."""

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

        todo_list.update(parsed)
        completed: int = sum(1 for t in parsed if t["status"] == "completed")
        total: int = len(parsed)
        return f"Updated todo list: {completed}/{total} completed"

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
