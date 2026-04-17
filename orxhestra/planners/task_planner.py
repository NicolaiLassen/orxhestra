"""TaskPlanner — a planner that reads from a shared TodoList.

Injects the current task status into the system prompt before each
LLM call so the model always knows which tasks are pending. Works
with the standard ``write_todos`` tool — no separate ``manage_tasks``
tool needed.

Usage::

    from orxhestra.tools.todo_tool import TodoList, make_todo_tool
    from orxhestra.planners import TaskPlanner

    todo_list = TodoList()
    planner = TaskPlanner(todo_list=todo_list)

    agent = LlmAgent(
        name="coder",
        model=model,
        tools=[make_todo_tool(todo_list)],
        planner=planner,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orxhestra.planners.base_planner import BasePlanner

if TYPE_CHECKING:
    from orxhestra.agents.readonly_context import ReadonlyContext
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.tools.todo_tool import TodoList


class TaskPlanner(BasePlanner):
    """Planner that injects TodoList status into the system prompt.

    Reads from a shared ``TodoList`` instance (the same one backing
    the ``write_todos`` tool) to build per-turn planning instructions.

    Parameters
    ----------
    todo_list : TodoList, optional
        Shared todo list. If None, the planner is inert (returns no
        instructions and reports no pending tasks).
    tasks : list[dict[str, Any]], optional
        Initial tasks to seed into the todo list on the first call.
    """

    def __init__(
        self,
        *,
        todo_list: Any | None = None,
        tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        self._todo_list: TodoList | None = todo_list
        self._initial_tasks: list[dict[str, Any]] = tasks or []
        self._seeded: bool = False

    def set_todo_list(self, todo_list: Any) -> None:
        """Set the shared TodoList instance (late binding).

        Used by the composer when wiring the planner together with a
        ``write_todos`` tool that must share the same backing list.

        Parameters
        ----------
        todo_list : TodoList
            The mutable task list shared between planner and tool.
        """
        self._todo_list = todo_list

    def _ensure_seeded(self) -> None:
        """Seed initial tasks into the todo list on first access."""
        if self._seeded or not self._initial_tasks or self._todo_list is None:
            return
        if not self._todo_list.todos:
            seeded: list[dict[str, str]] = [
                {
                    "content": t.get("title", t.get("content", "")),
                    "status": t.get("status", "pending"),
                }
                for t in self._initial_tasks
            ]
            self._todo_list.update(seeded)
        self._seeded = True

    def build_planning_instruction(
        self,
        readonly_context: ReadonlyContext,
        llm_request: LlmRequest,
    ) -> str | None:
        """Inject current task status into the system prompt.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            The current invocation context (read-only).
        llm_request : LlmRequest
            The LLM request being built for this turn.

        Returns
        -------
        str or None
            A task status block, or None if no tasks exist.
        """
        self._ensure_seeded()
        if self._todo_list is None:
            return None
        return self._todo_list.build_status_text()

    def has_pending_tasks(
        self, readonly_context: ReadonlyContext,
    ) -> bool:
        """Check whether the todo list has incomplete tasks.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            The current invocation context (read-only).

        Returns
        -------
        bool
            True if any task is not yet completed.
        """
        self._ensure_seeded()
        if self._todo_list is None:
            return False
        return self._todo_list.has_pending()
