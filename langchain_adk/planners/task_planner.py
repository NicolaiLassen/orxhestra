"""TaskPlanner - a planner that manages a structured task board.

Combines two roles:

1. **Planner**: Before each LLM call, injects the current task board
   status into the system prompt so the model always knows which tasks
   are pending, in-progress, or completed.

2. **Tool provider**: Exposes ``get_manage_tasks_tool()`` which returns
   a LangChain tool the agent can call to create, update, and complete
   tasks. Attach this tool to the ``LlmAgent`` alongside the planner.

Usage::

    from langchain_adk.planners import TaskPlanner

    planner = TaskPlanner(
        tasks=[
            {"title": "Research topic", "required": True},
            {"title": "Write summary"},
        ]
    )

    agent = LlmAgent(
        name="researcher",
        llm=llm,
        tools=[planner.get_manage_tasks_tool()],
        planner=planner,
        instructions="You are a research assistant.",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_adk.planners.base_planner import BasePlanner
from langchain_adk.planners.constants import StateKey, TaskStatus
from langchain_adk.planners.task_board import (
    apply_task_action,
    initialize_task_board,
    list_task_items,
)

if TYPE_CHECKING:
    from langchain_adk.agents.readonly_context import ReadonlyContext
    from langchain_adk.agents.context import Context
    from langchain_adk.models.llm_request import LlmRequest


# ---------------------------------------------------------------------------
# TaskPlanner
# ---------------------------------------------------------------------------


class TaskPlanner(BasePlanner):
    """Planner that tracks structured tasks and injects progress into prompts.

    Attach to an ``LlmAgent`` together with ``get_manage_tasks_tool()`` to
    give the agent a task board it can read and write through the planning
    loop.

    Parameters
    ----------
    tasks : list[dict[str, Any]], optional
        Initial task definitions. Each dict may contain ``title``,
        ``description``, ``status``, and ``required`` keys. Tasks are
        seeded into the board on the first invocation if no board exists
        yet in the session state.

    Attributes
    ----------
    initial_tasks : list[dict[str, Any]]
        The task definitions provided at construction time.
    """

    def __init__(self, tasks: list[dict[str, Any]] | None = None) -> None:
        self.initial_tasks: list[dict[str, Any]] = tasks or []

    def _ensure_board(self, ctx: Context) -> dict[str, Any]:
        """Seed the task board on the first call if it does not exist yet."""
        board = ctx.state.get(StateKey.TASK_BOARD)
        if not board and self.initial_tasks:
            board = initialize_task_board(self.initial_tasks)
            ctx.state[StateKey.TASK_BOARD] = board
        return board or {}

    def build_planning_instruction(
        self,
        readonly_context: ReadonlyContext,
        llm_request: LlmRequest,
    ) -> str | None:
        """Inject current task board status into the system prompt.

        Parameters
        ----------
        readonly_context : ReadonlyContext
            The current invocation context (read-only).
        llm_request : LlmRequest
            The LLM request being built for this turn.

        Returns
        -------
        str or None
            A task status block, or None if no task board exists.
        """
        board = readonly_context.state.get(StateKey.TASK_BOARD)
        if not board:
            return None

        items = list_task_items(board)
        if not items:
            return None

        lines = ["Current task board:"]
        for t in items:
            status_marker = "[done]" if t["status"] == TaskStatus.COMPLETED else f"[{t['status']}]"
            req = " (required)" if t.get("required") else ""
            desc = f" - {t['description']}" if t.get("description") else ""
            lines.append(f"  {t['id']} {status_marker} {t['title']}{req}{desc}")

        summary = board.get("summary", {})
        lines.append(
            f"\nProgress: {summary.get('completed', 0)}/{summary.get('total', 0)} tasks completed."
        )
        return "\n".join(lines)

    def get_manage_tasks_tool(self) -> BaseTool:
        """Return a LangChain tool for reading and writing the task board.

        The returned tool requires context injection before each call.
        Pass it to ``LlmAgent`` and the agent will inject context
        automatically via ``inject_context()``.

        Returns
        -------
        BaseTool
            A ``ManageTasksTool`` instance bound to this planner.
        """
        return ManageTasksTool(planner=self)


# ---------------------------------------------------------------------------
# ManageTasksTool
# ---------------------------------------------------------------------------


class _ManageTasksInput(BaseModel):
    """Input schema for the manage_tasks tool."""

    action: str = Field(
        description=(
            "Task action to perform. One of: "
            "initialize, list, create, update, complete, remove."
        )
    )
    task_id: str | None = Field(
        default=None,
        description="Task ID (e.g. 't1'). Required for update, complete, remove.",
    )
    title: str | None = Field(
        default=None,
        description="Task title. Required for create; used to look up tasks by name.",
    )
    description: str | None = Field(
        default=None,
        description="Task description.",
    )
    status: str | None = Field(
        default=None,
        description="Task status: pending, in_progress, completed, blocked.",
    )
    tasks: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of tasks for initialize action.",
    )


class ManageTasksTool(BaseTool):
    """Tool that reads and writes the task board in Context.state.

    The task board is stored at ``state[StateKey.TASK_BOARD]``. Agents use
    this tool to track progress on multi-step work.

    Attributes
    ----------
    planner : TaskPlanner
        The owning TaskPlanner, used to seed initial tasks.
    """

    name: str = "manage_tasks"
    description: str = (
        "Manage the task board: create, update, complete, list, or remove tasks. "
        "Use 'initialize' to set up tasks at the start of a session. "
        "Use 'complete' when a task is done. "
        "Use 'list' to see current task progress."
    )
    args_schema: type[BaseModel] = _ManageTasksInput

    planner: Any = None
    _ctx: Any | None = None

    def inject_context(self, ctx: Context) -> None:
        """Inject the invocation context before tool execution.

        Parameters
        ----------
        ctx : Context
            The invocation context to inject.
        """
        object.__setattr__(self, "_ctx", ctx)

    def _run(self, **kwargs: Any) -> str:
        """Raise an error - ManageTasksTool is async-only."""
        raise NotImplementedError("Use async ainvoke.")

    async def _arun(
        self,
        action: str,
        task_id: str | None = None,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        tasks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute a task board action asynchronously.

        Parameters
        ----------
        action : str
            The action to perform (initialize, list, create, update,
            complete, remove).
        task_id : str, optional
            Task ID for targeted operations.
        title : str, optional
            Task title for create or name-based lookup.
        description : str, optional
            Task description for create or update.
        status : str, optional
            New status for update operations.
        tasks : list[dict[str, Any]], optional
            Task list for initialize action.

        Returns
        -------
        str
            A human-readable result message for the LLM.
        """
        ctx: Any | None = object.__getattribute__(self, "_ctx")
        if ctx is None:
            return "Error: manage_tasks tool has no context. Call inject_context(ctx) first."

        planner: TaskPlanner | None = object.__getattribute__(self, "planner")
        if planner is not None:
            board = planner._ensure_board(ctx)
        else:
            board = ctx.state.get(StateKey.TASK_BOARD)

        action_lower = action.strip().lower()

        if action_lower in ("initialize", "init", "start"):
            new_board = initialize_task_board(tasks or [], existing=board)
            ctx.state[StateKey.TASK_BOARD] = new_board
            items = list_task_items(new_board)
            return f"Task board initialized with {len(items)} task(s).\n{_format_tasks(items)}"

        if action_lower in ("list", "view", "show"):
            if not board:
                return "No task board found. Use 'initialize' to set up tasks."
            items = list_task_items(board)
            return _format_tasks(items)

        updated_board, _, message = apply_task_action(
            board or {},
            action=action,
            actor=ctx.agent_name,
            task_id=task_id,
            title=title,
            description=description,
            status=status,
        )
        ctx.state[StateKey.TASK_BOARD] = updated_board
        return message


def _format_tasks(tasks: list[dict[str, Any]]) -> str:
    """Format a list of task dicts as a human-readable string."""
    if not tasks:
        return "No tasks."
    lines = []
    for t in tasks:
        tag = "[done]" if t["status"] == "completed" else f"[{t['status']}]"
        req = " (required)" if t.get("required") else ""
        desc = f": {t['description']}" if t.get("description") else ""
        lines.append(f"  {t['id']} {tag} {t['title']}{req}{desc}")
    return "\n".join(lines)
