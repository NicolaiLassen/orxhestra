"""Task tools — blocking delegation and background task lifecycle.

Two patterns for sub-agent delegation:

- **Blocking** (``make_task_tool``): spawns a sub-agent, waits for
  it to finish, returns the result. Use when the main agent needs
  the answer before continuing.

- **Background** (``make_background_task_tools``): spawns a sub-agent
  in the background and returns a task ID immediately. The main agent
  can check status and read output later. Use for parallel work.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool, StructuredTool


@dataclass
class BackgroundTask:
    """A background task with stored output."""

    id: str
    subject: str
    description: str
    status: str = "running"
    output: str = ""
    created_at: float = field(default_factory=time.time)
    _task: asyncio.Task[Any] | None = field(default=None, repr=False)


class TaskStore:
    """In-memory store for background tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}

    def add(self, task: BackgroundTask) -> None:
        """Add a task to the store."""
        self._tasks[task.id] = task

    def get(self, task_id: str) -> BackgroundTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def list_all(self) -> list[BackgroundTask]:
        """List all tasks, newest first."""
        return sorted(
            self._tasks.values(),
            key=lambda t: t.created_at,
            reverse=True,
        )

    def remove(self, task_id: str) -> None:
        """Remove a task from the store."""
        self._tasks.pop(task_id, None)


def make_task_tool(
    llm: Any,
    tools: list[BaseTool],
    workspace: str,
) -> BaseTool:
    """Create a blocking task delegation tool.

    Spawns an ephemeral sub-agent with isolated context, waits for
    it to finish, and returns the result.

    Parameters
    ----------
    llm : BaseChatModel
        LLM for the sub-agent.
    tools : list[BaseTool]
        Tools available to the sub-agent.
    workspace : str
        Workspace directory path.

    Returns
    -------
    BaseTool
        A ``task`` structured tool.
    """

    async def task(description: str) -> str:
        """Delegate a subtask to a fresh agent with isolated context.

        Args:
            description: Detailed description of the subtask.
        """
        from orxhestra.agents.llm_agent import LlmAgent

        sub_agent = LlmAgent(
            name="sub-task",
            llm=llm,
            tools=list(tools),
            instructions=(
                f"You are a sub-agent handling a specific task. "
                f"Workspace: {workspace}\n\n"
                f"Complete the task and return a concise summary."
            ),
            max_iterations=15,
        )

        final_answer: str = ""
        async for event in sub_agent.astream(description):
            if event.is_final_response() and event.text:
                final_answer = event.text

        return final_answer or "Completed without producing a summary."

    return StructuredTool.from_function(
        coroutine=task,
        name="task",
        description=(
            "Delegate a subtask to a fresh agent with isolated context. "
            "The sub-agent has the same tools but a clean conversation. "
            "Blocks until the sub-agent finishes and returns the result."
        ),
    )


def make_background_task_tools(
    llm: Any,
    tools: list[BaseTool],
    workspace: str,
    store: TaskStore | None = None,
) -> list[BaseTool]:
    """Create the full set of background task lifecycle tools.

    Parameters
    ----------
    llm : BaseChatModel
        LLM for spawned sub-agents.
    tools : list[BaseTool]
        Tools available to sub-agents.
    workspace : str
        Workspace directory path.
    store : TaskStore, optional
        Shared task store. Created if not provided.

    Returns
    -------
    list[BaseTool]
        Six tools: task_create, task_list, task_get, task_update,
        task_stop, task_output.
    """
    task_store = store or TaskStore()

    async def _run_subtask(
        task: BackgroundTask, description: str,
    ) -> None:
        """Run a subtask agent and capture its output."""
        from orxhestra.agents.llm_agent import LlmAgent

        agent = LlmAgent(
            name=f"task-{task.id[:8]}",
            llm=llm,
            tools=list(tools),
            instructions=(
                f"You are a background sub-agent. Workspace: {workspace}\n\n"
                f"Complete the task and return a concise summary."
            ),
            max_iterations=15,
        )

        try:
            output_parts: list[str] = []
            async for event in agent.astream(description):
                if event.is_final_response() and event.text:
                    output_parts.append(event.text)
            task.output = "\n".join(output_parts) or "Completed (no output)."
            task.status = "completed"
        except asyncio.CancelledError:
            task.status = "stopped"
            task.output = "Task was stopped."
        except Exception as exc:
            task.status = "failed"
            task.output = f"Error: {exc}"

    async def task_create(subject: str, description: str) -> str:
        """Create and start a background task.

        Args:
            subject: Brief title for the task.
            description: Detailed description of what to do.
        """
        task_id = str(uuid4())[:8]
        task = BackgroundTask(id=task_id, subject=subject, description=description)
        task_store.add(task)

        loop = asyncio.get_event_loop()
        task._task = loop.create_task(_run_subtask(task, description))

        return f"Task created: {task_id} — {subject}"

    async def task_list() -> str:
        """List all background tasks with their status."""
        tasks = task_store.list_all()
        if not tasks:
            return "No tasks."
        lines: list[str] = []
        for t in tasks:
            age = int(time.time() - t.created_at)
            lines.append(f"  {t.id} [{t.status}] {t.subject} ({age}s ago)")
        return "\n".join(lines)

    async def task_get(task_id: str) -> str:
        """Get details of a specific task.

        Args:
            task_id: The task ID returned by task_create.
        """
        task = task_store.get(task_id)
        if task is None:
            return f"Error: task '{task_id}' not found."
        return (
            f"ID: {task.id}\n"
            f"Subject: {task.subject}\n"
            f"Status: {task.status}\n"
            f"Description: {task.description}\n"
            f"Output: {task.output or '(pending)'}"
        )

    async def task_update(task_id: str, status: str) -> str:
        """Update the status of a task.

        Args:
            task_id: The task ID.
            status: New status (running, completed, failed, stopped).
        """
        task = task_store.get(task_id)
        if task is None:
            return f"Error: task '{task_id}' not found."
        task.status = status
        return f"Task {task_id} status updated to '{status}'."

    async def task_stop(task_id: str) -> str:
        """Stop a running background task.

        Args:
            task_id: The task ID to stop.
        """
        task = task_store.get(task_id)
        if task is None:
            return f"Error: task '{task_id}' not found."
        if task._task and not task._task.done():
            task._task.cancel()
            return f"Task {task_id} stop requested."
        return f"Task {task_id} is already {task.status}."

    async def task_output(task_id: str) -> str:
        """Read the output of a completed or running task.

        Args:
            task_id: The task ID to read output from.
        """
        task = task_store.get(task_id)
        if task is None:
            return f"Error: task '{task_id}' not found."
        if task.status == "running":
            return f"Task {task_id} is still running. Output so far:\n{task.output or '(none yet)'}"
        return task.output or "(no output)"

    return [
        StructuredTool.from_function(
            coroutine=task_create,
            name="task_create",
            description=(
                "Create and start a background task. Returns a task ID "
                "you can use to check status and read output later."
            ),
        ),
        StructuredTool.from_function(
            coroutine=task_list,
            name="task_list",
            description="List all background tasks with their status.",
        ),
        StructuredTool.from_function(
            coroutine=task_get,
            name="task_get",
            description="Get details of a specific background task.",
        ),
        StructuredTool.from_function(
            coroutine=task_update,
            name="task_update",
            description="Update the status of a background task.",
        ),
        StructuredTool.from_function(
            coroutine=task_stop,
            name="task_stop",
            description="Stop a running background task.",
        ),
        StructuredTool.from_function(
            coroutine=task_output,
            name="task_output",
            description=(
                "Read the output of a background task. Works for "
                "completed, failed, and in-progress tasks."
            ),
        ),
    ]
