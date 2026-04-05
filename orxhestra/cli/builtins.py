"""Register CLI-specific tools as composer builtins.

Calling :func:`register_cli_builtins` makes ``write_todos``, ``task``,
``sleep``, ``background_tasks``, and ``tool_search`` available in any
orx YAML via ``builtin: "<name>"``.

The shared :class:`TodoList` is accessible via :func:`get_todo_list` so
the REPL can render task progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orxhestra.composer.builders.tools import register_builtin
from orxhestra.tools.todo_tool import TodoList, make_todo_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_shared_todo_list: TodoList | None = None


def register_cli_builtins(
    workspace: str,
    llm: BaseChatModel | None = None,
) -> None:
    """Register CLI builtins into the composer tool registry.

    Must be called before building a orx spec so that YAML files
    can reference ``builtin: "write_todos"``, ``builtin: "task"``,
    ``builtin: "sleep"``, ``builtin: "background_tasks"``, and
    ``builtin: "tool_search"``.
    """
    global _shared_todo_list  # noqa: PLW0603
    _shared_todo_list = TodoList()

    todo_list = _shared_todo_list

    def _todo_factory() -> list:
        """Create the write_todos tool list."""
        return [make_todo_tool(todo_list)]

    register_builtin("write_todos", _todo_factory)

    # Memory tools — always available.
    def _memory_factory() -> list:
        """Create the memory tools."""
        from orxhestra.memory.file_memory_service import get_memory_dir
        from orxhestra.tools.memory_tools import make_memory_tools

        return make_memory_tools(get_memory_dir(workspace))

    register_builtin("memory", _memory_factory)

    # Sleep tool — always available.
    def _sleep_factory() -> list:
        """Create the sleep tool."""
        from orxhestra.tools.sleep_tool import make_sleep_tool

        return [make_sleep_tool()]

    register_builtin("sleep", _sleep_factory)

    if llm is not None:
        _llm = llm
        _ws = workspace

        def _task_factory() -> list:
            """Create the blocking task delegation tool."""
            from orxhestra.tools.filesystem import make_filesystem_tools
            from orxhestra.tools.shell import make_shell_tools
            from orxhestra.tools.task_tools import make_task_tool

            fs = make_filesystem_tools(workspace=_ws)
            sh = make_shell_tools(
                workspace=_ws, timeout=120, max_output_bytes=200_000,
            )
            return [make_task_tool(_llm, [*fs, *sh], _ws)]

        register_builtin("task", _task_factory)

        def _bg_tasks_factory() -> list:
            """Create background task lifecycle tools."""
            from orxhestra.tools.filesystem import make_filesystem_tools
            from orxhestra.tools.shell import make_shell_tools
            from orxhestra.tools.task_tools import make_background_task_tools

            fs = make_filesystem_tools(workspace=_ws)
            sh = make_shell_tools(
                workspace=_ws, timeout=120, max_output_bytes=200_000,
            )
            return make_background_task_tools(_llm, [*fs, *sh], _ws)

        register_builtin("background_tasks", _bg_tasks_factory)


def get_todo_list() -> TodoList | None:
    """Return the shared TodoList instance, or None if not registered."""
    return _shared_todo_list
