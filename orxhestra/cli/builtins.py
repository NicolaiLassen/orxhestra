"""Register CLI-specific tools as composer builtins.

Calling :func:`register_cli_builtins` makes ``write_todos`` and ``task``
available in any orx YAML via ``builtin: "write_todos"`` etc.

The shared :class:`TodoList` is accessible via :func:`get_todo_list` so
the REPL can render task progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orxhestra.cli.todo_tool import TodoList, make_todo_tool
from orxhestra.composer.builders.tools import register_builtin

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_shared_todo_list: TodoList | None = None


def register_cli_builtins(
    workspace: str,
    llm: BaseChatModel | None = None,
) -> None:
    """Register CLI builtins into the composer tool registry.

    Must be called before building a orx spec so that YAML files
    can reference ``builtin: "write_todos"`` and ``builtin: "task"``.
    """
    global _shared_todo_list
    _shared_todo_list = TodoList()

    todo_list = _shared_todo_list

    def _todo_factory() -> list:
        """Create the write_todos tool list."""
        return [make_todo_tool(todo_list)]

    register_builtin("write_todos", _todo_factory)

    if llm is not None:
        _llm = llm
        _ws = workspace

        def _task_factory() -> list:
            """Create the task delegation tool list."""
            from orxhestra.cli.task_tool import make_task_tool
            from orxhestra.tools.filesystem import make_filesystem_tools
            from orxhestra.tools.shell import make_shell_tools

            fs = make_filesystem_tools(workspace=_ws)
            sh = make_shell_tools(workspace=_ws, timeout=120, max_output_bytes=200_000)
            return [make_task_tool(_llm, [*fs, *sh], _ws)]

        register_builtin("task", _task_factory)


def get_todo_list() -> TodoList | None:
    """Return the shared TodoList instance, or None if not registered."""
    return _shared_todo_list
