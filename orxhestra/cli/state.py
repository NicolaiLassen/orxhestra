"""Shared REPL state — passed between app, commands, and builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from orxhestra.cli.todo_tool import TodoList
    from orxhestra.runner import Runner


@dataclass
class ReplState:
    """Mutable state for the interactive REPL."""

    runner: Runner
    session_id: str
    model_name: str
    todo_list: TodoList | None = None
    llm: BaseChatModel | None = None
    turn_count: int = 0
    should_continue: bool = True
    retry_message: str | None = field(default=None, repr=False)
