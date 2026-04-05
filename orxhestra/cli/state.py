"""Shared REPL state — passed between app, commands, and builder."""

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from orxhestra.runner import Runner
from orxhestra.tools.todo_tool import TodoList


class ReplState(BaseModel):
    """Mutable state for the interactive REPL."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Runner = Field(description="Active Runner instance for agent execution.")
    session_id: str = Field(description="Current session identifier.")
    model_name: str = Field(description="Name of the LLM model in use.")
    todo_list: TodoList | None = Field(
        default=None, description="Active todo list for task tracking."
    )
    llm: BaseChatModel | None = Field(
        default=None, description="LangChain LLM instance."
    )
    turn_count: int = Field(
        default=0, description="Number of completed REPL turns."
    )
    should_continue: bool = Field(
        default=True, description="Whether the REPL loop should continue."
    )
    retry_message: str | None = Field(
        default=None, repr=False, description="Message to retry on next turn."
    )
