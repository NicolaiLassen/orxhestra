"""Shared REPL state — passed between app, commands, and builder."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field

from orxhestra.runner import Runner
from orxhestra.tools.todo_tool import TodoList


class ReplState(BaseModel):
    """Mutable state for the interactive REPL.

    See Also
    --------
    Runner : Wraps the active agent with session persistence.
    TodoList : Shared task list rendered in the CLI.
    orxhestra.cli.commands : Slash-command handlers that mutate this state.

    Attributes
    ----------
    runner : Runner
        Active Runner instance for agent execution.
    session_id : str
        Current session identifier.
    model_name : str
        Name of the LLM model in use.
    todo_list : TodoList, optional
        Active todo list for task tracking.
    model : BaseChatModel, optional
        LangChain LLM instance.
    turn_count : int
        Number of completed REPL turns.
    should_continue : bool
        Whether the REPL loop should continue.
    retry_message : str, optional
        Message to retry on next turn.
    auto_approve : bool
        Whether tool calls are auto-approved without user confirmation.
    signer_did : str, optional
        DID of the active Ed25519 signing identity attached via
        ``--identity`` or ``$ORX_IDENTITY``.  ``None`` when signing
        is disabled.
    identity_key_path : str, optional
        On-disk path of the active signing key file, for display in
        the ``/session`` command.  ``None`` when signing is disabled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Runner = Field(description="Active Runner instance for agent execution.")
    session_id: str = Field(description="Current session identifier.")
    model_name: str = Field(description="Name of the LLM model in use.")
    effort: str | None = Field(
        default=None,
        description="Active reasoning effort level (low|medium|high).",
    )
    spec_raw: dict | None = Field(
        default=None,
        description=(
            "Parsed orx YAML used to build the agent tree. Drives the "
            "/agent command's per-agent introspection."
        ),
    )
    todo_list: TodoList | None = Field(
        default=None, description="Active todo list for task tracking."
    )
    model: BaseChatModel | None = Field(
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
    auto_approve: bool = Field(
        default=False, description="Whether tool calls are auto-approved without user confirmation."
    )
    signer_did: str | None = Field(
        default=None,
        description="Active signer DID — populated when --identity is in use.",
    )
    identity_key_path: str | None = Field(
        default=None,
        description="Path of the active signing key file.",
    )
