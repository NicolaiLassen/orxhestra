"""ToolContext - runtime context passed into tool execution.

Gives tools access to the invocation
state and a local ``EventActions`` instance so they can signal escalation,
agent transfer, and state mutations without importing Context
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_adk.events.event_actions import EventActions

if TYPE_CHECKING:
    from langchain_adk.agents.context import Context


class ToolContext:
    """Context object passed to tools during execution.

    Wraps ``Context`` with a tool-scoped ``EventActions`` that the
    agent inspects after the tool returns to apply side-effects (escalate,
    transfer, state updates).

    Parameters
    ----------
    ctx : Context
        The parent invocation context for this agent run.

    Attributes
    ----------
    state : dict[str, Any]
        Shared mutable state from the invocation context. Tools may read
        and write this freely; changes are visible to all subsequent agents
        in the same invocation.
    actions : EventActions
        Side-effects the tool wants to signal. Set ``actions.escalate``
        to stop a ``LoopAgent``, or ``actions.transfer_to_agent`` to hand
        off control to another agent.
    agent_name : str
        Name of the agent currently executing this tool.
    session_id : str
        Current session identifier.
    function_call_id : str, optional
        The ID of the current tool call. Set by the agent before execution.
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self.state: dict[str, Any] = ctx.state
        self.actions: EventActions = EventActions()
        self.agent_name: str = ctx.agent_name
        self.session_id: str = ctx.session_id
        self.function_call_id: str | None = None
        self._confirmation_pending: bool = False

    def request_confirmation(self) -> None:
        """Request user confirmation before proceeding with this tool call.

        When called from a ``before_tool_callback``, the tool execution is
        paused and a confirmation request event is yielded to the caller.
        The caller must approve or reject before execution continues.

        Example
        -------
        ::

            async def confirm_dangerous_tools(ctx, tool_name, tool_args):
                if tool_name in ("delete_file", "drop_table"):
                    tool_ctx = ToolContext(ctx)
                    tool_ctx.request_confirmation()
                    return tool_ctx

            agent = LlmAgent(
                ...,
                before_tool_callback=confirm_dangerous_tools,
            )
        """
        self._confirmation_pending = True

    @property
    def confirmation_pending(self) -> bool:
        """Whether this tool has a pending confirmation request."""
        return self._confirmation_pending
