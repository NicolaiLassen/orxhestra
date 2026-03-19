"""TransferTool - hand off control to another agent.

The parent LlmAgent calls this tool when it decides a sub-agent is more
suited to handle the current request. The tool sets a sentinel in the
tool result that LlmAgent intercepts to emit EventActions.transfer_to_agent.

Uses an enum constraint on agent_name to prevent LLMs from hallucinating invalid agent names.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_adk.agents.base_agent import BaseAgent

# Sentinel prefix — LlmAgent intercepts tool results starting with this
# to emit EventActions.transfer_to_agent.
TRANSFER_SENTINEL = "__TRANSFER_TO__"


def make_transfer_tool(available_agents: list[BaseAgent]) -> BaseTool:
    """Create a transfer_to_agent tool constrained to the given agent roster.

    The tool's `agent_name` parameter is restricted to an enum of valid
    names - this prevents the LLM from hallucinating non-existent agents.

    Parameters
    ----------
    available_agents : list[BaseAgent]
        The agents the LLM may transfer to.

    Returns
    -------
    BaseTool
        A LangChain BaseTool that signals a transfer via the
        ``TRANSFER_SENTINEL`` prefix in its return value.

    Raises
    ------
    ValueError
        If available_agents is empty.
    """
    if not available_agents:
        raise ValueError("available_agents must not be empty.")

    agent_names = [a.name for a in available_agents]
    agent_descriptions = "\n".join(
        f"  - {a.name}: {a.description}" for a in available_agents
    )

    # Build a Literal-like enum so the LLM only picks valid names
    AgentNameEnum = Enum("AgentName", {n: n for n in agent_names})  # type: ignore[misc]

    class TransferInput(BaseModel):
        agent_name: AgentNameEnum = Field(  # type: ignore[valid-type]
            description=(
                f"The agent to transfer to. Choose from:\n{agent_descriptions}"
            )
        )
        reason: str = Field(
            default="",
            description="Brief explanation of why this agent is being called.",
        )

    class TransferToAgentTool(BaseTool):
        name: str = "transfer_to_agent"
        description: str = (
            "Transfer the conversation to a specialized agent. "
            "Use this when another agent is better suited to handle the request."
        )
        args_schema: type[BaseModel] = TransferInput

        def _run(self, agent_name: str, reason: str = "") -> str:
            """Return the transfer sentinel string."""
            # agent_name may arrive as an enum member — extract .value
            raw = agent_name.value if isinstance(agent_name, Enum) else agent_name
            return f"{TRANSFER_SENTINEL}{raw}"

        async def _arun(self, agent_name: str, reason: str = "") -> str:
            """Async version of _run."""
            return self._run(agent_name, reason)

    return TransferToAgentTool()
