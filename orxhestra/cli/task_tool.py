"""Subagent delegation - spawn ephemeral agents for isolated subtasks.

Provides a `task` tool that creates a fresh LlmAgent with the same
tools but an isolated context window. Useful for complex subtasks
that would pollute the main conversation context.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool


def make_task_tool(
    llm: BaseChatModel,
    tools: list[BaseTool],
    workspace: str,
) -> BaseTool:
    """Create a task delegation tool that spawns ephemeral sub-agents."""

    async def task(description: str) -> str:
        """Delegate a complex subtask to a fresh agent with isolated context.

        The sub-agent has the same filesystem and shell tools but a clean
        conversation history. Use this for:
        - Multi-step operations that would clutter the main conversation
        - Exploratory research (reading many files, searching broadly)
        - Isolated refactoring that doesn't need main conversation context

        Args:
            description: Detailed description of the subtask to accomplish.
                Be specific about expected outputs and success criteria.
        """
        from orxhestra.agents.llm_agent import LlmAgent

        sub_agent = LlmAgent(
            name="sub-task",
            llm=llm,
            tools=list(tools),
            instructions=(
                f"You are a sub-agent handling a specific task. "
                f"Workspace: {workspace}\n\n"
                f"Complete the following task thoroughly and return a concise "
                f"summary of what you did and the results."
            ),
            max_iterations=15,
        )

        final_answer: str = ""
        async for event in sub_agent.astream(description):
            if event.is_final_response() and event.text:
                final_answer = event.text

        return final_answer or "Sub-agent completed without producing a summary."

    return StructuredTool.from_function(
        coroutine=task,
        name="task",
        description=(
            "Delegate a complex subtask to a fresh agent with isolated context. "
            "The sub-agent has filesystem and shell access but a clean conversation. "
            "Use for multi-step operations, exploratory research, or isolated work "
            "that would clutter the main conversation."
        ),
    )
