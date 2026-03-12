"""Task orchestration example - agent with manage_tasks tool.

Demonstrates an agent that:
  - Receives a multi-step task
  - Uses manage_tasks to track progress
  - Marks tasks complete as it works
"""

from __future__ import annotations

import asyncio

from langchain_adk.agents.llm_agent import LlmAgent
from langchain_adk.context.invocation_context import InvocationContext
from langchain_adk.events.event import FinalAnswerEvent, ToolCallEvent
from langchain_adk.planners.task_planner import ManageTasksTool
from langchain_adk.prompts.catalog import build_system_prompt
from langchain_adk.prompts.context import PromptContext


async def main() -> None:
    raise NotImplementedError("Set llm= below with a real LangChain model.")

    manage_tasks = ManageTasksTool()

    prompt = build_system_prompt(PromptContext(
        agent_name="ProjectAgent",
        goal="Complete the project tasks systematically.",
        instructions=(
            "Use manage_tasks to track your work. "
            "Initialize tasks at the start. "
            "Mark each task complete when done. "
            "Only produce a final answer once all required tasks are complete."
        ),
        workflow_lines=[
            "1. Call manage_tasks(action='initialize', tasks=[...]) to set up tasks.",
            "2. Work through each task in order.",
            "3. Call manage_tasks(action='complete', task_id='t1') when done.",
            "4. Call manage_tasks(action='list') to check progress.",
            "5. Produce a final answer summarising what was accomplished.",
        ],
    ))

    agent = LlmAgent(
        name="ProjectAgent",
        llm=llm,  # noqa: F821
        tools=[manage_tasks],
        instructions=prompt,
    )

    ctx = InvocationContext(
        session_id="tasks-demo",
        agent_name=agent.name,
    )

    # Inject context into the tool so it can read/write state
    manage_tasks.inject_context(ctx)

    print(f"Agent: {agent.name}\n{'='*40}")

    async for event in agent.run(
        "Set up a 3-step plan to launch a website: design, develop, deploy.",
        ctx=ctx,
    ):
        if isinstance(event, ToolCallEvent):
            print(f"[TOOL] {event.tool_name}")
        elif isinstance(event, FinalAnswerEvent):
            print(f"\n[DONE]\n{event.answer}")

    # Show final task board state
    from langchain_adk.planners.task_board import list_task_items
    from langchain_adk.planners.constants import StateKey
    board = ctx.state.get(StateKey.TASK_BOARD)
    if board:
        print(f"\nTask board: {list_task_items(board)}")


if __name__ == "__main__":
    asyncio.run(main())
