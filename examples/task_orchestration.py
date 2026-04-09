"""Task orchestration example - agent with TodoList and TaskPlanner.

Demonstrates an agent that:
  - Receives a multi-step task
  - Uses write_todos tool to track progress
  - TaskPlanner injects task status into each LLM call
  - Marks tasks complete as it works
"""

from __future__ import annotations

import asyncio

from orxhestra import LlmAgent
from orxhestra.planners.task_planner import TaskPlanner
from orxhestra.prompts.catalog import build_system_prompt
from orxhestra.prompts.context import PromptContext
from orxhestra.tools.todo_tool import TodoList, make_todo_tool


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    todo_list = TodoList()
    todo_tool = make_todo_tool(todo_list)
    planner = TaskPlanner(todo_list=todo_list)

    prompt = build_system_prompt(PromptContext(
        agent_name="ProjectAgent",
        goal="Complete the project tasks systematically.",
        instructions=(
            "Use write_todos to track your work. "
            "Initialize tasks at the start. "
            "Mark each task complete when done. "
            "Only produce a final answer once all required tasks are complete."
        ),
        workflow_instructions=(
            "1. Call write_todos to set up your task list.\n"
            "2. Work through each task in order.\n"
            "3. Update tasks as completed via write_todos.\n"
            "4. Produce a final answer summarising what was accomplished."
        ),
    ))

    agent = LlmAgent(
        name="ProjectAgent",
        llm=llm,  # noqa: F821
        tools=[todo_tool],
        planner=planner,
        instructions=prompt,
    )

    print(f"Agent: {agent.name}\n{'='*40}")

    async for event in agent.astream(
        "Set up a 3-step plan to launch a website: design, develop, deploy.",
    ):
        if event.has_tool_calls:
            print(f"[TOOL] {event.tool_name}")
        elif event.is_final_response():
            print(f"\n[DONE]\n{event.text}")


if __name__ == "__main__":
    asyncio.run(main())
