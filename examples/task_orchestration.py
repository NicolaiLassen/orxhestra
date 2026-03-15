"""Task orchestration example - agent with manage_tasks tool.

Demonstrates an agent that:
  - Receives a multi-step task
  - Uses manage_tasks to track progress
  - Marks tasks complete as it works
"""

from __future__ import annotations

import asyncio

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType
from langchain_adk.planners.task_planner import ManageTasksTool
from langchain_adk.prompts.catalog import build_system_prompt
from langchain_adk.prompts.context import PromptContext


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

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
        workflow_instructions=(
            "1. Call manage_tasks(action='initialize', tasks=[...]) to set up tasks.\n"
            "2. Work through each task in order.\n"
            "3. Call manage_tasks(action='complete', task_id='t1') when done.\n"
            "4. Call manage_tasks(action='list') to check progress.\n"
            "5. Produce a final answer summarising what was accomplished."
        ),
    ))

    agent = LlmAgent(
        name="ProjectAgent",
        llm=llm,  # noqa: F821
        tools=[manage_tasks],
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
