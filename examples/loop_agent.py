"""Loop agent example - iterative refinement with feedback.

Demonstrates:
  - LoopAgent repeating sub-agents until escalation
  - Writer agent produces draft, reviewer critiques it
  - Loop terminates when reviewer approves (escalate=True)
  - max_iterations as a safety net
"""

from __future__ import annotations

import asyncio

from langchain_adk import LlmAgent, LoopAgent
from langchain_adk.events.event import Event, EventType


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    writer = LlmAgent(
        name="Writer",
        llm=llm,  # noqa: F821
        description="Writes and revises copy based on feedback.",
        instructions=(
            "You are a copywriter. Write a short tagline (1-2 sentences) for "
            "the given product. If you received feedback from the Reviewer, "
            "revise your tagline accordingly. Output ONLY the tagline."
        ),
    )

    reviewer = LlmAgent(
        name="Reviewer",
        llm=llm,  # noqa: F821
        description="Reviews copy and provides feedback or approves.",
        instructions=(
            "You are a senior copywriter reviewing a tagline. "
            "If the tagline is compelling, concise, and ready to publish, "
            "respond with exactly: APPROVED\n"
            "Otherwise, give brief, actionable feedback to improve it. "
            "Be constructive but demanding — only approve truly great work."
        ),
    )

    loop = LoopAgent(
        name="WriteReviewLoop",
        agents=[writer, reviewer],
        max_iterations=5,
    )

    print(f"Running: {loop.name}\n{'=' * 50}")
    print("Product: 'An AI-powered code editor'\n")

    iteration = 0
    async for event in loop.astream(
        "Write a tagline for: An AI-powered code editor that understands your codebase",
    ):
        if event.is_final_response():
            agent = event.agent_name or "?"
            if agent == "Writer":
                iteration += 1
                print(f"--- Iteration {iteration} ---")
                print(f"  [Writer]   {event.text}")
            elif agent == "Reviewer":
                print(f"  [Reviewer] {event.text}")
                print()


if __name__ == "__main__":
    asyncio.run(main())
