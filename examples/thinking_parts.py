"""Thinking parts example — processing extended thinking/reasoning tokens.

Shows how to detect and display ThinkingPart content from models that
support extended thinking (e.g. Anthropic Claude with thinking enabled).

Prerequisites::

    pip install orxhestra[anthropic]

Run::

    python examples/thinking_parts.py
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from orxhestra import LlmAgent
from orxhestra.events.event import EventType


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the result."""
    # Stub - replace with a real evaluator
    return str(eval(expression))  # noqa: S307


async def main() -> None:
    # --- Replace with a real LLM that supports thinking ---
    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(
    #     model="claude-sonnet-4-6",
    #     thinking={"type": "enabled", "budget_tokens": 5000},
    # )
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="ThinkingAgent",
        llm=llm,  # noqa: F821
        tools=[calculate],
        instructions="You are a math assistant. Think through problems step by step.",
    )

    print(f"Running agent: {agent.name}\n{'='*40}")

    async for event in agent.astream(
        "What is the sum of the first 10 prime numbers?"
    ):
        # Check for thinking/reasoning content
        if event.thinking:
            print(f"[THINKING] {event.thinking[:200]}...")

        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
