"""Multi-provider example — same agent with different LLM backends.

Uses ``orxhestra.composer.builders.models.create`` to instantiate models
from OpenAI, Anthropic, and Google with a single factory function.

Prerequisites::

    pip install langchain-openai langchain-anthropic langchain-google-genai

Set the relevant environment variable for the provider you want to use::

    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    export GOOGLE_API_KEY="..."

Run::

    python examples/multi_provider.py
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from orxhestra import LlmAgent
from orxhestra.composer.builders.models import create
from orxhestra.events.event import EventType


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def build_agent(llm: BaseChatModel) -> LlmAgent:
    """Build a simple math agent with the given LLM."""
    return LlmAgent(
        name="MathAgent",
        llm=llm,
        tools=[add],
        instructions="You are a helpful math assistant. Use the add tool.",
    )


async def run_agent(llm: BaseChatModel, label: str) -> None:
    """Run the agent and print the final answer."""
    agent: LlmAgent = build_agent(llm)
    print(f"\n--- {label} ---")
    async for event in agent.astream("What is 17 + 25?"):
        if event.has_tool_calls:
            print(f"  [TOOL] {event.tool_name}({event.tool_input})")
        elif event.is_final_response():
            print(f"  [ANSWER] {event.text}")


async def main() -> None:
    """Create models from multiple providers and run them."""
    # Uncomment the provider(s) you have API keys for:
    raise NotImplementedError(
        "Uncomment one or more provider blocks below and comment out "
        "this raise."
    )

    # openai_llm = create("openai", "gpt-5.4")
    # await run_agent(openai_llm, "OpenAI gpt-5.4")

    # anthropic_llm = create("anthropic", "claude-sonnet-4-6")
    # await run_agent(anthropic_llm, "Anthropic Claude Sonnet")

    # google_llm = create("google", "gemini-2.0-flash")
    # await run_agent(google_llm, "Google Gemini 2.0 Flash")


if __name__ == "__main__":
    asyncio.run(main())
