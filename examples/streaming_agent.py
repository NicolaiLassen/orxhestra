"""Streaming example - token-by-token output from an LlmAgent.

Demonstrates:
  - Partial events (partial=True) for real-time text streaming
  - Complete events for tool calls / tool results
  - All streaming is automatic — no special config needed
"""

from __future__ import annotations

import asyncio
import sys

# Swap in any LangChain-supported LLM:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

from langchain_core.tools import tool

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is partly cloudy and 18°C."


async def main() -> None:
    # --- Replace with a real LLM ---
    # llm = ChatOpenAI(model="gpt-5.4")
    # llm = ChatAnthropic(model="claude-3-5-haiku-latest")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="StreamingWeatherAgent",
        llm=llm,  # noqa: F821
        tools=[get_weather],
        instructions="You are a helpful weather assistant. Use the get_weather tool.",
    )

    print(f"Running agent (streaming): {agent.name}\n{'=' * 50}")

    async for event in agent.astream(
        "What's the weather in Copenhagen and Berlin?",
    ):
        if event.has_tool_calls:
            print(f"\n[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text or event.error}")
        elif event.partial and event.type == EventType.AGENT_MESSAGE:
            # Stream tokens as they arrive
            sys.stdout.write(".")
            sys.stdout.flush()
        elif event.is_final_response():
            # Final complete answer
            print(f"\n\n[ANSWER]\n{event.text}")


if __name__ == "__main__":
    asyncio.run(main())
