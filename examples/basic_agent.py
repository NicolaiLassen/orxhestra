"""Basic agent example - LlmAgent with a single tool.

Shows the simplest possible SDK usage:
  1. Define a tool
  2. Create an LlmAgent
  3. Run it and print events
"""

from __future__ import annotations

import asyncio

# Swap in any LangChain-supported LLM:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.tools import tool

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Stub - replace with a real API call
    return f"The weather in {city} is sunny and 22°C."


async def main() -> None:
    # --- Replace with a real LLM ---
    # llm = ChatOpenAI(model="gpt-5.4")
    # llm = ChatAnthropic(model="claude-3-5-haiku-latest")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="WeatherAgent",
        llm=llm,  # noqa: F821
        tools=[get_weather],
        instructions="You are a helpful weather assistant. Use the get_weather tool.",
    )

    print(f"Running agent: {agent.name}\n{'='*40}")

    async for event in agent.astream("What's the weather in Copenhagen?"):
        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text or event.error}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
