"""Langfuse tracing example - single nested trace for the full agent run.

Requires env vars:
  LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  OPENAI_API_KEY (or swap the LLM)

Run:
  uv run python examples/langfuse_tracing.py
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler

from langchain_adk import (
    LlmAgent,
    AgentConfig,
    Runner,
    InMemorySessionService,
)
from langchain_adk.events.event import Event, EventType


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 22C."


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = LlmAgent(
        name="WeatherAgent",
        llm=llm,
        tools=[get_weather],
        instructions="You are a helpful weather assistant. Use the get_weather tool.",
    )

    # Langfuse picks up LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY from env
    langfuse_handler = CallbackHandler()

    run_config = AgentConfig(
        callbacks=[langfuse_handler],
        tags=["example", "langfuse"],
        metadata={"env": "dev"},
    )

    runner = Runner(
        agent=agent,
        app_name="langfuse-example",
        session_service=InMemorySessionService(),
    )

    print(f"Running agent: {agent.name}\n{'=' * 40}")

    async for event in runner.astream(
        user_id="user_1",
        session_id="session_1",
        new_message="What's the weather in Copenhagen and Berlin?",
        config=run_config,
    ):
        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text or event.error}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")

    # Flush to ensure trace is sent
    from langfuse import Langfuse
    Langfuse().flush()
    print("\nTrace sent to Langfuse.")


if __name__ == "__main__":
    asyncio.run(main())
