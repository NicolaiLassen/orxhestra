"""Streaming example — AgentTool sub-agents with event bubbling.

Demonstrates:
  - LlmAgent with AgentTool-wrapped sub-agents
  - Sub-agent events bubbling up with branch attribution
  - Real-time streaming via ``ctx.event_callback``
  - All streaming is automatic — no special config needed

Usage::

    export OPENAI_API_KEY="sk-..."
    python examples/streaming_agent.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.tools import tool

from langchain_adk import LlmAgent
from langchain_adk.events.event import EventType
from langchain_adk.tools.agent_tool import AgentTool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is partly cloudy and 18°C."


@tool
def get_attractions(city: str) -> str:
    """Get top attractions for a city."""
    attractions = {
        "copenhagen": "Tivoli Gardens, The Little Mermaid, Nyhavn",
        "berlin": "Brandenburg Gate, Museum Island, East Side Gallery",
    }
    return attractions.get(city.lower(), f"Popular landmarks in {city}")


def _print_event(event) -> None:
    """Pretty-print a single streaming event."""
    branch = f" branch={event.branch}" if event.branch else ""
    agent = f" agent={event.agent_name}" if event.agent_name else ""

    if event.has_tool_calls:
        print(f"\n  [TOOL CALL]{branch}{agent} {event.tool_name}({event.tool_input})")
    elif event.type == EventType.TOOL_RESPONSE:
        print(f"  [TOOL RESULT]{branch}{agent} {event.text or event.error}")
    elif event.partial and event.type == EventType.AGENT_MESSAGE:
        # Stream tokens inline
        sys.stdout.write(".")
        sys.stdout.flush()
    elif event.is_final_response():
        text = event.text or ""
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n  [ANSWER]{branch}{agent}\n  {preview}")
    else:
        print(f"  [EVENT] {event.type}{branch}{agent}")


async def main() -> None:
    """Run a parent LlmAgent that delegates to two AgentTool sub-agents."""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this example.")
        return

    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

    # Sub-agent 1: weather expert
    weather_agent = LlmAgent(
        name="WeatherAgent",
        llm=llm,
        tools=[get_weather],
        instructions="You are a weather assistant. Use the get_weather tool to answer.",
    )

    # Sub-agent 2: travel guide
    travel_agent = LlmAgent(
        name="TravelAgent",
        llm=llm,
        tools=[get_attractions],
        instructions="You are a travel guide. Use the get_attractions tool to answer.",
    )

    # Parent agent with AgentTool-wrapped sub-agents
    planner = LlmAgent(
        name="TripPlanner",
        llm=llm,
        tools=[
            AgentTool(weather_agent),
            AgentTool(travel_agent),
        ],
        instructions=(
            "You are a trip planner. Use the WeatherAgent tool to get weather info "
            "and the TravelAgent tool to get attraction info. "
            "Combine the results into a helpful travel summary."
        ),
    )

    print("=" * 60)
    print("AgentTool streaming: LlmAgent with sub-agent tools")
    print("=" * 60)

    async for event in planner.astream("Tell me about Copenhagen"):
        _print_event(event)

    print()


if __name__ == "__main__":
    asyncio.run(main())
