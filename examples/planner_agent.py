"""Planner agent example - structured Plan-ReAct reasoning.

Demonstrates:
  - PlanReActPlanner for structured chain-of-thought
  - Agent follows: PLAN → REASON → ACT → ANSWER
  - Transparent reasoning with explicit planning tags
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from langchain_adk import LlmAgent, PlanReActPlanner
from langchain_adk.events.event import Event, EventType


# --- Research tools ---

@tool
def get_population(country: str) -> str:
    """Get the population of a country."""
    data = {
        "denmark": "5.9 million",
        "sweden": "10.5 million",
        "norway": "5.5 million",
        "finland": "5.6 million",
    }
    return data.get(country.lower(), f"Population data not found for {country}.")


@tool
def get_gdp(country: str) -> str:
    """Get the GDP of a country in USD."""
    data = {
        "denmark": "$400 billion",
        "sweden": "$590 billion",
        "norway": "$480 billion",
        "finland": "$300 billion",
    }
    return data.get(country.lower(), f"GDP data not found for {country}.")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.(). ")
    if all(c in allowed for c in expression):
        return str(eval(expression))  # noqa: S307
    return "Invalid expression."


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    planner = PlanReActPlanner()

    agent = LlmAgent(
        name="PlannerAgent",
        llm=llm,  # noqa: F821
        tools=[get_population, get_gdp, calculate],
        planner=planner,
        instructions=(
            "You are an analytical assistant. Use the available tools to "
            "research and calculate answers. Follow the planning structure."
        ),
    )

    query = (
        "Compare the Scandinavian countries (Denmark, Sweden, Norway, Finland). "
        "Get each country's population and GDP, then rank them by GDP per capita."
    )

    print(f"Query: {query}\n{'=' * 60}\n")

    async for event in agent.astream(query):
        if event.metadata.get("react_step") == "thought":
            print(f"[THOUGHT] {event.text[:200]}")
        elif event.has_tool_calls:
            print(f"[TOOL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[RESULT] {event.text}")
        elif event.is_final_response():
            print(f"\n[ANSWER]\n{event.text}")


if __name__ == "__main__":
    asyncio.run(main())
