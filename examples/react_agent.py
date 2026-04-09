"""ReActAgent example — structured Reason+Act loop with tool calls.

Requires a real LLM with structured output support (OpenAI, Anthropic).

Run::

    python examples/react_agent.py
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from orxhestra.agents import ReActAgent


@tool
def lookup_capital(country: str) -> str:
    """Look up the capital city of a country."""
    capitals: dict[str, str] = {
        "france": "Paris",
        "japan": "Tokyo",
        "brazil": "Brasilia",
        "australia": "Canberra",
    }
    return capitals.get(country.lower(), f"Unknown capital for {country}")


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city."""
    populations: dict[str, str] = {
        "paris": "~2.1 million",
        "tokyo": "~14 million",
        "brasilia": "~3.0 million",
        "canberra": "~460 thousand",
    }
    return populations.get(city.lower(), f"Unknown population for {city}")


async def main() -> None:
    """Run a ReActAgent that reasons step by step."""
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    agent = ReActAgent(
        name="GeoAgent",
        llm=llm,  # noqa: F821
        tools=[lookup_capital, get_population],
        description="Answers geography questions step by step.",
        max_iterations=6,
    )

    print(f"Agent: {agent.name}\n{'=' * 50}")

    async for event in agent.astream(
        "What is the population of the capital of Japan?",
    ):
        meta: dict = event.metadata or {}
        step: str | None = meta.get("react_step")

        if step == "thought" and not event.partial:
            print(f"[THOUGHT] {event.text}")
        elif step == "action":
            print(f"[ACTION]  {meta.get('action')}({meta.get('action_input')})")
        elif step == "observation":
            print(f"[OBSERVE] {event.text}")
        elif event.is_final_response():
            print(f"\n[ANSWER]  {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
