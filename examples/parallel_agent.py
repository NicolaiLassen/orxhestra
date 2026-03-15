"""Parallel agent example - run multiple agents concurrently.

Demonstrates:
  - ParallelAgent running sub-agents in parallel
  - Each agent researches independently with its own tools
  - Events from all agents merged into a single stream
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from langchain_adk import LlmAgent, ParallelAgent
from langchain_adk.events.event import Event, EventType


# --- Simulated data sources ---

@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a topic."""
    return (
        f"Wikipedia result for '{query}': LLM agents are AI systems that use "
        "large language models to reason, plan, and take actions autonomously."
    )


@tool
def search_news(query: str) -> str:
    """Search recent news articles."""
    return (
        f"News result for '{query}': Major tech companies are racing to build "
        "autonomous AI agents that can perform complex tasks end-to-end."
    )


@tool
def search_arxiv(query: str) -> str:
    """Search academic papers on arXiv."""
    return (
        f"arXiv result for '{query}': Recent papers show chain-of-thought "
        "prompting and tool use significantly improve agent reliability."
    )


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # Three research agents, each with a different source
    wiki_agent = LlmAgent(
        name="WikiResearcher",
        llm=llm,  # noqa: F821
        tools=[search_wikipedia],
        description="Researches topics using Wikipedia.",
        instructions="Search Wikipedia for the given topic and summarize findings.",
    )

    news_agent = LlmAgent(
        name="NewsResearcher",
        llm=llm,  # noqa: F821
        tools=[search_news],
        description="Researches topics using recent news.",
        instructions="Search news for the given topic and summarize findings.",
    )

    arxiv_agent = LlmAgent(
        name="ArxivResearcher",
        llm=llm,  # noqa: F821
        tools=[search_arxiv],
        description="Researches topics using academic papers.",
        instructions="Search arXiv for the given topic and summarize findings.",
    )

    # Run all three in parallel
    parallel = ParallelAgent(
        name="ParallelResearchTeam",
        agents=[wiki_agent, news_agent, arxiv_agent],
    )

    print(f"Running: {parallel.name}\n{'=' * 50}")
    print("(3 agents researching in parallel)\n")

    async for event in parallel.astream("AI agents"):
        if event.has_tool_calls:
            print(f"  [{event.agent_name}] TOOL CALL: {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            result_preview = (event.text or "")[:80]
            print(f"  [{event.agent_name}] RESULT: {result_preview}...")
        elif event.is_final_response():
            print(f"\n[{event.agent_name} ANSWER]\n{event.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
