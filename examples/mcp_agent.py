"""MCP agent example - use tools from an MCP server.

Demonstrates:
  - Creating a FastMCP server with tools
  - Connecting via MCPClient (in-memory)
  - Wrapping MCP tools as LangChain tools via MCPToolAdapter
  - Running an LlmAgent with MCP tools

To use with a remote MCP server instead:
    client = MCPClient("http://localhost:8001/mcp")
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from langchain_adk import LlmAgent
from langchain_adk.events.event import Event, EventType
from langchain_adk.integrations.mcp import MCPClient, MCPToolAdapter


# --- Define an MCP server with tools ---
mcp_server = FastMCP("WeatherServer")


@mcp_server.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 22°C."


@mcp_server.tool
def get_forecast(city: str, days: int = 3) -> str:
    """Get a weather forecast for a city."""
    return f"{days}-day forecast for {city}: sunny, cloudy, rainy."


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # Connect to MCP server (in-memory for this example)
    client = MCPClient(mcp_server)
    adapter = MCPToolAdapter(client)
    mcp_tools = await adapter.load_tools()

    print(f"Loaded {len(mcp_tools)} MCP tools: {[t.name for t in mcp_tools]}")

    agent = LlmAgent(
        name="WeatherMCPAgent",
        llm=llm,  # noqa: F821
        tools=mcp_tools,
        instructions="You are a weather assistant. Use the available tools.",
    )

    print(f"\nRunning agent: {agent.name}\n{'=' * 40}")

    async for event in agent.astream(
        "What's the weather in Copenhagen and give me a 5-day forecast?",
    ):
        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text or event.error}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
