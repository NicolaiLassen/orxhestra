---
name: agent-tools
description: Create and use tools with orxhestra agents. Covers function_tool, AgentTool, transfer tools, exit_loop, MCP tools, and CallContext.
---

# Agent Tools

## function_tool — Wrap Python functions

```python
from orxhestra.tools import function_tool

@function_tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Or with custom name
@function_tool(name="web_search", description="Search the internet")
async def search(query: str) -> str:
    return f"Results for: {query}"
```

## AgentTool — Agent as a callable tool

Make any agent callable as a tool by a parent agent.

```python
from orxhestra.tools import AgentTool

researcher = LlmAgent(name="researcher", llm=llm, description="Research topics.", instructions="...")

# Parent agent can call researcher as a tool
manager = LlmAgent(
    name="manager",
    llm=llm,
    tools=[AgentTool(agent=researcher)],
    instructions="Use the researcher tool when you need information.",
)
```

## make_transfer_tool — Agent handoff

```python
from orxhestra.tools import make_transfer_tool

transfer = make_transfer_tool([sales_agent, support_agent])
triage = LlmAgent(name="triage", llm=llm, tools=[transfer], instructions="Route requests.")
```

## exit_loop_tool — Break out of LoopAgent

```python
from orxhestra.tools import exit_loop_tool

reviewer = LlmAgent(
    name="reviewer",
    llm=llm,
    tools=[exit_loop_tool],
    instructions="Call exit_loop when the draft is approved.",
)
```

## CallContext — Access state inside tools

```python
from orxhestra.tools import CallContext

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something"

    async def _arun(self, input: str, **kwargs) -> str:
        ctx: CallContext = kwargs.get("tool_context")
        if ctx:
            ctx.state["last_query"] = input  # write to shared state
            session_id = ctx.session_id
        return f"Processed: {input}"
```

## MCPToolAdapter — Connect MCP servers

```python
from orxhestra.integrations.mcp import MCPClient, MCPToolAdapter

# HTTP MCP server
client = MCPClient("http://localhost:8001/mcp")
adapter = MCPToolAdapter(client)
tools = await adapter.load_tools()

agent = LlmAgent(name="agent", llm=llm, tools=tools)

# In-memory FastMCP server
from mcp_server import mcp  # FastMCP instance
client = MCPClient(mcp)
tools = await MCPToolAdapter(client).load_tools()
```

## LongRunningFunctionTool — Long operations

```python
from orxhestra.tools import LongRunningFunctionTool

async def deploy(env: str) -> str:
    """Deploy to environment."""
    await run_deploy(env)
    return f"Deployed to {env}"

tool = LongRunningFunctionTool(func=deploy)
```
