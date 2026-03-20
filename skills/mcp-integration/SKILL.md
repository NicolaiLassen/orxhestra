---
name: mcp-integration
description: Connect langchain-adk agents to MCP tool servers using MCPClient and MCPToolAdapter.
---

# MCP Integration

Connect to any MCP-compatible tool server.

```bash
pip install langchain-adk[mcp]
```

## Usage

```python
from langchain_adk.integrations.mcp import MCPClient, MCPToolAdapter

client = MCPClient("http://localhost:8001/mcp")
adapter = MCPToolAdapter(client)
mcp_tools = await adapter.load_tools()

agent = LlmAgent(
    name="MCPAgent",
    llm=llm,
    tools=mcp_tools,
    instructions="Use the available tools to answer questions.",
)
```

## Testing with in-memory server

Pass a FastMCP server object directly (no HTTP needed):

```python
from fastmcp import FastMCP

server = FastMCP("TestServer")

@server.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

client = MCPClient(server)  # in-memory, no network
adapter = MCPToolAdapter(client)
tools = await adapter.load_tools()
```

`MCPToolAdapter.load_tools()` fetches the tool list from the MCP server and wraps each as a LangChain `BaseTool`.

## In YAML Composer

```yaml
tools:
  weather:
    mcp:
      url: "http://localhost:8001/mcp"
  local:
    mcp:
      server: "myapp.mcp_server.server"  # dotted import to FastMCP instance
```
