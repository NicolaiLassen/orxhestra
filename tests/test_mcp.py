"""Tests for MCP integration — client, adapter, and tool wrapping.

Uses fastMCP's in-memory transport (no HTTP, no subprocess).
"""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from langchain_adk.integrations.mcp.client import MCPClient
from langchain_adk.integrations.mcp.adapter import MCPToolAdapter


# ---------------------------------------------------------------------------
# Fixture: a simple MCP server with two tools
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_server() -> FastMCP:
    server = FastMCP("TestServer")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    return server


@pytest.fixture
def mcp_client(mcp_server: FastMCP) -> MCPClient:
    return MCPClient(mcp_server)


# ---------------------------------------------------------------------------
# MCPClient tests
# ---------------------------------------------------------------------------


async def test_client_list_tools(mcp_client: MCPClient):
    tools = await mcp_client.list_tools()
    names = {t.name for t in tools}
    assert "add" in names
    assert "greet" in names


async def test_client_call_tool_add(mcp_client: MCPClient):
    result = await mcp_client.call_tool("add", {"a": 3, "b": 4})
    # fastMCP returns CallToolResult with .content list
    assert "7" in str(result)


async def test_client_call_tool_greet(mcp_client: MCPClient):
    result = await mcp_client.call_tool("greet", {"name": "Alice"})
    assert "Hello, Alice!" in str(result)


async def test_client_url_property(mcp_server: FastMCP):
    # In-memory client has no URL
    client = MCPClient(mcp_server)
    assert client.url is None

    # HTTP client has URL
    client_http = MCPClient("http://localhost:8001/mcp")
    assert client_http.url == "http://localhost:8001/mcp"


# ---------------------------------------------------------------------------
# MCPToolAdapter tests
# ---------------------------------------------------------------------------


async def test_adapter_load_tools(mcp_client: MCPClient):
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.load_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"add", "greet"}


async def test_adapter_tool_has_description(mcp_client: MCPClient):
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.load_tools()
    tool_map = {t.name: t for t in tools}
    assert "Add two numbers" in tool_map["add"].description
    assert "Greet someone" in tool_map["greet"].description


async def test_adapter_tool_has_args_schema(mcp_client: MCPClient):
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.load_tools()
    tool_map = {t.name: t for t in tools}

    add_schema = tool_map["add"].args_schema
    assert "a" in add_schema.model_fields
    assert "b" in add_schema.model_fields


async def test_adapter_tool_call_add(mcp_client: MCPClient):
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.load_tools()
    tool_map = {t.name: t for t in tools}

    result = await tool_map["add"].ainvoke({"a": 10, "b": 20})
    assert "30" in str(result)


async def test_adapter_tool_call_greet(mcp_client: MCPClient):
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.load_tools()
    tool_map = {t.name: t for t in tools}

    result = await tool_map["greet"].ainvoke({"name": "Bob"})
    assert "Hello, Bob!" in str(result)


async def test_adapter_sync_raises(mcp_client: MCPClient):
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.load_tools()

    with pytest.raises(NotImplementedError):
        tools[0]._run(a=1, b=2)
