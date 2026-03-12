"""MCP client - thin wrapper around FastMCP.

Supports both HTTP URLs and in-memory FastMCP server objects.
Uses a fresh session per call. Suitable for request-scoped usage.
"""

from __future__ import annotations

from typing import Any, Union


class MCPClient:
    """Async wrapper around the FastMCP Client.

    Parameters
    ----------
    transport : str or FastMCP server
        Either an HTTP URL (e.g. "http://localhost:8001/mcp")
        or a FastMCP server object for in-memory usage.
    """

    def __init__(self, transport: Union[str, Any]) -> None:
        self._transport = transport

    @property
    def url(self) -> str | None:
        """Return the URL if transport is a string, else None."""
        return self._transport if isinstance(self._transport, str) else None

    def _make_client(self) -> Any:
        from fastmcp import Client

        if isinstance(self._transport, str):
            from fastmcp.client.transports import StreamableHttpTransport
            return Client(StreamableHttpTransport(self._transport))
        else:
            # In-memory: pass server object directly
            return Client(self._transport)

    async def list_tools(self) -> list[Any]:
        """Return the list of tools exposed by the MCP server."""
        async with self._make_client() as client:
            return await client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name with the given arguments."""
        async with self._make_client() as client:
            return await client.call_tool(name, arguments)
