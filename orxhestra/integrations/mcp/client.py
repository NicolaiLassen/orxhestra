"""MCP client - thin wrapper around FastMCP.

Supports both HTTP URLs and in-memory FastMCP server objects.
Uses a fresh session per call. Suitable for request-scoped usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Client


class MCPClient:
    """Async wrapper around the FastMCP Client.

    Parameters
    ----------
    transport : str or FastMCP server
        Either an HTTP URL (e.g. "http://localhost:8080/mcp")
        or a FastMCP server object for in-memory usage.
    """

    def __init__(self, transport: str | Any) -> None:
        self._transport = transport

    @property
    def url(self) -> str | None:
        """Return the URL if transport is a string, else None."""
        return self._transport if isinstance(self._transport, str) else None

    def _make_client(self) -> Client:
        from fastmcp import Client as _Client

        if isinstance(self._transport, str):
            from fastmcp.client.transports import StreamableHttpTransport
            return _Client(StreamableHttpTransport(self._transport))
        else:
            # In-memory: pass server object directly
            return _Client(self._transport)

    async def list_tools(self) -> list[Any]:
        """Return the list of tools exposed by the MCP server.

        Returns
        -------
        list[Any]
            FastMCP tool descriptors (name, description, input schema).
        """
        async with self._make_client() as client:
            return await client.list_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a tool by name with the given arguments.

        Parameters
        ----------
        name : str
            MCP tool name to invoke.
        arguments : dict[str, Any]
            Keyword arguments matching the tool's input schema.

        Returns
        -------
        Any
            The ``CallToolResult`` from FastMCP; typically an object
            with a ``.content`` list of text/data items.
        """
        async with self._make_client() as client:
            return await client.call_tool(name, arguments)

    async def list_resources(self) -> list[Any]:
        """Return the list of resources exposed by the MCP server.

        Returns
        -------
        list[Any]
            FastMCP resource descriptors (URI, MIME type, metadata).
        """
        async with self._make_client() as client:
            return await client.list_resources()

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI.

        Parameters
        ----------
        uri : str
            Resource URI as advertised by :meth:`list_resources`.

        Returns
        -------
        Any
            The resource contents as returned by FastMCP.
        """
        async with self._make_client() as client:
            return await client.read_resource(uri)
