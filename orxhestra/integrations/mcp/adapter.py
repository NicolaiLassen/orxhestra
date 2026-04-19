"""MCPToolAdapter — convert MCP tools into LangChain BaseTool instances.

At runtime, fetches the tool list from an MCP server via
:class:`~orxhestra.integrations.mcp.client.MCPClient` and wraps each
tool as a LangChain
:class:`~langchain_core.tools.BaseTool`.  The Pydantic input schema
is generated dynamically from the MCP tool's JSON Schema so standard
LangChain tool-call plumbing (validation, OpenAPI-style typing) just
works on top.

See Also
--------
orxhestra.integrations.mcp.client.MCPClient : FastMCP client used
    under the hood.
orxhestra.tools.function_tool : Pure-Python alternative when a
    remote server is not needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from orxhestra.integrations.mcp.client import MCPClient

if TYPE_CHECKING:
    import datetime

    from fastmcp.client.progress import ProgressHandler


def _build_input_model(tool_name: str, json_schema: dict[str, Any]) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema properties dict.

    Parameters
    ----------
    tool_name : str
        Used to name the generated model class.
    json_schema : dict[str, Any]
        A JSON Schema object with ``properties`` and ``required`` keys.

    Returns
    -------
    Type[BaseModel]
        A dynamically created Pydantic model class.
    """
    properties: dict[str, Any] = json_schema.get("properties", {})
    required: list[str] = json_schema.get("required", [])

    fields: dict[str, Any] = {}
    for field_name, field_schema in properties.items():
        field_type = _json_type_to_python(field_schema.get("type", "string"))
        description = field_schema.get("description", "")
        if field_name in required:
            from pydantic import Field
            fields[field_name] = (field_type, Field(description=description))
        else:
            from pydantic import Field
            fields[field_name] = (
                field_type | None,
                Field(default=None, description=description),
            )

    return create_model(f"{tool_name}Input", **fields)


def _json_type_to_python(json_type: str) -> type:
    """Map a JSON Schema type string to a Python type.

    Parameters
    ----------
    json_type : str
        A JSON Schema primitive type name.

    Returns
    -------
    type
        The corresponding Python type, defaulting to str.
    """
    mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, str)


class MCPToolAdapter:
    """Converts an MCP server's tool list into LangChain BaseTool instances.

    Parameters
    ----------
    client : MCPClient
        Source of the tool catalogue.
    progress_handler : ProgressHandler, optional
        Default progress handler applied to every ``call_tool`` call
        made by the wrapped tools.  Overridden per-call by the MCP
        client's own progress handler when set.
    timeout : float or timedelta, optional
        Default per-call timeout applied to every wrapped tool.
    raise_on_error : bool
        When ``True`` (default), MCP tool errors surface as
        exceptions inside ``ainvoke``; when ``False`` the raw
        ``CallToolResult`` is returned so the agent can inspect
        ``isError``.

    Examples
    --------
    >>> client = MCPClient("http://localhost:8001/mcp")
    >>> adapter = MCPToolAdapter(client)
    >>> tools = await adapter.load_tools()
    >>> agent = LlmAgent("agent", model=model, tools=tools)
    """

    def __init__(
        self,
        client: MCPClient,
        *,
        progress_handler: ProgressHandler | None = None,
        timeout: float | datetime.timedelta | None = None,
        raise_on_error: bool = True,
    ) -> None:
        self._client = client
        self._progress_handler = progress_handler
        self._timeout = timeout
        self._raise_on_error = raise_on_error

    async def load_tools(self) -> list[BaseTool]:
        """Fetch tools from the MCP server and wrap them as LangChain tools.

        Returns
        -------
        list[BaseTool]
            One LangChain BaseTool per tool exposed by the MCP server.
        """
        mcp_tools = await self._client.list_tools()
        return [self._wrap(tool) for tool in mcp_tools]

    def _wrap(self, mcp_tool: Any) -> BaseTool:
        """Wrap a single MCP Tool as a LangChain BaseTool.

        Parameters
        ----------
        mcp_tool : Any
            A tool definition object returned by the MCP server.

        Returns
        -------
        BaseTool
            A LangChain tool that proxies calls to the MCP server.
        """
        client = self._client
        tool_name: str = mcp_tool.name
        tool_description: str = mcp_tool.description or ""
        input_schema: dict = (
            mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}
        )
        input_model = _build_input_model(tool_name, input_schema)
        progress_handler = self._progress_handler
        timeout = self._timeout
        raise_on_error = self._raise_on_error

        class WrappedMCPTool(BaseTool):
            """LangChain tool that proxies calls to a remote MCP tool.

            The tool's name, description, and input schema are derived
            from the MCP tool descriptor. This class is async-only —
            ``_run`` raises ``NotImplementedError`` because MCP calls
            are always awaited.
            """

            name: str = tool_name
            description: str = tool_description
            args_schema: type[BaseModel] = input_model

            def _run(self, **kwargs: Any) -> Any:
                """Raise an error — WrappedMCPTool is async-only."""
                raise NotImplementedError("Use async ainvoke.")

            async def _arun(self, **kwargs: Any) -> Any:
                """Call the MCP tool and return its text content."""
                result = await client.call_tool(
                    tool_name,
                    kwargs,
                    progress_handler=progress_handler,
                    timeout=timeout,
                    raise_on_error=raise_on_error,
                )
                # Extract text from MCP result
                content = result
                # CallToolResult has a .content list
                if hasattr(result, "content"):
                    content = result.content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if hasattr(item, "text"):
                            parts.append(item.text)
                        elif isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        else:
                            parts.append(str(item))
                    return "\n".join(parts)
                return str(result)

        return WrappedMCPTool()
