"""MCPToolAdapter - convert MCP tools into LangChain BaseTool instances.

At runtime, fetch the tool list from an MCP server and wrap each tool
as a LangChain BaseTool. The Pydantic input schema is built from the
MCP tool's JSON Schema definition.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from langchain_adk.integrations.mcp.client import MCPClient


def _build_input_model(tool_name: str, json_schema: dict[str, Any]) -> Type[BaseModel]:
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
            fields[field_name] = (Optional[field_type], Field(default=None, description=description))

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

    Examples
    --------
    >>> client = MCPClient("http://localhost:8001/mcp")
    >>> adapter = MCPToolAdapter(client)
    >>> tools = await adapter.load_tools()
    >>> agent = LlmAgent("agent", llm=llm, tools=tools)
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

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

        class WrappedMCPTool(BaseTool):
            name: str = tool_name
            description: str = tool_description
            args_schema: Type[BaseModel] = input_model

            def _run(self, **kwargs: Any) -> Any:
                """Raise an error - WrappedMCPTool is async-only."""
                raise NotImplementedError("Use async ainvoke.")

            async def _arun(self, **kwargs: Any) -> Any:
                """Call the MCP tool and return its text content."""
                result = await client.call_tool(tool_name, kwargs)
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
