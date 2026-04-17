"""ToolSearchTool — lazy-load deferred tool schemas on demand.

Allows agents to discover and load tools at runtime without all tools
being bound to the LLM upfront. Useful when the total tool count is
large and you want the agent to search for relevant tools.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool


def make_tool_search_tool(
    available_tools: dict[str, BaseTool],
) -> BaseTool:
    """Create a tool that searches and describes available tools.

    Parameters
    ----------
    available_tools : dict[str, BaseTool]
        Registry of all tools the agent could access, keyed by name.

    Returns
    -------
    BaseTool
        A ``tool_search`` structured tool.

    See Also
    --------
    ToolRegistry : Shared registry typically used as the source of
        ``available_tools``.
    """

    async def tool_search(query: str, max_results: int = 5) -> str:
        """Search for available tools by name or description.

        Args:
            query: Keyword to search for in tool names and descriptions.
            max_results: Maximum number of results to return (default 5).
        """
        query_lower = query.lower()
        matches: list[tuple[str, str, str]] = []

        for name, tool in available_tools.items():
            desc = tool.description or ""
            # Score: name match > description match.
            if query_lower in name.lower():
                matches.append((name, desc, "name"))
            elif query_lower in desc.lower():
                matches.append((name, desc, "desc"))

        # Name matches first, then description matches.
        matches.sort(key=lambda m: (m[2] != "name", m[0]))
        matches = matches[:max_results]

        if not matches:
            return f"No tools matching '{query}'. Available: {', '.join(sorted(available_tools))}"

        lines: list[str] = []
        for name, desc, _ in matches:
            tool = available_tools[name]
            schema = ""
            if hasattr(tool, "args_schema") and tool.args_schema:
                try:
                    fields = tool.args_schema.model_fields
                    params = []
                    for k, v in fields.items():
                        ann = v.annotation
                        type_name = getattr(ann, "__name__", str(ann))
                        params.append(f"{k}: {type_name}")
                    schema = f"({', '.join(params)})"
                except Exception:
                    schema = ""
            short_desc = desc[:120] + "..." if len(desc) > 120 else desc
            lines.append(f"  {name}{schema}\n    {short_desc}")

        return f"Found {len(matches)} tool(s):\n" + "\n".join(lines)

    return StructuredTool.from_function(
        coroutine=tool_search,
        name="tool_search",
        description=(
            "Search for available tools by keyword. Returns tool names, "
            "descriptions, and parameter schemas. Use when you need to "
            "find the right tool for a task."
        ),
    )
