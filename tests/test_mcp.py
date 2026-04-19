"""Tests for MCP integration — client, adapter, and tool wrapping.

Uses fastMCP's in-memory transport (no HTTP, no subprocess).
"""

from __future__ import annotations

import pytest

fastmcp = pytest.importorskip(
    "fastmcp", reason="fastmcp not installed (install with: pip install orxhestra[mcp])"
)
FastMCP = fastmcp.FastMCP

from orxhestra.integrations.mcp.adapter import MCPToolAdapter  # noqa: E402
from orxhestra.integrations.mcp.client import MCPClient  # noqa: E402

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


# ---------------------------------------------------------------------------
# Session-mode tests
# ---------------------------------------------------------------------------


async def test_session_mode_reuses_connection(mcp_server: FastMCP) -> None:
    """``async with client:`` opens one session shared by all calls."""
    client = MCPClient(mcp_server)
    async with client:
        # Multiple calls inside the `with` block share the open session.
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": 1, "b": 2})
    assert {t.name for t in tools} == {"add", "greet"}
    assert "3" in str(result)


async def test_session_mode_nested_entries(mcp_server: FastMCP) -> None:
    """Nested ``async with`` blocks share the outermost session."""
    client = MCPClient(mcp_server)
    async with client:
        async with client:
            result = await client.call_tool("greet", {"name": "Nested"})
        # After inner exit the outer session must still be usable.
        tools = await client.list_tools()
    assert "Nested" in str(result)
    assert {t.name for t in tools} == {"add", "greet"}


async def test_ping(mcp_server: FastMCP) -> None:
    client = MCPClient(mcp_server)
    assert await client.ping() is True


# ---------------------------------------------------------------------------
# Resources + resource templates
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_server_with_resources() -> FastMCP:
    server = FastMCP("ResourceServer")

    @server.resource("config://app")
    def app_config() -> str:
        return "app-config-contents"

    @server.resource("file://{name}.txt")
    def file_template(name: str) -> str:
        return f"contents of {name}.txt"

    return server


async def test_list_resources(mcp_server_with_resources: FastMCP) -> None:
    client = MCPClient(mcp_server_with_resources)
    resources = await client.list_resources()
    uris = {str(r.uri) for r in resources}
    assert "config://app" in uris


async def test_list_resource_templates(mcp_server_with_resources: FastMCP) -> None:
    client = MCPClient(mcp_server_with_resources)
    templates = await client.list_resource_templates()
    template_uris = {t.uriTemplate for t in templates}
    assert any("{name}" in uri for uri in template_uris)


async def test_read_resource(mcp_server_with_resources: FastMCP) -> None:
    client = MCPClient(mcp_server_with_resources)
    result = await client.read_resource("config://app")
    assert "app-config-contents" in str(result)


# ---------------------------------------------------------------------------
# Prompts + MCPPromptAdapter
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_server_with_prompts() -> FastMCP:
    server = FastMCP("PromptServer")

    @server.prompt
    def greet_prompt(name: str) -> str:
        """Produce a friendly greeting."""
        return f"Say hello to {name}!"

    @server.prompt
    def review_code(language: str = "python") -> str:
        """Code-review system prompt, optionally scoped to a language."""
        return f"Review this {language} code carefully."

    return server


async def test_list_prompts_raw(mcp_server_with_prompts: FastMCP) -> None:
    client = MCPClient(mcp_server_with_prompts)
    prompts = await client.list_prompts()
    names = {p.name for p in prompts}
    assert names == {"greet_prompt", "review_code"}


async def test_get_prompt_raw(mcp_server_with_prompts: FastMCP) -> None:
    client = MCPClient(mcp_server_with_prompts)
    result = await client.get_prompt("greet_prompt", {"name": "Alice"})
    # GetPromptResult has a messages list; first message should include "Alice".
    assert any("Alice" in str(getattr(m, "content", "")) for m in result.messages)


async def test_prompt_adapter_list(mcp_server_with_prompts: FastMCP) -> None:
    from orxhestra.integrations.mcp import MCPPromptAdapter, PromptDescriptor

    adapter = MCPPromptAdapter(MCPClient(mcp_server_with_prompts))
    prompts = await adapter.list_prompts()
    assert all(isinstance(p, PromptDescriptor) for p in prompts)
    by_name = {p.name: p for p in prompts}
    assert by_name["greet_prompt"].arguments[0].name == "name"
    assert by_name["greet_prompt"].arguments[0].required is True


async def test_prompt_adapter_get_messages(mcp_server_with_prompts: FastMCP) -> None:
    from langchain_core.messages import BaseMessage

    from orxhestra.integrations.mcp import MCPPromptAdapter

    adapter = MCPPromptAdapter(MCPClient(mcp_server_with_prompts))
    messages = await adapter.get_messages("greet_prompt", {"name": "Zoe"})
    assert messages
    assert all(isinstance(m, BaseMessage) for m in messages)
    assert any("Zoe" in str(m.content) for m in messages)


async def test_prompt_adapter_get_text(mcp_server_with_prompts: FastMCP) -> None:
    from orxhestra.integrations.mcp import MCPPromptAdapter

    adapter = MCPPromptAdapter(MCPClient(mcp_server_with_prompts))
    text = await adapter.get_text("review_code", {"language": "rust"})
    assert "rust" in text.lower()


async def test_prompt_adapter_load_as_tools(mcp_server_with_prompts: FastMCP) -> None:
    from orxhestra.integrations.mcp import MCPPromptAdapter

    adapter = MCPPromptAdapter(MCPClient(mcp_server_with_prompts))
    tools = await adapter.load_as_tools()
    by_name = {t.name: t for t in tools}
    assert "prompt_greet_prompt" in by_name
    assert "prompt_review_code" in by_name

    result = await by_name["prompt_greet_prompt"].ainvoke({"name": "Max"})
    assert "Max" in str(result)


# ---------------------------------------------------------------------------
# Sampling + logging + progress handler helpers
# ---------------------------------------------------------------------------


class _StubLLM:
    """Tiny stand-in for a LangChain chat model."""

    def __init__(self, reply: str = "stubbed reply") -> None:
        self._reply = reply
        self.calls: list[tuple[list, dict]] = []
        self.model_name = "stub-llm"

    async def ainvoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage

        self.calls.append((messages, kwargs))
        return AIMessage(
            content=self._reply,
            response_metadata={"model_name": self.model_name},
        )


async def test_make_langchain_sampling_handler_round_trip() -> None:
    import mcp.types as mcp_types

    from orxhestra.integrations.mcp import make_langchain_sampling_handler

    llm = _StubLLM(reply="hello from llm")
    handler = make_langchain_sampling_handler(llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            ),
        ],
        systemPrompt="be concise",
        maxTokens=64,
        temperature=0.2,
    )
    result = await handler(context=None, params=params)

    assert result.role == "assistant"
    assert result.content.text == "hello from llm"
    assert result.model == "stub-llm"
    # The LLM saw a system + user message and the max_tokens kwarg.
    sent_messages, kwargs = llm.calls[0]
    assert len(sent_messages) == 2  # system + user
    assert kwargs.get("max_tokens") == 64


async def test_make_python_logging_handler_forwards(caplog) -> None:
    import logging

    import mcp.types as mcp_types

    from orxhestra.integrations.mcp import make_python_logging_handler

    handler = make_python_logging_handler("test.mcp.logger")
    with caplog.at_level(logging.WARNING, logger="test.mcp.logger"):
        await handler(
            mcp_types.LoggingMessageNotificationParams(
                level="warning",
                logger="remote-server",
                data="something off",
            ),
        )
    assert any("something off" in rec.message for rec in caplog.records)


async def test_make_stream_progress_handler_sync_sink() -> None:
    from orxhestra.integrations.mcp import make_stream_progress_handler

    received: list[tuple[float, float | None, str | None]] = []

    def sink(progress, total, message):
        received.append((progress, total, message))

    handler = make_stream_progress_handler(sink)
    await handler(0.5, 1.0, "halfway")
    await handler(1.0, 1.0, "done")
    assert received == [(0.5, 1.0, "halfway"), (1.0, 1.0, "done")]


async def test_make_stream_progress_handler_async_sink() -> None:
    from orxhestra.integrations.mcp import make_stream_progress_handler

    received: list[tuple[float, float | None, str | None]] = []

    async def sink(progress, total, message):
        received.append((progress, total, message))

    handler = make_stream_progress_handler(sink)
    await handler(0.25, 1.0, "quarter")
    assert received == [(0.25, 1.0, "quarter")]


# ---------------------------------------------------------------------------
# Roots normalisation
# ---------------------------------------------------------------------------


def test_client_normalises_roots(tmp_path) -> None:
    """Path-like roots convert to file:// URIs; existing URIs pass through."""
    client = MCPClient(
        "http://localhost:8001/mcp",
        roots=[tmp_path, str(tmp_path), "http://example.com/data"],
    )
    assert client._roots is not None
    assert all(root.startswith(("file://", "http://")) for root in client._roots)
    # One http:// URI was preserved, two converted paths are file://.
    assert any(root == "http://example.com/data" for root in client._roots)
