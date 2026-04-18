"""Model Context Protocol (MCP) integration.

Full coverage of the MCP primitives plus the bidirectional session
surface — everything FastMCP's client exposes, wrapped in
orxhestra-shaped types:

- :class:`MCPClient` — async client wrapping
  :class:`fastmcp.Client`.  Supports stateless per-call usage and
  ``async with client: ...`` persistent sessions for bidirectional
  callbacks.
- :class:`MCPToolAdapter` — converts the server's tool catalogue
  into LangChain :class:`~langchain_core.tools.BaseTool` instances
  (with optional per-call progress / timeout / error-handling
  knobs).
- :class:`MCPPromptAdapter` — surfaces server-published prompt
  templates as LangChain messages, tools, or plain text.
- :class:`PromptDescriptor`, :class:`PromptArgumentSpec` —
  lightweight descriptors returned by
  :meth:`MCPPromptAdapter.list_prompts`.
- :func:`make_langchain_sampling_handler` — plug a LangChain chat
  model in as the MCP sampling callback so server-side tools that
  request ``ctx.sample(...)`` delegate LLM calls back to us.
- :func:`make_python_logging_handler` — forward MCP server logs to
  a Python :class:`logging.Logger`.
- :func:`make_stream_progress_handler` — adapt a plain async sink
  into an MCP progress handler.

Requires the ``mcp`` extra::

    pip install 'orxhestra[mcp]'

Examples
--------
Tools:

>>> from orxhestra.integrations.mcp import MCPClient, MCPToolAdapter
>>> client = MCPClient("http://localhost:8001/mcp")
>>> tools = await MCPToolAdapter(client).load_tools()
>>> agent = LlmAgent("agent", model=model, tools=tools)

Prompts:

>>> from orxhestra.integrations.mcp import MCPPromptAdapter
>>> prompts = await MCPPromptAdapter(client).list_prompts()
>>> messages = await MCPPromptAdapter(client).get_messages(
...     "summarize", {"topic": "graph theory"},
... )

Bidirectional (sampling + logging):

>>> from orxhestra.integrations.mcp import (
...     make_langchain_sampling_handler, make_python_logging_handler,
... )
>>> client = MCPClient(
...     "http://localhost:8001/mcp",
...     sampling_handler=make_langchain_sampling_handler(llm),
...     log_handler=make_python_logging_handler("my.app.mcp"),
... )
>>> async with client:
...     tools = await MCPToolAdapter(client).load_tools()
...     # ... run agent ...
"""

from orxhestra.integrations.mcp.adapter import MCPToolAdapter
from orxhestra.integrations.mcp.client import MCPClient
from orxhestra.integrations.mcp.prompts import (
    MCPPromptAdapter,
    PromptArgumentSpec,
    PromptDescriptor,
)
from orxhestra.integrations.mcp.sampling import (
    make_langchain_sampling_handler,
    make_python_logging_handler,
    make_stream_progress_handler,
)

__all__ = [
    "MCPClient",
    "MCPPromptAdapter",
    "MCPToolAdapter",
    "PromptArgumentSpec",
    "PromptDescriptor",
    "make_langchain_sampling_handler",
    "make_python_logging_handler",
    "make_stream_progress_handler",
]
