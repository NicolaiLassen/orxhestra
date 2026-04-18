"""MCP prompt adapter — pull server-published prompts into agent context.

MCP's third primitive (alongside tools and resources) is *prompts*:
named, argument-templated conversations the server publishes for the
client to inject into its own LLM context.  Typical uses: "use the
/summarize prompt with topic=X", "load the server's canonical
system message for code review", etc.

This module turns the prompt catalogue into LangChain-shaped
primitives:

- :meth:`MCPPromptAdapter.list_prompts` returns lightweight
  :class:`PromptDescriptor` records (name + description + argument
  specs) so agents can enumerate what's available.
- :meth:`MCPPromptAdapter.get_messages` renders a prompt into a
  ``list[BaseMessage]`` ready to splice into an :class:`LlmAgent`'s
  message history.
- :meth:`MCPPromptAdapter.load_as_tools` exposes every prompt as a
  LangChain :class:`~langchain_core.tools.BaseTool` that returns the
  rendered prompt text — useful when you want the LLM itself to
  decide which server prompt to pull in.

See Also
--------
MCPClient : Underlying transport.
MCPToolAdapter : Sibling adapter for MCP tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from orxhestra.integrations.mcp.client import MCPClient


@dataclass(frozen=True)
class PromptArgumentSpec:
    """Spec for a single prompt argument.

    Attributes
    ----------
    name : str
        Argument name.
    description : str
        Human-readable description from the server.
    required : bool
        Whether the server marked this argument as required.
    """

    name: str
    description: str = ""
    required: bool = False


@dataclass(frozen=True)
class PromptDescriptor:
    """Lightweight view of an MCP prompt for UI / discovery.

    Attributes
    ----------
    name : str
        Prompt name — the key passed to :meth:`MCPPromptAdapter.get_messages`.
    description : str
        Short description from the server.
    arguments : list[PromptArgumentSpec]
        Declared arguments and whether each is required.
    """

    name: str
    description: str = ""
    arguments: list[PromptArgumentSpec] = field(default_factory=list)


class MCPPromptAdapter:
    """Adapter that surfaces MCP prompts as LangChain messages / tools.

    Parameters
    ----------
    client : MCPClient
        Source of the prompt catalogue.

    Examples
    --------
    Enumerate:

    >>> adapter = MCPPromptAdapter(client)
    >>> for prompt in await adapter.list_prompts():
    ...     print(prompt.name, prompt.description)

    Render into LangChain messages for manual injection:

    >>> messages = await adapter.get_messages(
    ...     "summarize", {"topic": "graph theory"},
    ... )
    >>> agent = LlmAgent("s", model=model, instructions=messages[0].content)

    Expose as tools the LLM can choose to call:

    >>> prompt_tools = await adapter.load_as_tools()
    >>> agent = LlmAgent("s", model=model, tools=prompt_tools)

    See Also
    --------
    MCPClient : Transport-layer primitive.
    MCPToolAdapter : Tool-side counterpart.
    """

    #: Prefix applied to prompt-backed tool names in :meth:`load_as_tools`.
    TOOL_NAME_PREFIX: str = "prompt_"

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def list_prompts(self) -> list[PromptDescriptor]:
        """Return descriptors for every prompt the server publishes.

        Returns
        -------
        list[PromptDescriptor]
        """
        raw_prompts = await self._client.list_prompts()
        return [self._describe(p) for p in raw_prompts]

    async def get_messages(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[BaseMessage]:
        """Fetch and render a prompt into LangChain messages.

        Parameters
        ----------
        name : str
            Prompt name from :meth:`list_prompts`.
        arguments : dict, optional
            Values for the prompt's templated arguments.

        Returns
        -------
        list[BaseMessage]
            LangChain-typed messages ready to splice into an LLM
            conversation.  Assistant-role MCP messages become
            :class:`AIMessage`; every other role (including
            ``system`` and ``user``) becomes :class:`HumanMessage`
            so orxhestra's LLM layer treats it as input.
        """
        result = await self._client.get_prompt(name, arguments)
        return [
            _mcp_message_to_langchain(msg) for msg in getattr(result, "messages", [])
        ]

    async def get_text(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        separator: str = "\n\n",
    ) -> str:
        """Flatten a prompt into plain text.

        Convenience for callers that want to concatenate every
        message into a single string — e.g. to drop into an agent's
        ``instructions`` or expose the rendered prompt as a tool
        response.

        Parameters
        ----------
        name : str
        arguments : dict, optional
        separator : str
            Joined between message contents.

        Returns
        -------
        str
        """
        messages = await self.get_messages(name, arguments)
        parts = [m.content if isinstance(m.content, str) else str(m.content) for m in messages]
        return separator.join(p for p in parts if p)

    async def load_as_tools(self) -> list[BaseTool]:
        """Expose every prompt as a LangChain tool.

        Each returned tool, when called, fetches the prompt with the
        supplied arguments and returns the rendered text.  The tool's
        input schema is derived from the prompt's declared arguments
        so the LLM gets proper per-field validation.

        Returns
        -------
        list[BaseTool]
            One tool per prompt; names are prefixed with
            :attr:`TOOL_NAME_PREFIX` so they don't collide with real
            MCP tools.
        """
        prompts = await self.list_prompts()
        return [self._wrap_as_tool(p) for p in prompts]

    # ── Internals ────────────────────────────────────────────────

    def _describe(self, prompt: Any) -> PromptDescriptor:
        """Coerce a FastMCP prompt descriptor into a :class:`PromptDescriptor`.

        Parameters
        ----------
        prompt : Any
            ``mcp.types.Prompt`` or a duck-typed equivalent.

        Returns
        -------
        PromptDescriptor
        """
        args: list[PromptArgumentSpec] = []
        for raw_arg in getattr(prompt, "arguments", None) or []:
            args.append(
                PromptArgumentSpec(
                    name=getattr(raw_arg, "name", ""),
                    description=getattr(raw_arg, "description", "") or "",
                    required=bool(getattr(raw_arg, "required", False)),
                ),
            )
        return PromptDescriptor(
            name=prompt.name,
            description=getattr(prompt, "description", "") or "",
            arguments=args,
        )

    def _wrap_as_tool(self, descriptor: PromptDescriptor) -> BaseTool:
        """Wrap a single prompt as a LangChain tool.

        Parameters
        ----------
        descriptor : PromptDescriptor

        Returns
        -------
        BaseTool
            Async tool that fetches + renders the prompt on call.
        """
        adapter = self
        tool_name = f"{self.TOOL_NAME_PREFIX}{descriptor.name}"
        input_model = _prompt_args_to_model(descriptor)

        class WrappedMCPPrompt(BaseTool):
            """LangChain tool that fetches an MCP prompt on call."""

            name: str = tool_name
            description: str = (
                descriptor.description
                or f"Fetch the MCP prompt '{descriptor.name}' and return its text."
            )
            args_schema: type[BaseModel] = input_model

            def _run(self, **kwargs: Any) -> Any:
                """Raise — MCP calls are always awaited."""
                raise NotImplementedError("Use async ainvoke.")

            async def _arun(self, **kwargs: Any) -> str:
                """Fetch the prompt and return the rendered text."""
                clean = {k: v for k, v in kwargs.items() if v is not None}
                return await adapter.get_text(descriptor.name, clean)

        return WrappedMCPPrompt()


def _prompt_args_to_model(descriptor: PromptDescriptor) -> type[BaseModel]:
    """Build a Pydantic model from a prompt's declared arguments.

    Every argument is typed ``str`` (MCP doesn't currently ship
    argument type schemas for prompts).  Required arguments become
    required Pydantic fields; optional ones default to ``None``.

    Parameters
    ----------
    descriptor : PromptDescriptor

    Returns
    -------
    type[BaseModel]
        Dynamically-created Pydantic model class.
    """
    fields: dict[str, Any] = {}
    for arg in descriptor.arguments:
        if arg.required:
            fields[arg.name] = (str, Field(description=arg.description))
        else:
            fields[arg.name] = (
                str | None,
                Field(default=None, description=arg.description),
            )
    return create_model(f"{descriptor.name}PromptInput", **fields)


def _mcp_message_to_langchain(message: Any) -> BaseMessage:
    """Convert an MCP ``PromptMessage`` into a LangChain :class:`BaseMessage`.

    MCP prompts can contain several content types (``TextContent``,
    ``ImageContent``, embedded ``Resource``).  This adapter flattens
    non-text content into a descriptive placeholder so downstream
    LangChain models don't have to understand MCP content shapes.

    Parameters
    ----------
    message : Any
        ``mcp.types.PromptMessage`` or a duck-typed equivalent.

    Returns
    -------
    BaseMessage
        :class:`AIMessage` when the MCP role is ``"assistant"``,
        :class:`HumanMessage` otherwise.
    """
    role = getattr(message, "role", "user")
    content = getattr(message, "content", None)
    text = _flatten_prompt_content(content)
    if role == "assistant":
        return AIMessage(content=text)
    return HumanMessage(content=text)


def _flatten_prompt_content(content: Any) -> str:
    """Render MCP prompt content (text / image / resource) as plain text.

    Parameters
    ----------
    content : Any
        A single MCP content item or a list of them.

    Returns
    -------
    str
    """
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(_flatten_prompt_content(item) for item in content if item)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "data") and hasattr(content, "mimeType"):
        return f"[{content.mimeType} content]"
    if hasattr(content, "resource"):
        uri = getattr(content.resource, "uri", "")
        return f"[resource: {uri}]"
    return str(content)
