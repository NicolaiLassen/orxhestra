"""MCP client — full async wrapper around FastMCP.

Wraps FastMCP's :class:`~fastmcp.Client` behind an orxhestra-shaped
interface that covers the *whole* Model Context Protocol surface:
tools, resources, resource templates, prompts, plus the bidirectional
session features (sampling, elicitation, logging, progress, roots).

Two usage modes:

- **Stateless (default).**  Each call opens, uses, and closes its own
  FastMCP session.  Good for one-shot reads (``list_tools``,
  ``read_resource``) from request-scoped code.
- **Session (``async with client: ...``).**  Opens a single long-lived
  FastMCP session and reuses it across every call inside the block.
  Required when you configure ``sampling_handler``,
  ``elicitation_handler``, ``log_handler``, or ``progress_handler``
  — those callbacks only fire against an open session — and also
  faster when you're making several calls in a row.

See Also
--------
MCPToolAdapter : Turns the tool catalogue into LangChain BaseTools.
MCPPromptAdapter : Fetches prompts and returns LangChain messages.
make_langchain_sampling_handler : Plug a LangChain chat model in as
    the MCP sampling callback so server-side tools can delegate LLM
    calls back to us.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datetime

    import mcp.types
    from fastmcp import Client
    from fastmcp.client.elicitation import ElicitationHandler
    from fastmcp.client.logging import LogHandler
    from fastmcp.client.progress import ProgressHandler
    from fastmcp.client.sampling import ClientSamplingHandler


class MCPClient:
    """Async wrapper around :class:`fastmcp.Client`.

    Parameters
    ----------
    transport : str, Path, or FastMCP server
        Either a transport URL (``"http://localhost:8080/mcp"``,
        ``"stdio://..."``), a file path, or an in-memory ``FastMCP``
        server object.  Passed directly to ``fastmcp.Client``.
    sampling_handler : ClientSamplingHandler, optional
        Async callback invoked when the MCP server requests an LLM
        completion back from the client.  Signature:
        ``async (params, context) -> CreateMessageResult``.  Use
        :func:`~orxhestra.integrations.mcp.sampling.make_langchain_sampling_handler`
        to plug a LangChain chat model in here.
    elicitation_handler : ElicitationHandler, optional
        Async callback invoked when the server pauses a tool to ask
        the client for structured input.
    log_handler : LogHandler, optional
        Async callback invoked for every log message emitted by the
        server.  Defaults to FastMCP's built-in handler (forwards to
        Python logging).
    progress_handler : ProgressHandler, optional
        Async callback invoked for progress events.  Can be
        overridden per-call in :meth:`call_tool`.
    roots : list[str | Path], optional
        Filesystem roots advertised to the server so it can reason
        about what it's allowed to touch.  Paths are converted to
        ``file://`` URIs.
    timeout : float, optional
        Default per-request timeout, in seconds.  Overridable per
        call.
    init_timeout : float, optional
        Timeout for the MCP handshake on connect.

    See Also
    --------
    MCPToolAdapter : Tool-catalogue adapter.
    MCPPromptAdapter : Prompt-catalogue adapter.
    """

    def __init__(
        self,
        transport: str | Path | Any,
        *,
        sampling_handler: ClientSamplingHandler | None = None,
        elicitation_handler: ElicitationHandler | None = None,
        log_handler: LogHandler | None = None,
        progress_handler: ProgressHandler | None = None,
        roots: list[str | Path] | None = None,
        timeout: float | None = None,
        init_timeout: float | None = None,
    ) -> None:
        self._transport = transport
        self._sampling_handler = sampling_handler
        self._elicitation_handler = elicitation_handler
        self._log_handler = log_handler
        self._progress_handler = progress_handler
        self._roots = self._normalise_roots(roots) if roots else None
        self._timeout = timeout
        self._init_timeout = init_timeout

        # Persistent session used when the caller enters via
        # ``async with client: ...``.  When ``None`` every call opens
        # and closes its own FastMCP session.
        self._session: Client | None = None
        self._session_depth: int = 0

    @property
    def url(self) -> str | None:
        """Return the transport URL when ``transport`` is a string."""
        return self._transport if isinstance(self._transport, str) else None

    @staticmethod
    def _normalise_roots(roots: list[str | Path]) -> list[str]:
        """Coerce path-like roots into ``file://`` URIs.

        Bare strings are passed through untouched so callers can
        supply pre-built URIs of any scheme.

        Parameters
        ----------
        roots : list[str or Path]

        Returns
        -------
        list[str]
        """
        out: list[str] = []
        for root in roots:
            if isinstance(root, Path):
                out.append(root.resolve().as_uri())
            elif isinstance(root, str) and "://" not in root:
                out.append(Path(root).resolve().as_uri())
            else:
                out.append(str(root))
        return out

    def _make_client(self) -> Client:
        """Construct a fresh FastMCP client with configured handlers.

        Returns
        -------
        fastmcp.Client
            A client instance with every configured handler wired.
            Callers are responsible for opening/closing it via
            ``async with`` (or the :meth:`__aenter__` path of
            :class:`MCPClient` itself).
        """
        from fastmcp import Client as _Client

        kwargs: dict[str, Any] = {}
        if self._sampling_handler is not None:
            kwargs["sampling_handler"] = self._sampling_handler
        if self._elicitation_handler is not None:
            kwargs["elicitation_handler"] = self._elicitation_handler
        if self._log_handler is not None:
            kwargs["log_handler"] = self._log_handler
        if self._progress_handler is not None:
            kwargs["progress_handler"] = self._progress_handler
        if self._roots is not None:
            kwargs["roots"] = self._roots
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._init_timeout is not None:
            kwargs["init_timeout"] = self._init_timeout

        if isinstance(self._transport, str):
            from fastmcp.client.transports import StreamableHttpTransport

            return _Client(StreamableHttpTransport(self._transport), **kwargs)
        return _Client(self._transport, **kwargs)

    # ── Session lifecycle ────────────────────────────────────────

    async def __aenter__(self) -> MCPClient:
        """Open a persistent FastMCP session for the duration of the block.

        Nested ``async with`` blocks share the outermost session —
        entering a second time just increments a depth counter so
        callers can freely compose session-scoped helpers without
        re-handshaking.

        Returns
        -------
        MCPClient
            ``self``, for convenience.
        """
        if self._session is None:
            self._session = self._make_client()
            await self._session.__aenter__()
        self._session_depth += 1
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the session once the outermost ``async with`` exits."""
        self._session_depth -= 1
        if self._session_depth <= 0 and self._session is not None:
            session = self._session
            self._session = None
            self._session_depth = 0
            await session.__aexit__(exc_type, exc, tb)

    async def ping(self) -> bool:
        """Round-trip a ping through the MCP server.

        Returns
        -------
        bool
            ``True`` when the server replies.
        """
        async with self._session_ctx() as client:
            return await client.ping()

    # ── Tools ────────────────────────────────────────────────────

    async def list_tools(self) -> list[Any]:
        """Return the server's tool catalogue.

        Returns
        -------
        list
            FastMCP :class:`~mcp.types.Tool` descriptors.
        """
        async with self._session_ctx() as client:
            return await client.list_tools()

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        progress_handler: ProgressHandler | None = None,
        timeout: float | datetime.timedelta | None = None,
        raise_on_error: bool = True,
    ) -> Any:
        """Invoke a tool by name.

        Parameters
        ----------
        name : str
            Tool name as advertised by :meth:`list_tools`.
        arguments : dict, optional
            Keyword arguments matching the tool's input schema.
        progress_handler : ProgressHandler, optional
            Overrides the client-wide handler for this call.  Lets
            callers attach a per-invocation progress sink without
            mutating shared state.
        timeout : float or timedelta, optional
            Overrides the client-wide request timeout.
        raise_on_error : bool
            When ``True`` (default) tool errors raise; when ``False``
            they are returned as part of the ``CallToolResult``.

        Returns
        -------
        mcp.types.CallToolResult
            The FastMCP call result.  Contains a ``content`` list and
            ``isError`` flag.
        """
        async with self._session_ctx() as client:
            return await client.call_tool(
                name,
                arguments,
                timeout=timeout,
                progress_handler=progress_handler,
                raise_on_error=raise_on_error,
            )

    # ── Resources ────────────────────────────────────────────────

    async def list_resources(self) -> list[Any]:
        """Return the server's static resource catalogue.

        Returns
        -------
        list
            FastMCP :class:`~mcp.types.Resource` descriptors.
        """
        async with self._session_ctx() as client:
            return await client.list_resources()

    async def list_resource_templates(self) -> list[Any]:
        """Return the server's parameterized resource templates.

        Templates are resource URIs with placeholders (e.g.
        ``file://{path}``).  Pair with :meth:`read_resource` to fetch
        a concrete URI derived from one of these templates.

        Returns
        -------
        list
            FastMCP :class:`~mcp.types.ResourceTemplate` descriptors.
        """
        async with self._session_ctx() as client:
            return await client.list_resource_templates()

    async def read_resource(self, uri: str) -> Any:
        """Read a resource by URI.

        Parameters
        ----------
        uri : str
            Resource URI as advertised by :meth:`list_resources` or
            produced from a template from
            :meth:`list_resource_templates`.

        Returns
        -------
        Any
            FastMCP ``ReadResourceResult`` — typically a list of
            text / blob content items.
        """
        async with self._session_ctx() as client:
            return await client.read_resource(uri)

    # ── Prompts ──────────────────────────────────────────────────

    async def list_prompts(self) -> list[Any]:
        """Return the server's prompt catalogue.

        Prompts are MCP's third primitive alongside tools and
        resources — named, argument-templated conversations the
        server publishes for clients to pull into their own LLM
        context.

        Returns
        -------
        list
            FastMCP :class:`~mcp.types.Prompt` descriptors.
        """
        async with self._session_ctx() as client:
            return await client.list_prompts()

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Materialise a prompt template into concrete messages.

        Parameters
        ----------
        name : str
            Prompt name as advertised by :meth:`list_prompts`.
        arguments : dict, optional
            Values to interpolate into the prompt's placeholders.

        Returns
        -------
        mcp.types.GetPromptResult
            The resolved prompt — a description plus a list of
            role/content messages ready to splice into an LLM
            conversation.
        """
        async with self._session_ctx() as client:
            return await client.get_prompt(name, arguments)

    # ── Completion & cancellation ────────────────────────────────

    async def complete(
        self,
        ref: mcp.types.ResourceTemplateReference | mcp.types.PromptReference,
        argument: dict[str, str],
        context_arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Request argument completion from the server.

        Used to implement IDE-style autocomplete for prompt and
        resource-template parameters.

        Parameters
        ----------
        ref : PromptReference or ResourceTemplateReference
            Which prompt or resource template to complete against.
        argument : dict[str, str]
            The ``(name, value)`` pair being completed.  The server
            suggests completions for ``value`` given ``name``.
        context_arguments : dict, optional
            Prior argument values that scope the completion.

        Returns
        -------
        mcp.types.Completion
        """
        async with self._session_ctx() as client:
            return await client.complete(ref, argument, context_arguments)

    async def cancel(
        self, request_id: str | int, reason: str | None = None,
    ) -> None:
        """Cancel an in-flight MCP request.

        Parameters
        ----------
        request_id : str or int
            The request to cancel.
        reason : str, optional
            Short human-readable reason attached to the notification.
        """
        async with self._session_ctx() as client:
            await client.cancel(request_id, reason)

    async def set_logging_level(self, level: Any) -> None:
        """Subscribe to server-side logs at ``level`` or above.

        Parameters
        ----------
        level : mcp.types.LoggingLevel
            One of ``"debug"``, ``"info"``, ``"notice"``,
            ``"warning"``, ``"error"``, ``"critical"``, ``"alert"``,
            ``"emergency"``.
        """
        async with self._session_ctx() as client:
            await client.set_logging_level(level)

    # ── Internal helpers ─────────────────────────────────────────

    def _session_ctx(self) -> _SessionContext:
        """Return an async context manager that yields an open client.

        When a persistent session is already open (we're inside
        ``async with client: ...``), reuse it.  Otherwise open a
        short-lived one for the duration of the call.  Makes every
        public method safe to call in either mode.
        """
        return _SessionContext(self)


class _SessionContext:
    """Async context manager that hands out an open FastMCP client.

    Implementation detail of :class:`MCPClient` — not exported.
    """

    def __init__(self, parent: MCPClient) -> None:
        self._parent = parent
        self._own: Client | None = None

    async def __aenter__(self) -> Client:
        if self._parent._session is not None:
            return self._parent._session
        self._own = self._parent._make_client()
        await self._own.__aenter__()
        return self._own

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._own is not None:
            await self._own.__aexit__(exc_type, exc, tb)
            self._own = None
