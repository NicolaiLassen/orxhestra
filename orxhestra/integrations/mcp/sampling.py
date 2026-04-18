"""Sampling + elicitation + progress + logging handler helpers for MCP.

MCP is bidirectional: once a tool is called the server can call back
into the *client* to request an LLM completion (sampling), ask the
user for structured input (elicitation), emit progress events, or
forward log lines.  These helpers take care of the plumbing so each
callback has a sensible orxhestra-shaped default:

- :func:`make_langchain_sampling_handler` — wraps a LangChain chat
  model so server-side tools can delegate LLM work back to it.
- :func:`make_python_logging_handler` — forwards MCP log messages to
  a Python :class:`logging.Logger`.
- :func:`make_stream_progress_handler` — routes progress events to a
  plain async callable (useful for wiring into a :class:`Writer` or
  a ``tqdm``-style sink).

Elicitation doesn't get a generic helper — the response shape is
completely application-specific (CLI prompt, web form, Slack
interactive, ...) so callers register their own handler.

See Also
--------
MCPClient : Accepts any of these via constructor kwargs.
"""

from __future__ import annotations

import logging as _logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mcp.types
    from langchain_core.language_models import BaseChatModel


logger: _logging.Logger = _logging.getLogger(__name__)


def make_langchain_sampling_handler(
    llm: BaseChatModel,
    *,
    default_max_tokens: int = 1024,
    default_temperature: float | None = None,
) -> Callable[..., Awaitable[mcp.types.CreateMessageResult]]:
    """Build an MCP sampling handler backed by a LangChain chat model.

    Returns a callable matching MCP's ``SamplingFnT`` so the server
    can request ``ctx.sample(...)`` and have the work routed through
    our own LangChain model — which means our provider, our billing,
    our tracing.

    The handler converts :class:`mcp.types.SamplingMessage` input to
    LangChain messages, calls :meth:`BaseChatModel.ainvoke`, and
    wraps the reply back into :class:`mcp.types.CreateMessageResult`.

    Parameters
    ----------
    llm : BaseChatModel
        LangChain chat model that fulfils the sampling request.
    default_max_tokens : int
        Applied when the server doesn't specify ``max_tokens``.  MCP
        requires a ``max_tokens`` value in the request; this default
        just keeps us safe if the server sends a loose one.
    default_temperature : float, optional
        Applied when the server doesn't specify ``temperature``.

    Returns
    -------
    Callable
        ``async (context, params) -> CreateMessageResult`` suitable
        for :class:`~orxhestra.integrations.mcp.client.MCPClient`'s
        ``sampling_handler`` kwarg.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    def _to_langchain(
        messages: list[mcp.types.SamplingMessage],
        system_prompt: str | None,
    ) -> list[Any]:
        lc_messages: list[Any] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        for msg in messages:
            text = _flatten_sampling_content(msg.content)
            if msg.role == "assistant":
                lc_messages.append(AIMessage(content=text))
            else:
                lc_messages.append(HumanMessage(content=text))
        return lc_messages

    async def handler(
        context: Any,  # noqa: ARG001
        params: mcp.types.CreateMessageRequestParams,
    ) -> mcp.types.CreateMessageResult:
        """Serve one MCP sampling request via the configured LangChain model."""
        import mcp.types as _mcp_types

        lc_messages = _to_langchain(params.messages, params.systemPrompt)
        max_tokens = params.maxTokens or default_max_tokens
        temperature = (
            params.temperature
            if params.temperature is not None
            else default_temperature
        )

        try:
            kwargs: dict[str, Any] = {"max_tokens": max_tokens}
            if temperature is not None:
                kwargs["temperature"] = temperature
            ai_message = await llm.ainvoke(lc_messages, **kwargs)
        except TypeError:
            # Model doesn't accept max_tokens/temperature kwargs on
            # ainvoke — fall back to the bare call.
            ai_message = await llm.ainvoke(lc_messages)

        text = ai_message.content if isinstance(ai_message.content, str) else str(
            ai_message.content,
        )
        model_name = (
            getattr(ai_message, "response_metadata", {}).get("model_name")
            or getattr(ai_message, "response_metadata", {}).get("model")
            or getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or "langchain-chat-model"
        )

        return _mcp_types.CreateMessageResult(
            role="assistant",
            content=_mcp_types.TextContent(type="text", text=text),
            model=str(model_name),
            stopReason="endTurn",
        )

    return handler


def make_python_logging_handler(
    target: _logging.Logger | str | None = None,
) -> Callable[[mcp.types.LoggingMessageNotificationParams], Awaitable[None]]:
    """Build a log handler that forwards MCP server logs to Python logging.

    MCP log levels map directly onto Python's numeric levels:

    - ``debug`` → :data:`logging.DEBUG`
    - ``info`` / ``notice`` → :data:`logging.INFO`
    - ``warning`` → :data:`logging.WARNING`
    - ``error`` → :data:`logging.ERROR`
    - ``critical`` / ``alert`` / ``emergency`` → :data:`logging.CRITICAL`

    Parameters
    ----------
    target : logging.Logger or str, optional
        Destination logger.  When a string, looks up the logger by
        name via :func:`logging.getLogger`.  Defaults to the
        ``orxhestra.integrations.mcp`` logger.

    Returns
    -------
    Callable
        ``async (params) -> None`` suitable for the ``log_handler``
        kwarg of :class:`~orxhestra.integrations.mcp.client.MCPClient`.
    """
    if target is None:
        target_logger = logger
    elif isinstance(target, str):
        target_logger = _logging.getLogger(target)
    else:
        target_logger = target

    level_map: dict[str, int] = {
        "debug": _logging.DEBUG,
        "info": _logging.INFO,
        "notice": _logging.INFO,
        "warning": _logging.WARNING,
        "error": _logging.ERROR,
        "critical": _logging.CRITICAL,
        "alert": _logging.CRITICAL,
        "emergency": _logging.CRITICAL,
    }

    async def handler(params: mcp.types.LoggingMessageNotificationParams) -> None:
        """Forward a single MCP log message to the target logger."""
        level = level_map.get(params.level, _logging.INFO)
        data = params.data
        if isinstance(data, str):
            message = data
        else:
            message = str(data)
        scope = f":{params.logger}" if params.logger else ""
        target_logger.log(level, "[MCP%s] %s", scope, message)

    return handler


def make_stream_progress_handler(
    sink: Callable[[float, float | None, str | None], Awaitable[None] | None],
) -> Callable[[float, float | None, str | None], Awaitable[None]]:
    """Adapt a simple sink into an MCP ``ProgressHandler``.

    Accepts either a sync or async sink so callers can pass a plain
    lambda or an async function without caring about the callback
    signature MCP expects.

    Parameters
    ----------
    sink : callable
        ``(progress, total, message) -> None`` — sync or async.

    Returns
    -------
    Callable
        ``async (progress, total, message) -> None`` suitable for
        :class:`~orxhestra.integrations.mcp.client.MCPClient`'s
        ``progress_handler`` or the per-call ``progress_handler``
        kwarg on :meth:`~orxhestra.integrations.mcp.client.MCPClient.call_tool`.
    """
    import inspect

    is_async = inspect.iscoroutinefunction(sink)

    async def handler(
        progress: float, total: float | None, message: str | None,
    ) -> None:
        """Route a single progress event to the sink."""
        if is_async:
            await sink(progress, total, message)  # type: ignore[misc]
        else:
            sink(progress, total, message)

    return handler


def _flatten_sampling_content(content: Any) -> str:
    """Render :class:`mcp.types.SamplingMessage` content as plain text.

    Parameters
    ----------
    content : Any
        MCP content item(s) — ``TextContent``, ``ImageContent``, or a
        list thereof.

    Returns
    -------
    str
    """
    if content is None:
        return ""
    if isinstance(content, list):
        return "\n".join(
            _flatten_sampling_content(item) for item in content if item
        )
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "data") and hasattr(content, "mimeType"):
        return f"[{content.mimeType} content]"
    return str(content)
