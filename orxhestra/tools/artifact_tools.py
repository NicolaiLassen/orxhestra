"""Built-in artifact tools for LLM agents.

Provides tools that let agents save, load, and list artifacts
during execution.  These tools require an ``artifact_service``
on the ``InvocationContext``.

Usage::

    from orxhestra.tools.artifact_tools import make_artifact_tools

    tools = make_artifact_tools()
"""

from __future__ import annotations

import base64
import mimetypes
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from orxhestra.agents.invocation_context import InvocationContext


def make_artifact_tools() -> list[BaseTool]:
    """Create artifact tools for saving, loading, and listing artifacts.

    The tools operate on the artifact service attached to the current
    :class:`InvocationContext`. Context is injected automatically by
    the agent runtime via :meth:`AgentTool.inject_context`.

    Returns
    -------
    list[BaseTool]
        Three tools: ``save_artifact``, ``load_artifact``,
        ``list_artifacts``.

    See Also
    --------
    BaseArtifactService : Backend the tools delegate to.
    CallContext.save_artifact : High-level save API used inside tools.
    """
    from orxhestra.tools.call_context import CallContext

    # Shared state across the closure — holds the injected context
    _holder: dict[str, Any] = {"ctx": None}

    def _inject(ctx: InvocationContext) -> None:
        _holder["ctx"] = ctx

    def _get_call_ctx() -> CallContext:
        ctx = _holder["ctx"]
        if ctx is None:
            msg = "Artifact tools require a context. Call inject_context(ctx) before invoking."
            raise RuntimeError(msg)
        return CallContext(ctx)

    async def save_artifact(
        filename: str,
        content: str,
        mime_type: str = "",
        is_base64: bool = False,
    ) -> str:
        """Save a file or blob to the artifact store.

        Args:
            filename: Name of the artifact (e.g. "report.md").
            content: The artifact content as text, or base64-encoded binary.
            mime_type: MIME type. Auto-detected from filename if empty.
            is_base64: Set to true if content is base64-encoded binary data.
        """
        call_ctx: CallContext = _get_call_ctx()

        if is_base64:
            data: bytes = base64.b64decode(content)
        else:
            data = content.encode("utf-8")

        if not mime_type:
            guessed: str | None = mimetypes.guess_type(filename)[0]
            mime_type = guessed or "application/octet-stream"

        version: int | None = await call_ctx.save_artifact(
            filename, data, mime_type=mime_type,
        )
        if version is None:
            return "Error: no artifact service configured."
        return f"Saved artifact '{filename}' (version {version}, {len(data)} bytes)."

    async def load_artifact(
        filename: str,
        version: int | None = None,
    ) -> str:
        """Load an artifact from the store.

        Args:
            filename: Name of the artifact to load.
            version: Specific version number to load. Omit for latest.
        """
        call_ctx: CallContext = _get_call_ctx()
        data: bytes | None = await call_ctx.load_artifact(filename, version=version)
        if data is None:
            return f"Artifact '{filename}' not found."

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            encoded: str = base64.b64encode(data).decode("ascii")
            return f"[binary, {len(data)} bytes, base64]: {encoded}"

    async def list_artifacts() -> str:
        """List all available artifact filenames in the current session."""
        call_ctx: CallContext = _get_call_ctx()
        keys: list[str] = await call_ctx.list_artifacts()
        if not keys:
            return "No artifacts saved."
        return "\n".join(keys)

    save_tool: BaseTool = StructuredTool.from_function(
        coroutine=save_artifact,
        name="save_artifact",
        description="Save a file or blob to the artifact store.",
    )
    load_tool: BaseTool = StructuredTool.from_function(
        coroutine=load_artifact,
        name="load_artifact",
        description="Load an artifact from the store by filename.",
    )
    list_tool: BaseTool = StructuredTool.from_function(
        coroutine=list_artifacts,
        name="list_artifacts",
        description="List all available artifact filenames.",
    )

    # Attach inject_context so the agent runtime can inject context.
    # Use object.__setattr__ because StructuredTool is a Pydantic model.
    for tool in (save_tool, load_tool, list_tool):
        object.__setattr__(tool, "inject_context", _inject)

    return [save_tool, load_tool, list_tool]
