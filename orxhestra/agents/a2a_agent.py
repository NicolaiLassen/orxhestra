"""A2AAgent — proxy requests to a remote A2A v1.0 server.

Uses the ``SendMessage`` JSON-RPC method and yields SDK events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext
from orxhestra.events.event import Event, EventType

# A2A v1.0 protocol types (lightweight aliases, no external dep).
A2APart = dict[str, Any]
A2AMessage = dict[str, Any]
A2AArtifact = dict[str, Any]
A2AStatus = dict[str, Any]
A2ATask = dict[str, Any]
A2AResponse = dict[str, Any]


class A2AAgent(BaseAgent):
    """Agent that delegates to a remote A2A v1.0 server over HTTP.

    Parameters
    ----------
    name : str
        Local name for this agent in the agent tree.
    url : str
        Base URL of the remote A2A server (e.g. ``"http://localhost:9000"``).
    description : str
        Description used for routing decisions.
    """

    def __init__(
        self,
        name: str,
        url: str,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.url: str = url.rstrip("/")

    async def astream(
        self,
        input: str,
        config: RunnableConfig | None = None,
        *,
        ctx: InvocationContext | None = None,
    ) -> AsyncIterator[Event]:
        """Send a message to the remote A2A server and yield events.

        Parameters
        ----------
        input : str
            The user message to forward to the remote server.
        config : RunnableConfig, optional
            LangChain-compatible config dict (tags, callbacks, etc.).
        ctx : InvocationContext, optional
            Invocation context. Auto-created if not provided.

        Yields
        ------
        Event
            AGENT_START, AGENT_MESSAGE (with the remote answer), and
            AGENT_END events, in that order.
        """
        ctx = self._ensure_ctx(config, ctx)

        yield self._emit_event(ctx, EventType.AGENT_START)

        response_text: str = await self._send_message(input)

        yield self._emit_event(
            ctx,
            EventType.AGENT_MESSAGE,
            text=response_text,
        )

        yield self._emit_event(ctx, EventType.AGENT_END)

    async def _send_message(self, text: str) -> str:
        """Send a ``SendMessage`` JSON-RPC request and return the answer."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "A2AAgent requires httpx. Install with: pip install httpx"
            ) from None

        message: A2AMessage = {
            "messageId": str(uuid4()),
            "role": "user",
            "parts": [{"text": text, "mediaType": "text/plain"}],
        }

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": message,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                self.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "A2A-Version": "1.0",
                },
            )
            resp.raise_for_status()
            data: A2AResponse = resp.json()

        return self._extract_answer(data)

    @staticmethod
    def _extract_answer(data: A2AResponse) -> str:
        """Extract the text answer from a ``SendMessage`` response.

        The A2A v1.0 ``SendMessageResponse`` is a oneof: either a
        ``task`` (with artifacts, status, history) or a direct ``message``.
        """
        result: dict[str, Any] = data.get("result", {})

        # Direct message response.
        direct_msg: A2AMessage | None = result.get("message")
        if direct_msg is not None:
            text: str = _extract_text_from_parts(direct_msg.get("parts", []))
            if text:
                return text

        # Task response — check if result itself is a task (has "id" and "status").
        task: A2ATask | None = None
        if "id" in result and "status" in result:
            task = result
        elif "task" in result:
            task = result["task"]

        if task is None:
            return ""

        # 1. Artifacts carry the final output.
        artifacts: list[A2AArtifact] = task.get("artifacts", [])
        for artifact in artifacts:
            text = _extract_text_from_parts(artifact.get("parts", []))
            if text:
                return text

        # 2. Status message.
        status: A2AStatus = task.get("status", {})
        status_msg: A2AMessage | None = status.get("message")
        if status_msg is not None:
            text = _extract_text_from_parts(status_msg.get("parts", []))
            if text:
                return text

        # 3. History — last agent message.
        history: list[A2AMessage] = task.get("history", [])
        for msg in reversed(history):
            if msg.get("role") == "agent":
                text = _extract_text_from_parts(msg.get("parts", []))
                if text:
                    return text

        return ""


def _extract_text_from_parts(parts: list[A2APart]) -> str:
    """Return the first text value from a list of A2A v1.0 parts."""
    for part in parts:
        text: str | None = part.get("text")
        if text:
            return text
    return ""
