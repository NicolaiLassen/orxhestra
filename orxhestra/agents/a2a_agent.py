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

_Ed25519PrivateKey = Any
_DidResolver = Any


class A2AAgent(BaseAgent):
    """Agent that delegates to a remote A2A v1.0 server over HTTP.

    Sends a JSON-RPC ``SendMessage`` request per turn and converts
    the server's response stream back into :class:`Event` objects.

    Parameters
    ----------
    name : str
        Local name for this agent in the agent tree.
    url : str
        Base URL of the remote A2A server
        (e.g. ``"http://localhost:9000"``).
    description : str
        Description used for routing decisions.
    signing_key : Ed25519PrivateKey, optional
        When set, outgoing A2A messages are signed with this key and
        the server's :class:`AgentCard` is used to verify responses.
        Requires ``orxhestra[auth]``.
    signing_did : str
        DID matching ``signing_key``.  Attached to each signed
        message.
    require_signed_response : bool
        When ``True``, responses from the remote server that lack a
        valid signature raise :class:`RuntimeError`.  Defaults to
        ``False`` for back-compat with unsigned peers.
    resolver : DidResolver, optional
        Resolver used to verify response signatures.  Defaults to a
        :class:`CompositeResolver` covering ``did:key``.

    See Also
    --------
    BaseAgent : Base class this extends.
    AgentCard : Remote card the server publishes for discovery.
    Message : A2A-wire message type sent on each turn.
    Task : Remote task wrapping the message execution.
    orxhestra.a2a.signing : Signature helpers used on the wire.

    Examples
    --------
    >>> remote = A2AAgent(
    ...     name="remote_researcher",
    ...     url="https://researcher.example.com",
    ...     description="Web research specialist.",
    ... )
    >>> async for event in remote.astream("Summarize arxiv:2024.12345"):
    ...     print(event.text)
    """

    def __init__(
        self,
        name: str,
        url: str,
        description: str = "",
        *,
        signing_key: _Ed25519PrivateKey | None = None,
        signing_did: str = "",
        require_signed_response: bool = False,
        resolver: _DidResolver | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            signing_key=signing_key,
            signing_did=signing_did,
        )
        self.url: str = url.rstrip("/")
        self.require_signed_response = require_signed_response
        self._resolver = resolver

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
        """Send a ``SendMessage`` JSON-RPC request and return the answer.

        When :attr:`signing_key` is set, the outgoing message is
        signed via :func:`orxhestra.a2a.signing.sign_message`.  When
        :attr:`require_signed_response` is true, the returned agent
        message is verified before its text is extracted.

        Parameters
        ----------
        text : str
            Body of the outgoing user message.

        Returns
        -------
        str
            Plain text of the remote agent's reply.

        Raises
        ------
        ImportError
            If ``httpx`` is not installed.
        RuntimeError
            If ``require_signed_response`` is set and the response
            message is unsigned or the signature fails to verify.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "A2AAgent requires httpx. Install with: pip install httpx"
            ) from None

        from orxhestra.a2a.types import Message as A2AMessageModel
        from orxhestra.a2a.types import Part as A2APartModel
        from orxhestra.a2a.types import Role

        message_model = A2AMessageModel(
            message_id=str(uuid4()),
            role=Role.USER,
            parts=[A2APartModel(text=text, media_type="text/plain")],
        )

        if self.signing_key is not None and self.signing_did:
            from orxhestra.a2a.signing import sign_message as sign_a2a_message

            message_model = sign_a2a_message(
                message_model, self.signing_key, self.signing_did,
            )

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "SendMessage",
            "params": {
                "message": message_model.model_dump(by_alias=True, exclude_none=True),
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

        if self.require_signed_response:
            await self._verify_response(data)

        return self._extract_answer(data)

    async def _verify_response(self, data: A2AResponse) -> None:
        """Raise ``RuntimeError`` when the response lacks a valid signature.

        Walks the JSON-RPC result looking for the agent's
        ``status.message`` (or direct ``message``) and verifies it via
        :func:`orxhestra.a2a.signing.verify_message`.

        Parameters
        ----------
        data : dict[str, Any]
            Parsed JSON-RPC response body.

        Raises
        ------
        RuntimeError
            If no agent message is present or signature verification
            fails.
        """
        from orxhestra.a2a.signing import verify_message as verify_a2a_message
        from orxhestra.a2a.types import Message as A2AMessageModel

        resolver = self._resolver
        if resolver is None:
            from orxhestra.security.did import DidKeyResolver

            resolver = DidKeyResolver()
            self._resolver = resolver

        result = data.get("result", {}) or {}
        raw_message = result.get("message")
        if raw_message is None:
            status = (result.get("status") or {}) if isinstance(result, dict) else {}
            raw_message = status.get("message")

        if raw_message is None:
            raise RuntimeError(
                "A2AAgent require_signed_response=True but response has no agent message."
            )

        message = A2AMessageModel.model_validate(raw_message)
        if not await verify_a2a_message(message, resolver):
            raise RuntimeError(
                "A2AAgent response signature missing or invalid; "
                "refusing to accept under require_signed_response=True."
            )

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
