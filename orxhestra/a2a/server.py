"""Spec-compliant A2A v1.0 server over JSON-RPC 2.0.

Exposes any :class:`BaseAgent` as an A2A endpoint with:

- ``POST /``                          — JSON-RPC 2.0 dispatch
  (``SendMessage``, ``SendStreamingMessage``, ``GetTask``,
  ``CancelTask``).
- ``GET  /.well-known/agent-card.json`` — :class:`AgentCard`
  discovery, including an optional :class:`VerificationMethod` list
  when the server has a signing identity.

Optionally signs every outgoing agent message with Ed25519 and
verifies incoming signed messages against a
:class:`~orxhestra.security.did.DidResolver`.  Signing is **opt-in**
— when ``signing_key`` is unset the server behaves exactly as it did
before the identity layer existed.

Run with::

    uvicorn my_module:app

See Also
--------
orxhestra.agents.a2a_agent.A2AAgent : Client-side counterpart.
orxhestra.a2a.signing : Message signing / verification helpers.
orxhestra.a2a.types.VerificationMethod : Published on agent cards.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as _exc:
    raise ImportError(
        "A2AServer requires FastAPI. Install with: pip install orxhestra[a2a]"
    ) from _exc

from orxhestra.a2a.converters import events_to_a2a_stream
from orxhestra.a2a.signing import (
    sign_message as sign_a2a_message,
)
from orxhestra.a2a.signing import (
    verify_message as verify_a2a_message,
)
from orxhestra.a2a.types import (
    TERMINAL_STATES,
    A2AErrorCode,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Artifact,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskIdParams,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    VerificationMethod,
)
from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.sessions.base_session_service import BaseSessionService

# Local type alias to avoid a runtime dep on cryptography.
_Ed25519PrivateKey = Any
_DidResolver = Any


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns
    -------
    str
        Timezone-aware ISO 8601 timestamp — used to stamp
        :class:`TaskStatus` transitions.
    """
    return datetime.now(timezone.utc).isoformat()


class A2AServer:
    """Spec-compliant A2A v1.0 server that adapts a :class:`BaseAgent`.

    Implements:
      - ``SendMessage``            — run agent, return completed Task
      - ``SendStreamingMessage``   — run agent, stream SSE events
      - ``GetTask``                — retrieve task by ID
      - ``CancelTask``             — cancel a running task
      - Agent Card at ``/.well-known/agent-card.json``

    See Also
    --------
    A2AAgent : Client-side counterpart for calling remote servers.
    AgentCard : Discovery manifest served at the well-known URL.
    Task : Task object returned by ``SendMessage``.
    events_to_a2a_stream : Converter from SDK events to A2A events.

    Examples
    --------
    >>> from orxhestra import InMemorySessionService
    >>> from orxhestra.a2a.server import A2AServer
    >>> server = A2AServer(
    ...     agent=my_agent,
    ...     session_service=InMemorySessionService(),
    ...     url="http://localhost:8000",
    ... )
    >>> app = server.app  # FastAPI instance ready for uvicorn
    """

    def __init__(
        self,
        agent: BaseAgent,
        *,
        session_service: BaseSessionService,
        app_name: str = "agent-sdk",
        version: str = "1.0.0",
        url: str = "http://localhost:8000",
        skills: list[AgentSkill] | None = None,
        signing_key: _Ed25519PrivateKey | None = None,
        signer_did: str = "",
        require_signed: bool = False,
        resolver: _DidResolver | None = None,
    ) -> None:
        """Initialize the A2A server.

        Parameters
        ----------
        agent : BaseAgent
            The agent to expose via A2A.
        session_service : BaseSessionService | None
            Session backend for conversation state.
        app_name : str
            Application name used in session creation.
        version : str
            Version string for the Agent Card.
        url : str
            Base URL where the server is reachable.
        skills : list[AgentSkill] | None
            Skills advertised in the Agent Card.
        signing_key : Ed25519PrivateKey, optional
            When set, the server signs every outgoing A2A message and
            publishes its :class:`VerificationMethod` in the agent
            card.  Requires ``orxhestra[auth]``.
        signer_did : str
            DID matching ``signing_key``.  Required when
            ``signing_key`` is set.
        require_signed : bool
            When ``True``, incoming messages without a valid signature
            are rejected with ``INVALID_REQUEST``.  Defaults to ``False``
            for back-compat with unsigned peers.
        resolver : DidResolver, optional
            Resolver used to verify incoming signatures.  Defaults to
            a :class:`CompositeResolver` covering ``did:key`` only.
        """
        self.agent = agent
        self.session_service = session_service
        self.app_name = app_name
        self.version = version
        self.url = url
        self.skills = skills or []
        self.signing_key = signing_key
        self.signer_did = signer_did
        self.require_signed = require_signed
        self._resolver = resolver

        # In-memory task store: task_id -> Task (bounded to prevent leaks).
        self._tasks: dict[str, Task] = {}
        self._max_tasks: int = 10_000


    def _build_agent_card(self) -> AgentCard:
        """Return the :class:`AgentCard` published at the well-known URL.

        When ``signing_key`` is set, derives the matching ``did:key``
        and publishes a :class:`VerificationMethod` that remote peers
        can resolve to verify signed responses.  For ``did:web``
        identities the fragment falls back to ``#key-1`` since the
        spec doesn't mandate a canonical derivation.

        Returns
        -------
        AgentCard
            Fully-populated card ready for
            :meth:`pydantic.BaseModel.model_dump`.
        """
        verification_methods: list[VerificationMethod] | None = None
        controller: str | None = None
        if self.signing_key is not None and self.signer_did:
            import base58

            from orxhestra.security.crypto import (
                did_key_fragment,
                public_key_to_did_key,
                serialize_public_key,
            )

            public_key = self.signing_key.public_key()
            multibase = "z" + base58.b58encode(
                bytes([0xED, 0x01]) + serialize_public_key(public_key),
            ).decode("ascii")
            controller = self.signer_did
            try:
                fragment = did_key_fragment(self.signer_did)
            except ValueError:
                # did:web or other — fall back to fixed fragment.
                fragment = "#key-1"
            # Validate the advertised DID matches the signing key.
            derived_did = public_key_to_did_key(public_key)
            if self.signer_did.startswith("did:key:") and self.signer_did != derived_did:
                controller = derived_did
            verification_methods = [
                VerificationMethod(
                    id=f"{controller}{fragment}",
                    type="Ed25519VerificationKey2020",
                    controller=controller,
                    public_key_multibase=multibase,
                ),
            ]

        return AgentCard(
            name=self.agent.name,
            description=self.agent.description or "An AI agent exposed via A2A.",
            supported_interfaces=[
                AgentInterface(
                    url=self.url,
                    protocol_binding="JSONRPC",
                    protocol_version="1.0",
                ),
            ],
            version=self.version,
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
            ),
            skills=self.skills,
            controller=controller,
            verification_method=verification_methods,
        )

    def _default_resolver(self):
        """Return the lazily-constructed default :class:`~orxhestra.security.did.DidResolver`.

        When no resolver was supplied at construction, a
        :class:`~orxhestra.security.did.DidKeyResolver` is created on first use and cached
        for subsequent calls.

        Returns
        -------
        DidResolver
        """
        if self._resolver is not None:
            return self._resolver
        from orxhestra.security.did import DidKeyResolver

        self._resolver = DidKeyResolver()
        return self._resolver

    def _maybe_sign(self, message: Message) -> Message:
        """Sign ``message`` when the server has a signing identity configured.

        Parameters
        ----------
        message : Message
            Outgoing message to stamp.

        Returns
        -------
        Message
            ``message`` unchanged when no signing key is configured;
            otherwise a copy carrying a detached Ed25519 signature via
            :func:`orxhestra.a2a.signing.sign_message`.
        """
        if self.signing_key is None or not self.signer_did:
            return message
        return sign_a2a_message(message, self.signing_key, self.signer_did)


    def _create_task(self, message: Message, context_id: str | None = None) -> Task:
        """Create, store, and return a new :class:`Task` for ``message``.

        Parameters
        ----------
        message : Message
            Initial user message kicking off the task.
        context_id : str, optional
            Conversation identifier carried on the task.  A fresh
            UUID is generated when omitted.

        Returns
        -------
        Task
            Newly-registered task in ``SUBMITTED`` state.  The oldest
            tasks are evicted once ``_max_tasks`` is exceeded so the
            in-memory store cannot grow unbounded.
        """
        task = Task(
            id=str(uuid.uuid4()),
            context_id=context_id or str(uuid.uuid4()),
            status=TaskStatus(state=TaskState.SUBMITTED, timestamp=_now_iso()),
            history=[message],
        )
        self._tasks[task.id] = task
        # Evict oldest tasks when limit is reached.
        while len(self._tasks) > self._max_tasks:
            oldest_key = next(iter(self._tasks))
            del self._tasks[oldest_key]
        return task

    def _update_task_status(
        self,
        task: Task,
        state: TaskState,
        agent_message: Message | None = None,
    ) -> None:
        """Update ``task.status`` with a new state and optional message.

        Parameters
        ----------
        task : Task
            Task whose status is changing.
        state : TaskState
            New lifecycle state.
        agent_message : Message, optional
            Latest agent message to attach to the status snapshot
            (e.g. an error or progress update).
        """
        task.status = TaskStatus(
            state=state,
            message=agent_message,
            timestamp=_now_iso(),
        )

    async def _run_agent_for_task(
        self,
        task: Task,
        user_message: str,
    ) -> None:
        """Run the agent, collect the final answer, and sign the response.

        Parameters
        ----------
        task : Task
            Task object to mutate in place with status updates,
            artifacts, and history.
        user_message : str
            Plain text extracted from the incoming user message.

        Notes
        -----
        The constructed agent response message passes through
        :meth:`_maybe_sign` so it inherits the server's signing
        identity when configured.
        """
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id="anonymous",
        )
        ctx = Context(
            session_id=session.id,
            user_id="anonymous",
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            session=session,
        )

        self._update_task_status(task, TaskState.WORKING)

        final_answer = ""
        async for event in self.agent.astream(user_message, ctx=ctx):
            if event.is_final_response():
                final_answer = event.text

        # Build agent response message
        agent_msg = Message(
            role=Role.AGENT,
            parts=[Part(text=final_answer, media_type="text/plain")],
            task_id=task.id,
            context_id=task.context_id,
        )
        agent_msg = self._maybe_sign(agent_msg)

        # Add artifact
        artifact = Artifact(parts=[Part(text=final_answer, media_type="text/plain")])
        task.artifacts = [artifact]

        if task.history is not None:
            task.history.append(agent_msg)

        self._update_task_status(task, TaskState.COMPLETED, agent_message=agent_msg)


    async def _handle_send_message(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        send_params = MessageSendParams.model_validate(params)
        rejection = await self._verify_incoming(send_params.message, request_id)
        if rejection is not None:
            return rejection
        user_text = _extract_text(send_params.message)
        task = self._create_task(
            send_params.message,
            context_id=send_params.message.context_id,
        )

        await self._run_agent_for_task(task, user_text)

        return _jsonrpc_success(request_id, task)

    async def _handle_send_streaming_message(
        self, params: dict[str, Any], request_id: Any,
    ) -> StreamingResponse | JSONResponse:
        send_params = MessageSendParams.model_validate(params)
        rejection = await self._verify_incoming(send_params.message, request_id)
        if rejection is not None:
            return rejection
        user_text = _extract_text(send_params.message)
        task = self._create_task(
            send_params.message,
            context_id=send_params.message.context_id,
        )

        return StreamingResponse(
            self._stream_task(task, user_text, request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _verify_incoming(
        self, message: Message, request_id: Any,
    ) -> JSONResponse | None:
        """Enforce ``require_signed`` on an incoming message.

        Parameters
        ----------
        message : Message
            Incoming message extracted from the JSON-RPC params.
        request_id : Any
            Correlation id used for error envelopes.

        Returns
        -------
        JSONResponse or None
            ``None`` when the message is acceptable.  A JSON-RPC
            error response (with :data:`A2AErrorCode.INVALID_REQUEST`)
            when ``require_signed`` is set and verification failed.
        """
        if not self.require_signed:
            return None
        if await verify_a2a_message(message, self._default_resolver()):
            return None
        return _jsonrpc_error(
            request_id,
            A2AErrorCode.INVALID_REQUEST,
            "Message signature missing or invalid; server requires signed messages.",
        )

    async def _stream_task(
        self, task: Task, user_text: str, request_id: Any,
    ) -> AsyncIterator[str]:
        """Run the agent and yield SSE lines as JSON-RPC responses.

        Parameters
        ----------
        task : Task
            The task to stream updates for.
        user_text : str
            Plain text extracted from the incoming user message.
        request_id : Any
            Correlation id echoed on each streamed envelope.

        Yields
        ------
        str
            ``data: {...}\\n\\n`` SSE frames containing
            :class:`TaskStatusUpdateEvent` /
            :class:`TaskArtifactUpdateEvent` payloads wrapped in
            JSON-RPC responses.
        """
        session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id="anonymous",
        )
        ctx = Context(
            session_id=session.id,
            user_id="anonymous",
            app_name=self.app_name,
            agent_name=self.agent.name,
            state=dict(session.state),
            session=session,
        )

        self._update_task_status(task, TaskState.WORKING)

        # Emit initial "working" status
        working_event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.WORKING, timestamp=_now_iso()),
            final=False,
        )
        yield _sse_line(request_id, working_event)

        # Stream agent events, converting to A2A events
        async for a2a_event in events_to_a2a_stream(
            self.agent.astream(user_text, ctx=ctx),
            task_id=task.id,
            context_id=task.context_id,
        ):
            yield _sse_line(request_id, a2a_event)

        # Emit final "completed" status
        self._update_task_status(task, TaskState.COMPLETED)

        completed_event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.COMPLETED, timestamp=_now_iso()),
            final=True,
        )
        yield _sse_line(request_id, completed_event)

    async def _handle_get_task(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        query = TaskQueryParams.model_validate(params)
        task = self._tasks.get(query.id)
        if task is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"Task {query.id} not found",
            )
        return _jsonrpc_success(request_id, task)

    async def _handle_cancel_task(
        self, params: dict[str, Any], request_id: Any,
    ) -> JSONResponse:
        task_params = TaskIdParams.model_validate(params)
        task = self._tasks.get(task_params.id)
        if task is None:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_FOUND,
                f"Task {task_params.id} not found",
            )
        if task.status.state in TERMINAL_STATES:
            return _jsonrpc_error(
                request_id,
                A2AErrorCode.TASK_NOT_CANCELABLE,
                f"Task {task_params.id} is in terminal state {task.status.state.value}",
            )
        self._update_task_status(task, TaskState.CANCELED)
        return _jsonrpc_success(request_id, task)


    _METHOD_MAP = {
        "SendMessage": "_handle_send_message",
        "SendStreamingMessage": "_handle_send_streaming_message",
        "GetTask": "_handle_get_task",
        "CancelTask": "_handle_cancel_task",
        # Backwards compatibility with v0.x method names
        "message/send": "_handle_send_message",
        "message/stream": "_handle_send_streaming_message",
        "tasks/get": "_handle_get_task",
        "tasks/cancel": "_handle_cancel_task",
    }

    async def _dispatch(self, request: Request) -> Any:
        """Parse a JSON-RPC envelope and route to the matching handler.

        Parameters
        ----------
        request : Request
            FastAPI request containing a JSON-RPC 2.0 body.

        Returns
        -------
        Any
            :class:`JSONResponse` or :class:`StreamingResponse`
            depending on the method invoked.  Always a valid JSON-RPC
            response — parse errors and unknown methods are mapped to
            error envelopes rather than raised.
        """
        try:
            body = await request.json()
        except Exception:
            return _jsonrpc_error(None, A2AErrorCode.PARSE_ERROR, "Invalid JSON")

        try:
            rpc = JSONRPCRequest.model_validate(body)
        except Exception:
            return _jsonrpc_error(
                body.get("id"), A2AErrorCode.INVALID_REQUEST, "Invalid JSON-RPC request",
            )

        handler_name = self._METHOD_MAP.get(rpc.method)
        if handler_name is None:
            return _jsonrpc_error(
                rpc.id, A2AErrorCode.METHOD_NOT_FOUND, f"Method {rpc.method!r} not found",
            )

        handler = getattr(self, handler_name)
        return await handler(rpc.params or {}, rpc.id)


    def as_fastapi_app(self) -> FastAPI:
        """Build and return a FastAPI application with A2A v1.0 routes.

        Returns
        -------
        FastAPI
            Configured application with JSON-RPC and Agent Card endpoints.
        """
        app = FastAPI(title=f"{self.agent.name} A2A Server")
        server = self

        @app.get("/.well-known/agent-card.json")
        @app.get("/.well-known/agent.json")
        async def agent_card() -> dict:
            card = server._build_agent_card()
            return card.model_dump(by_alias=True, exclude_none=True)

        @app.post("/")
        async def jsonrpc_endpoint(request: Request) -> Any:
            return await server._dispatch(request)

        @app.get("/")
        async def health() -> dict:
            return {"status": "ok", "agent": server.agent.name}

        return app




def _extract_text(message: Message) -> str:
    """Extract plain text from a Message's parts.

    Parameters
    ----------
    message : Message
        The A2A message to extract text from.

    Returns
    -------
    str
        Concatenated text from all text parts, or empty string.
    """
    texts: list[str] = []
    for part in message.parts:
        if part.text is not None:
            texts.append(part.text)
    return " ".join(texts) if texts else ""


def _jsonrpc_success(request_id: Any, result: Any) -> JSONResponse:
    """Wrap a result in a JSON-RPC 2.0 success response.

    Parameters
    ----------
    request_id : str | int | None
        The JSON-RPC request ID to echo back.
    result : dict
        Payload to include as the ``result`` field.

    Returns
    -------
    JSONResponse
        FastAPI JSON response with the JSON-RPC envelope.
    """
    if hasattr(result, "model_dump"):
        result = result.model_dump(by_alias=True, exclude_none=True)
    resp = JSONRPCResponse(id=request_id, result=result)
    return JSONResponse(resp.model_dump(by_alias=True, exclude_none=True))


def _jsonrpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    """Wrap an error in a JSON-RPC 2.0 error response.

    Parameters
    ----------
    request_id : str | int | None
        The JSON-RPC request ID to echo back.
    code : int
        Numeric error code (see ``A2AErrorCode``).
    message : str
        Human-readable error description.

    Returns
    -------
    JSONResponse
        FastAPI JSON response with the JSON-RPC error envelope.
    """
    resp = JSONRPCResponse(
        id=request_id or 0,
        error=JSONRPCError(code=code, message=message),
    )
    return JSONResponse(
        resp.model_dump(by_alias=True, exclude_none=True),
        status_code=400 if code != A2AErrorCode.INTERNAL_ERROR else 500,
    )


def _sse_line(request_id: Any, event: Any) -> str:
    """Format an A2A event as a Server-Sent Events data line.

    Parameters
    ----------
    request_id : str | int | None
        The JSON-RPC request ID to echo back.
    event : Event
        A2A event (e.g. ``TaskStatusUpdateEvent``) to serialize.

    Returns
    -------
    str
        SSE-formatted ``data: ...`` line with trailing double newline.
    """
    if hasattr(event, "model_dump"):
        result_data = event.model_dump(by_alias=True, exclude_none=True)
    else:
        result_data = event
    resp = JSONRPCResponse(id=request_id, result=result_data)
    return f"data: {json.dumps(resp.model_dump(by_alias=True, exclude_none=True))}\n\n"
