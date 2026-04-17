"""A2AServer — spec-compliant A2A v1.0 server over JSON-RPC 2.0.

Exposes any BaseAgent as an A2A endpoint with:
  - POST / — JSON-RPC 2.0 dispatch (SendMessage, SendStreamingMessage, etc.)
  - GET  /.well-known/agent-card.json  — Agent Card discovery

Run with:
    uvicorn my_module:app
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
)
from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.invocation_context import InvocationContext as Context
from orxhestra.sessions.base_session_service import BaseSessionService


def _now_iso() -> str:
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
        """
        self.agent = agent
        self.session_service = session_service
        self.app_name = app_name
        self.version = version
        self.url = url
        self.skills = skills or []

        # In-memory task store: task_id -> Task (bounded to prevent leaks).
        self._tasks: dict[str, Task] = {}
        self._max_tasks: int = 10_000


    def _build_agent_card(self) -> AgentCard:
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
        )


    def _create_task(self, message: Message, context_id: str | None = None) -> Task:
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
        """Run the agent and collect the final answer into the task."""
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
        user_text = _extract_text(send_params.message)
        task = self._create_task(
            send_params.message,
            context_id=send_params.message.context_id,
        )

        await self._run_agent_for_task(task, user_text)

        return _jsonrpc_success(request_id, task)

    async def _handle_send_streaming_message(
        self, params: dict[str, Any], request_id: Any,
    ) -> StreamingResponse:
        send_params = MessageSendParams.model_validate(params)
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

    async def _stream_task(
        self, task: Task, user_text: str, request_id: Any,
    ) -> AsyncIterator[str]:
        """Run the agent and yield SSE events as JSON-RPC responses."""
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
