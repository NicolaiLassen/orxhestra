"""A2AServer — spec-compliant A2A server over JSON-RPC 2.0.

Exposes any BaseAgent as an A2A endpoint with:
  - POST / — JSON-RPC 2.0 dispatch (message/send, message/stream, etc.)
  - GET  /.well-known/agent.json  — Agent Card discovery

Run with:
    uvicorn my_module:app

Examples
--------
>>> from langchain_adk.agents.llm_agent import LlmAgent
>>> from langchain_adk.a2a.server import A2AServer
>>> from langchain_adk.sessions.in_memory_session_service import InMemorySessionService
>>>
>>> agent = LlmAgent("my_agent", llm=llm)
>>> server = A2AServer(agent, session_service=InMemorySessionService())
>>> app = server.as_fastapi_app()
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from langchain_adk.a2a.converters import events_to_a2a_stream
from langchain_adk.a2a.types import (
    TERMINAL_STATES,
    A2AErrorCode,
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Artifact,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MessageSendParams,
    Role,
    Task,
    TaskIdParams,
    TaskQueryParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from langchain_adk.agents.base_agent import BaseAgent
from langchain_adk.agents.context import Context
from langchain_adk.sessions.base_session_service import BaseSessionService


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class A2AServer:
    """Spec-compliant A2A server that adapts a BaseAgent.

    Implements:
      - ``message/send``    — run agent, return completed Task
      - ``message/stream``  — run agent, stream SSE events
      - ``tasks/get``       — retrieve task by ID
      - ``tasks/cancel``    — cancel a running task (stub)
      - Agent Card at ``/.well-known/agent.json``
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
        self.agent = agent
        self.session_service = session_service
        self.app_name = app_name
        self.version = version
        self.url = url
        self.skills = skills or []

        # In-memory task store: task_id -> Task
        self._tasks: dict[str, Task] = {}

    # ------------------------------------------------------------------
    # Agent Card
    # ------------------------------------------------------------------

    def _build_agent_card(self) -> AgentCard:
        return AgentCard(
            name=self.agent.name,
            description=self.agent.description or "An AI agent exposed via A2A.",
            url=self.url,
            version=self.version,
            capabilities=AgentCapabilities(
                streaming=True,
                push_notifications=False,
            ),
            skills=self.skills,
        )

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------

    def _create_task(self, message: Message, context_id: str | None = None) -> Task:
        task = Task(
            id=str(uuid.uuid4()),
            context_id=context_id or str(uuid.uuid4()),
            status=TaskStatus(state=TaskState.submitted, timestamp=_now_iso()),
            history=[message],
        )
        self._tasks[task.id] = task
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

        self._update_task_status(task, TaskState.working)

        final_answer = ""
        async for event in self.agent.astream(user_message, ctx=ctx):
            if event.is_final_response():
                final_answer = event.text

        # Build agent response message
        agent_msg = Message(
            role=Role.agent,
            parts=[TextPart(text=final_answer)],
            task_id=task.id,
            context_id=task.context_id,
        )

        # Add artifact
        artifact = Artifact(parts=[TextPart(text=final_answer)])
        task.artifacts = [artifact]

        if task.history is not None:
            task.history.append(agent_msg)

        self._update_task_status(task, TaskState.completed, agent_message=agent_msg)

    # ------------------------------------------------------------------
    # JSON-RPC handlers
    # ------------------------------------------------------------------

    async def _handle_message_send(
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

    async def _handle_message_stream(
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

        self._update_task_status(task, TaskState.working)

        # Emit initial "working" status
        working_event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.working, timestamp=_now_iso()),
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

        # Collect final answer from task updates
        # The converter already yielded artifact-update events.
        # Now emit final "completed" status.
        self._update_task_status(task, TaskState.completed)

        completed_event = TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=TaskStatus(state=TaskState.completed, timestamp=_now_iso()),
            final=True,
        )
        yield _sse_line(request_id, completed_event)

    async def _handle_tasks_get(
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

    async def _handle_tasks_cancel(
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
        self._update_task_status(task, TaskState.canceled)
        return _jsonrpc_success(request_id, task)

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------

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

        method = rpc.method
        params = rpc.params or {}

        if method == "message/send":
            return await self._handle_message_send(params, rpc.id)
        elif method == "message/stream":
            return await self._handle_message_stream(params, rpc.id)
        elif method == "tasks/get":
            return await self._handle_tasks_get(params, rpc.id)
        elif method == "tasks/cancel":
            return await self._handle_tasks_cancel(params, rpc.id)
        else:
            return _jsonrpc_error(
                rpc.id, A2AErrorCode.METHOD_NOT_FOUND, f"Method {method!r} not found",
            )

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def as_fastapi_app(self) -> FastAPI:
        """Build and return a FastAPI application with A2A-compliant routes."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(message: Message) -> str:
    """Extract plain text from a Message's parts."""
    texts = []
    for part in message.parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
    return " ".join(texts) if texts else ""


def _jsonrpc_success(request_id: Any, result: Any) -> JSONResponse:
    if hasattr(result, "model_dump"):
        result = result.model_dump(by_alias=True, exclude_none=True)
    resp = JSONRPCResponse(id=request_id, result=result)
    return JSONResponse(resp.model_dump(by_alias=True, exclude_none=True))


def _jsonrpc_error(request_id: Any, code: int, message: str) -> JSONResponse:
    resp = JSONRPCResponse(
        id=request_id or 0,
        error=JSONRPCError(code=code, message=message),
    )
    return JSONResponse(
        resp.model_dump(by_alias=True, exclude_none=True),
        status_code=400 if code != A2AErrorCode.INTERNAL_ERROR else 500,
    )


def _sse_line(request_id: Any, event: Any) -> str:
    if hasattr(event, "model_dump"):
        result_data = event.model_dump(by_alias=True, exclude_none=True)
    else:
        result_data = event
    resp = JSONRPCResponse(id=request_id, result=result_data)
    import json
    return f"data: {json.dumps(resp.model_dump(by_alias=True, exclude_none=True))}\n\n"
