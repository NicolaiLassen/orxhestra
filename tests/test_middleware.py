"""Tests for the Middleware protocol, stack, and Runner integration."""

from __future__ import annotations

from typing import Any

import pytest

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.agents.callbacks import LlmAgentCallbacks
from orxhestra.events.event import Event, EventType
from orxhestra.middleware.base import BaseMiddleware, Middleware, ToolCall
from orxhestra.middleware.callback import CallbackMiddleware
from orxhestra.middleware.logging import LoggingMiddleware
from orxhestra.middleware.stack import MiddlewareStack
from orxhestra.runner import Runner
from orxhestra.sessions.in_memory_session_service import InMemorySessionService

# ── Helpers ──────────────────────────────────────────────────────────

class _RecorderAgent(BaseAgent):
    """Minimal agent that yields a pre-seeded list of events."""

    _events_out: list[Event]

    def __init__(self, name: str, events: list[Event]) -> None:
        super().__init__(name=name)
        object.__setattr__(self, "_events_out", events)

    async def astream(self, new_message, ctx):  # type: ignore[override]
        for e in self._events_out:
            yield e


def _make_event(text: str, session_id: str = "s1") -> Event:
    from orxhestra.models.part import Content
    return Event(
        type=EventType.AGENT_MESSAGE,
        session_id=session_id,
        content=Content.from_text(text),
    )


async def _run(runner: Runner) -> list[Event]:
    out: list[Event] = []
    async for e in runner.astream(
        user_id="u1", session_id="s1", new_message="hi",
    ):
        out.append(e)
    return out


# ── MiddlewareStack direct tests ─────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_stack_is_falsy():
    stack = MiddlewareStack()
    assert not stack
    assert len(stack) == 0


@pytest.mark.asyncio
async def test_stack_extend_returns_new_instance():
    class _M(BaseMiddleware): ...

    stack = MiddlewareStack()
    m = _M()
    new_stack = stack.extend([m])
    assert new_stack is not stack
    assert len(new_stack) == 1


@pytest.mark.asyncio
async def test_stack_on_event_threads_transforms():
    class _Append(BaseMiddleware):
        def __init__(self, suffix: str) -> None:
            self.suffix = suffix

        async def on_event(self, ctx, event):
            from orxhestra.models.part import Content
            return Event(
                type=event.type,
                session_id=event.session_id,
                content=Content.from_text((event.text or "") + self.suffix),
            )

    stack = MiddlewareStack([_Append("-A"), _Append("-B")])
    e = _make_event("x")
    out = await stack.on_event(None, e)  # type: ignore[arg-type]
    assert out.text == "x-A-B"


@pytest.mark.asyncio
async def test_stack_on_event_can_drop():
    class _Drop(BaseMiddleware):
        async def on_event(self, ctx, event):
            return None

    stack = MiddlewareStack([_Drop()])
    out = await stack.on_event(None, _make_event("x"))  # type: ignore[arg-type]
    assert out is None


@pytest.mark.asyncio
async def test_stack_wrap_tool_onion_order():
    order: list[str] = []

    class _Outer(BaseMiddleware):
        async def wrap_tool(self, ctx, call, call_next):
            order.append("outer-before")
            r = await call_next(call)
            order.append("outer-after")
            return r

    class _Inner(BaseMiddleware):
        async def wrap_tool(self, ctx, call, call_next):
            order.append("inner-before")
            r = await call_next(call)
            order.append("inner-after")
            return r

    stack = MiddlewareStack([_Outer(), _Inner()])

    async def _exec(call: ToolCall) -> str:
        order.append("exec")
        return "ok"

    result = await stack.wrap_tool(None, ToolCall(name="t", args={}), _exec)  # type: ignore[arg-type]
    assert result == "ok"
    assert order == [
        "outer-before", "inner-before", "exec", "inner-after", "outer-after",
    ]


@pytest.mark.asyncio
async def test_stack_wrap_tool_empty_stack_passthrough():
    stack = MiddlewareStack()

    async def _exec(call: ToolCall) -> str:
        return "bare"

    result = await stack.wrap_tool(None, ToolCall(name="t", args={}), _exec)  # type: ignore[arg-type]
    assert result == "bare"


@pytest.mark.asyncio
async def test_stack_after_invoke_reverses_order():
    order: list[str] = []

    class _M(BaseMiddleware):
        def __init__(self, name: str) -> None:
            self.name = name

        async def before_invoke(self, ctx):
            order.append(f"before-{self.name}")

        async def after_invoke(self, ctx, error=None):
            order.append(f"after-{self.name}")

    stack = MiddlewareStack([_M("A"), _M("B")])
    await stack.before_invoke(None)  # type: ignore[arg-type]
    await stack.after_invoke(None, None)  # type: ignore[arg-type]
    assert order == ["before-A", "before-B", "after-B", "after-A"]


@pytest.mark.asyncio
async def test_stack_tolerates_partial_implementations():
    """A Middleware that only defines on_event must not break other hooks."""
    class _Partial:
        async def on_event(self, ctx, event):
            return event

    stack = MiddlewareStack([_Partial()])  # type: ignore[list-item]
    # Should not raise even though before_invoke is not defined.
    await stack.before_invoke(None)  # type: ignore[arg-type]
    await stack.after_invoke(None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_before_model_threads_value():
    class _M(BaseMiddleware):
        async def before_model(self, ctx, request):
            # Return unchanged; middleware must not force mutation.
            return request

    stack = MiddlewareStack([_M()])
    sentinel = object()
    out = await stack.before_model(None, sentinel)  # type: ignore[arg-type]
    assert out is sentinel


# ── Runner integration ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_runner_without_middleware_is_unchanged():
    """Empty middleware stack must produce identical behavior."""
    events = [_make_event("hello", "s1")]
    agent = _RecorderAgent("root", events)
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="test", session_service=svc)
    out = await _run(runner)
    # User + agent event
    assert len(out) == 1
    assert out[0].text == "hello"


@pytest.mark.asyncio
async def test_runner_fires_before_and_after_invoke():
    calls: list[str] = []

    class _M(BaseMiddleware):
        async def before_invoke(self, ctx):
            calls.append(f"before:{ctx.agent_name}")

        async def after_invoke(self, ctx, error=None):
            calls.append(f"after:{error}")

    agent = _RecorderAgent("root", [_make_event("hi")])
    runner = Runner(
        agent=agent,
        app_name="test",
        session_service=InMemorySessionService(),
        middleware=[_M()],
    )
    await _run(runner)
    assert calls == ["before:root", "after:None"]


@pytest.mark.asyncio
async def test_runner_on_event_can_transform():
    class _Upper(BaseMiddleware):
        async def on_event(self, ctx, event):
            from orxhestra.models.part import Content
            return Event(
                type=event.type,
                session_id=event.session_id,
                content=Content.from_text((event.text or "").upper()),
            )

    agent = _RecorderAgent("root", [_make_event("hi")])
    runner = Runner(
        agent=agent,
        app_name="test",
        session_service=InMemorySessionService(),
        middleware=[_Upper()],
    )
    out = await _run(runner)
    assert out[0].text == "HI"


@pytest.mark.asyncio
async def test_runner_on_event_can_drop():
    class _Drop(BaseMiddleware):
        async def on_event(self, ctx, event):
            return None

    agent = _RecorderAgent("root", [_make_event("a"), _make_event("b")])
    runner = Runner(
        agent=agent,
        app_name="test",
        session_service=InMemorySessionService(),
        middleware=[_Drop()],
    )
    out = await _run(runner)
    assert out == []


@pytest.mark.asyncio
async def test_runner_after_invoke_sees_error():
    class _BoomAgent(BaseAgent):
        async def astream(self, new_message, ctx):  # type: ignore[override]
            raise RuntimeError("boom")
            yield  # pragma: no cover

    captured: list[Any] = []

    class _M(BaseMiddleware):
        async def after_invoke(self, ctx, error=None):
            captured.append(error)

    agent = _BoomAgent(name="root")
    runner = Runner(
        agent=agent,
        app_name="test",
        session_service=InMemorySessionService(),
        middleware=[_M()],
    )
    with pytest.raises(RuntimeError, match="boom"):
        await _run(runner)
    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)


# ── CallbackMiddleware backward compat ───────────────────────────────

@pytest.mark.asyncio
async def test_callback_middleware_forwards_before_after_model():
    from orxhestra.agents.invocation_context import InvocationContext
    from orxhestra.models.llm_request import LlmRequest
    from orxhestra.models.llm_response import LlmResponse

    seen: list[str] = []

    async def _before(ctx, req):
        seen.append("before")

    async def _after(ctx, resp):
        seen.append("after")

    cb = LlmAgentCallbacks(before_model=_before, after_model=_after)
    mw = CallbackMiddleware(cb)

    ctx = InvocationContext(
        session_id="s", agent_name="a", app_name="app",
    )
    req = LlmRequest(messages=[])
    await mw.before_model(ctx, req)

    resp = LlmResponse()
    await mw.after_model(ctx, resp)

    assert seen == ["before", "after"]


@pytest.mark.asyncio
async def test_callback_middleware_wraps_tool_calls():
    from orxhestra.agents.invocation_context import InvocationContext

    seen: list[str] = []

    async def _before(ctx, name, args):
        seen.append(f"before:{name}")

    async def _after(ctx, name, result):
        seen.append(f"after:{name}:{result}")

    cb = LlmAgentCallbacks(before_tool=_before, after_tool=_after)
    mw = CallbackMiddleware(cb)

    ctx = InvocationContext(
        session_id="s", agent_name="a", app_name="app",
    )

    async def _exec(call: ToolCall) -> str:
        seen.append("exec")
        return "42"

    result = await mw.wrap_tool(ctx, ToolCall(name="t", args={}), _exec)
    assert result == "42"
    assert seen == ["before:t", "exec", "after:t:42"]


# ── LoggingMiddleware smoke test ─────────────────────────────────────

@pytest.mark.asyncio
async def test_logging_middleware_does_not_raise(caplog):
    import logging

    caplog.set_level(logging.DEBUG, logger="orxhestra.middleware.logging")
    agent = _RecorderAgent("root", [_make_event("hi")])
    runner = Runner(
        agent=agent,
        app_name="test",
        session_service=InMemorySessionService(),
        middleware=[LoggingMiddleware()],
    )
    out = await _run(runner)
    assert len(out) == 1


# ── Middleware protocol checks ───────────────────────────────────────

def test_base_middleware_satisfies_protocol():
    assert isinstance(BaseMiddleware(), Middleware)


def test_callback_middleware_satisfies_protocol():
    cb = LlmAgentCallbacks()
    assert isinstance(CallbackMiddleware(cb), Middleware)
