# Context

`Context` is the runtime state passed through every agent in the call tree. It carries the session binding, a mutable shared state dict, run config, and an optional event callback for real-time event pushing.

```python
from langchain_adk import Context

ctx = Context(
    session_id="session-1",
    user_id="user-1",
    app_name="my-app",
    agent_name="RootAgent",
    state={"user_name": "Alice"},
    session=session,  # provides access to conversation history via session.events
)
```

## Fields

| Field | Type | Description |
|---|---|---|
| `invocation_id` | `str` | Unique ID for this invocation (auto-generated) |
| `session_id` | `str` | The session this invocation belongs to |
| `user_id` | `str` | The user who initiated the session |
| `app_name` | `str` | The application running the agent |
| `agent_name` | `str` | The name of the currently executing agent |
| `branch` | `str` | Dot-separated path for nested execution (e.g. `"root.child"`) |
| `state` | `dict` | Mutable key-value store shared across the call tree |
| `session` | `Session` | Session with conversation history and persisted state |
| `run_config` | `dict` | LangChain `RunnableConfig` for callbacks/tracing |
| `memory_service` | `Any` | Optional memory service for long-term recall |
| `event_callback` | `Callable` | Optional callback for pushing events to the parent stream |

## Derived contexts

Sub-agents receive a **derived** context with their own `agent_name` and `branch` for isolation, while sharing the same `state` reference and `event_callback`:

```python
child_ctx = ctx.derive(agent_name="ChildAgent", branch_suffix="child")
# child_ctx.branch == "RootAgent.child"
# child_ctx.state is ctx.state  <- shared reference
# child_ctx.event_callback is ctx.event_callback  <- propagated
```

All orchestrators (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) and `AgentTool` call `derive()` automatically. `AgentTool` additionally calls `clear_session()` to give the child a fresh context without the parent's conversation history.

## clear_session()

Returns a copy of the context with an empty session — same IDs but no conversation history. Used by `AgentTool` to give child agents a clean slate:

```python
child_ctx = ctx.derive(agent_name="ChildAgent").clear_session()
# child_ctx.session has no events — child starts fresh
# child_ctx.state is still shared with parent
```

## event_callback

`event_callback` is an optional callable that tools can use to push events to the parent agent's event stream in real-time. `LlmAgent` sets this automatically before tool execution using an `asyncio.Queue`, so `AgentTool` (and any custom tool) can emit events while running:

```python
# Inside a custom tool's _arun():
if ctx.event_callback is not None:
    ctx.event_callback(my_event)  # pushed to parent stream immediately
```

The callback propagates through `derive()`, so nested sub-agents also push events up to the root.

## ReadonlyContext & CallbackContext

```python
from langchain_adk import ReadonlyContext, CallbackContext

# Planners and instruction providers receive ReadonlyContext
# state is exposed as MappingProxyType — no accidental mutations
def instructions(ctx: ReadonlyContext) -> str:
    return f"Help {ctx.state['user_name']}."

# Callbacks receive CallbackContext — state is mutable, actions available
async def after_model(ctx: CallbackContext, response: LlmResponse) -> None:
    ctx.state["last_model"] = response.model_version
    ctx.actions.state_delta["last_model"] = response.model_version
```
