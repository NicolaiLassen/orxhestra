# Runner & Sessions

`Runner` is the main entry point for session-managed execution. It wires an agent, a session service, and the invocation context together.

```python
from langchain_adk import Runner, InMemorySessionService, RunConfig, StreamingMode

runner = Runner(
    agent=agent,
    app_name="my-app",
    session_service=InMemorySessionService(),
)

# Non-streaming (default)
async for event in runner.run_async(
    user_id="user-1",
    session_id="session-abc",
    new_message="Hello!",
):
    ...

# SSE streaming
async for event in runner.run_async(
    user_id="user-1",
    session_id="session-abc",
    new_message="Hello!",
    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
):
    if isinstance(event, FinalAnswerEvent) and event.partial:
        print(event.answer, end="", flush=True)
```

`Runner` automatically:

1. Fetches or creates the session
2. Builds an `InvocationContext` from the session state
3. Persists every event to the session via `append_event()`
4. Applies `EventActions.state_delta` to the session state

## Using sessions directly

```python
from langchain_adk import InMemorySessionService

svc = InMemorySessionService()
session = await svc.create_session(app_name="demo", user_id="user-1")

# All sessions for a user
sessions = await svc.list_sessions(app_name="demo", user_id="user-1")

# Delete
await svc.delete_session(session.id)
```

!!! tip "Custom backends"
    Implement `BaseSessionService` to back sessions with any database. See [Architecture](../architecture.md#custom-session-backend) for an example.
