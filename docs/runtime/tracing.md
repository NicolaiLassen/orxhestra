# Tracing (Langfuse, LangSmith, etc.)

`AgentConfig` mirrors LangChain's `RunnableConfig` fields — pass `callbacks`, `tags`, `metadata`, `run_name` directly. The entire agent run is wrapped in a single parent trace with all child operations nested automatically.

```python
from langfuse.langchain import CallbackHandler
from langchain_adk import AgentConfig

run_config = AgentConfig(
    callbacks=[CallbackHandler()],  # Langfuse, LangSmith, or any BaseCallbackHandler
    tags=["production", "user-facing"],
    metadata={"user_id": "u-123"},
    run_name="MyAgent",
)

async for event in runner.run_async(
    user_id="user-1",
    session_id="session-1",
    new_message="Hello!",
    config=run_config,
):
    ...
```

!!! tip "Environment variables"
    ```bash
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
    ```

## What gets traced

- Each agent run as a single parent trace (named after the agent)
- All LLM calls (`ainvoke` / `astream`) as child spans with token usage
- All tool executions as child spans with inputs/outputs
- Structured output fallback calls
- Works with composite agents — each sub-agent creates its own nested trace

!!! note "A2A boundaries"
    A2A is HTTP, so callbacks can't cross the wire. Each A2A server should create its own handler. Link traces by passing a `trace_id` in task metadata.
