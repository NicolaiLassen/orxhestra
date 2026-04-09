---
name: agent-callbacks
description: Add callbacks to orxhestra agents for logging, monitoring, and error handling. Covers model and tool callbacks.
---

# Agent Callbacks

LlmAgent supports before/after hooks at the model and tool level.

## Model callbacks

```python
from orxhestra import LlmAgent, Context
from orxhestra.models.llm_request import LlmRequest
from orxhestra.models.llm_response import LlmResponse

async def log_before_model(ctx: Context, request: LlmRequest) -> None:
    print(f"Calling LLM with {len(request.messages)} messages")
    print(f"Tools available: {[t.name for t in request.tools]}")

async def log_after_model(ctx: Context, response: LlmResponse) -> None:
    print(f"LLM responded: {response.text[:100]}")
    if response.has_tool_calls:
        print(f"Tool calls: {[tc['name'] for tc in response.tool_calls]}")

async def handle_error(
    ctx: Context, request: LlmRequest, error: Exception
) -> LlmResponse | None:
    print(f"LLM error: {error}")
    return None  # push error event; or return LlmResponse to recover

agent = LlmAgent(
    name="monitored",
    llm=llm,
    before_model_callback=log_before_model,
    after_model_callback=log_after_model,
    on_model_error_callback=handle_error,
)
```

## Tool callbacks

```python
from typing import Any

async def log_tool_start(ctx: Context, tool_name: str, tool_args: dict) -> None:
    print(f"Calling tool: {tool_name}({tool_args})")

async def log_tool_end(ctx: Context, tool_name: str, result: Any) -> None:
    print(f"Tool {tool_name} returned: {str(result)[:100]}")

agent = LlmAgent(
    name="tracked",
    llm=llm,
    tools=[search],
    before_tool_callback=log_tool_start,
    after_tool_callback=log_tool_end,
)
```

## AgentTool callbacks

Intercept events from child agents when using AgentTool.

```python
from orxhestra.tools import AgentTool

def before_agent(ctx, agent):
    print(f"Delegating to sub-agent: {agent.name}")

def after_agent(ctx, agent, events):
    print(f"Sub-agent {agent.name} produced {len(events)} events")

tool = AgentTool(
    agent=researcher,
    before_agent_callback=before_agent,
    after_agent_callback=after_agent,
)
```

## Tracing with Langfuse

```python
from langfuse.callback import CallbackHandler
from orxhestra import AgentConfig

handler = CallbackHandler(public_key="...", secret_key="...", host="...")

config = AgentConfig(
    configurable={"callbacks": [handler]},
)

async for event in agent.astream("Hello", config=config):
    print(event)
```
