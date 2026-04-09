---
name: build-agent
description: Build an LLM agent with orxhestra. Use when creating a new agent, setting up LlmAgent or ReActAgent, or wiring tools to an agent.
---

# Building Agents with orxhestra

All agents extend `BaseAgent` and implement `astream(input, *, ctx)` returning `AsyncIterator[Event]`.

## LlmAgent — Standard tool-calling agent

```python
from orxhestra import LlmAgent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
async def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = LlmAgent(
    name="assistant",
    llm=ChatOpenAI(model="gpt-5.4"),
    tools=[search],
    instructions="You are a helpful assistant.",
    max_iterations=10,
)

# Async streaming
async for event in agent.astream("Hello"):
    if event.is_final_response():
        print(event.text)

# Sync convenience
result = agent.invoke("Hello")
print(result.text)
```

### Key parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique agent name |
| `llm` | `BaseChatModel` | Any LangChain chat model |
| `tools` | `list[BaseTool]` | Tools available to the agent |
| `instructions` | `str \| Callable` | System prompt (static or dynamic) |
| `planner` | `BasePlanner` | Optional planning strategy |
| `output_schema` | `type` | Optional Pydantic model for structured output |
| `max_iterations` | `int` | Max tool-call loop iterations (default: 10) |

### Dynamic instructions

```python
async def dynamic_instructions(ctx):
    return f"You are helping user in session {ctx.session_id}."

agent = LlmAgent(
    name="dynamic",
    llm=llm,
    instructions=dynamic_instructions,
)
```

## ReActAgent — Structured reasoning

Uses `with_structured_output()` to enforce a typed `ReActStep` at every iteration. Extends LlmAgent so it inherits instructions, planners, skills, and callbacks.

```python
from orxhestra import ReActAgent

agent = ReActAgent(
    name="reasoner",
    llm=ChatOpenAI(model="gpt-5.4"),
    tools=[search],
    instructions="Think step by step.",  # appended to ReAct prompt
    max_iterations=10,
)
```

## Running with sessions (Runner)

```python
from orxhestra import Runner, InMemorySessionService

runner = Runner(
    agent=agent,
    app_name="my-app",
    session_service=InMemorySessionService(),
)

session = await runner.session_service.create_session(app_name="my-app")

async for event in runner.run(
    user_message="Hello!",
    session_id=session.id,
):
    if event.is_final_response():
        print(event.text)
```
