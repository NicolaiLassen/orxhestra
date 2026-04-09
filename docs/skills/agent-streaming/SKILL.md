---
name: agent-streaming
description: Stream events from orxhestra agents including token-by-token output, sub-agent events via AgentTool, and Runner streaming.
---

# Agent Streaming

All agents stream via `astream()`, yielding `Event` objects.

## Basic streaming

```python
from orxhestra.events.event import Event, EventType

async for event in agent.astream("Write about distributed systems"):
    if event.type == EventType.AGENT_MESSAGE and event.partial:
        print(event.text, end="", flush=True)
    elif event.is_final_response():
        print(f"\n[DONE] {event.text}")
```

## Sub-agent streaming via AgentTool

Sub-agent events stream through the parent in real-time. Events carry `branch` and `agent_name` fields.

```python
from orxhestra import LlmAgent
from orxhestra.tools.agent_tool import AgentTool

weather_agent = LlmAgent(name="WeatherAgent", llm=llm, tools=[get_weather])
travel_agent = LlmAgent(name="TravelAgent", llm=llm, tools=[get_attractions])

planner = LlmAgent(
    name="TripPlanner",
    llm=llm,
    tools=[AgentTool(weather_agent), AgentTool(travel_agent)],
    instructions="Use the sub-agents to plan a trip.",
)

async for event in planner.astream("Plan a trip to Copenhagen"):
    if event.branch:
        print(f"  [{event.agent_name}] {event.text}", end="")
    elif event.is_final_response():
        print(f"\nFinal: {event.text}")
```

## How it works

1. `LlmAgent` creates an `asyncio.Queue` and sets `ctx.event_callback = queue.put_nowait`.
2. `AgentTool` calls `ctx.event_callback(event)` for each child event.
3. Events yield from the queue concurrently while tools run.
4. `event_callback` propagates through `ctx.derive()` for nested sub-agents.

Any custom tool can use `ctx.event_callback` to push events.

## With Runner

```python
async for event in runner.astream(
    user_id="user-1",
    session_id="session-1",
    new_message="Write me a long essay.",
):
    if event.is_final_response():
        print(f"\n[DONE] {event.text}")
    elif event.type == EventType.AGENT_MESSAGE and event.partial:
        print(event.text, end="", flush=True)
```
