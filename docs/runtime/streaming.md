# Streaming

All agents stream by default — `astream()` yields `Event` objects as they occur:

```python
from langchain_adk.events.event import Event, EventType

async for event in agent.astream("Write about distributed systems"):
    if event.type == EventType.AGENT_MESSAGE and event.partial:
        # Token-by-token streaming
        print(event.text, end="", flush=True)
    elif event.is_final_response():
        print(f"\n[DONE] {event.text}")
```

Partial events are suppressed automatically if the LLM decides to call a tool instead of answering — only real text chunks are streamed.

## Sub-agent streaming via AgentTool

When using `AgentTool`, sub-agent events stream through the parent in real-time via `ctx.event_callback`. Events carry `branch` and `agent_name` fields so you can distinguish which sub-agent is producing output:

```python
from langchain_adk import LlmAgent
from langchain_adk.tools.agent_tool import AgentTool

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
        # Event from a sub-agent (e.g. branch="WeatherAgent")
        print(f"  [{event.agent_name}] {event.text}", end="")
    elif event.is_final_response():
        print(f"\nFinal: {event.text}")
```

Sub-agents run in parallel when the LLM calls multiple tools at once — their events are interleaved in the stream as they arrive.

## How it works

1. Before tool execution, `LlmAgent` creates an `asyncio.Queue` and sets `ctx.event_callback = queue.put_nowait` on the tool context.
2. `AgentTool` calls `ctx.event_callback(event)` for each child event as it streams.
3. `LlmAgent` yields events from the queue concurrently while tools run, so events appear in real-time.
4. The `event_callback` propagates through `ctx.derive()`, so nested sub-agents also push events up to the root.

Any custom tool can use `ctx.event_callback` to push events — it's not limited to `AgentTool`.

## With Runner

```python
async for event in runner.astream(
    user_id="user-1",
    session_id="session-1",
    new_message="Write me a long essay about distributed systems.",
):
    if event.is_final_response():
        print(f"\n[DONE] {event.text}")
    elif event.type == EventType.AGENT_MESSAGE and event.partial:
        print(event.text, end="", flush=True)
```
