# Streaming

Enable SSE streaming to get incremental text as the LLM generates it:

```python
from langchain_adk import AgentConfig, StreamingMode
from langchain_adk.events.event import Event, EventType

async for event in runner.run_async(
    user_id="user-1",
    session_id="session-1",
    new_message="Write me a long essay about distributed systems.",
    config=AgentConfig(streaming_mode=StreamingMode.SSE),
):
    if event.is_final_response():
        # Final complete event
        print(f"\n[DONE] tokens: {event.llm_response.output_tokens}")
    elif event.type == EventType.AGENT_MESSAGE and event.partial:
        # Stream chunk to the client (WebSocket, SSE endpoint, etc.)
        print(event.text, end="", flush=True)
```

Partial events are suppressed automatically if the LLM decides to call a tool instead of answering — only real text chunks are streamed.
