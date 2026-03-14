# Streaming

Enable SSE streaming to get incremental text as the LLM generates it:

```python
from langchain_adk import RunConfig, StreamingMode

async for event in runner.run_async(
    user_id="user-1",
    session_id="session-1",
    new_message="Write me a long essay about distributed systems.",
    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
):
    if isinstance(event, FinalAnswerEvent):
        if event.partial:
            # Stream chunk to the client (WebSocket, SSE endpoint, etc.)
            print(event.answer, end="", flush=True)
        else:
            # Final complete event
            print(f"\n[DONE] tokens: {event.llm_response.output_tokens}")
```

Partial events are suppressed automatically if the LLM decides to call a tool instead of answering — only real text chunks are streamed.
