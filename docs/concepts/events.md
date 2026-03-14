# Events

Every agent yields a stream of typed `Event` objects:

| Event type | When emitted | Key fields |
|---|---|---|
| `AgentStartEvent` | Start of `run_with_callbacks()` | `agent_name` |
| `AgentEndEvent` | End of `run_with_callbacks()` | `agent_name` |
| `ThoughtEvent` | ReActAgent reasoning step | `thought`, `scratchpad` |
| `ActionEvent` | ReActAgent action decision | `action`, `action_input` |
| `ObservationEvent` | ReActAgent tool result | `observation`, `tool_name` |
| `ToolCallEvent` | LlmAgent tool invocation | `tool_name`, `tool_input`, `llm_response` |
| `ToolResultEvent` | Tool execution result | `tool_name`, `result`, `error` |
| `FinalAnswerEvent` | Agent's final response | `answer`, `scratchpad`, `llm_response`, `partial` |
| `ErrorEvent` | Unhandled exception | `message`, `exception_type` |

## EventActions

Events carry `EventActions` for side-effects:

```python
class EventActions(BaseModel):
    state_delta: dict[str, Any] = {}     # merged into session state
    transfer_to_agent: str | None = None # trigger agent handoff
    escalate: bool | None = None         # stop parent LoopAgent
    skip_summarization: bool | None = None
    end_of_agent: bool | None = None
    compaction: EventCompaction | None = None
```

## LlmResponse

`FinalAnswerEvent` and `ToolCallEvent` carry an `llm_response: LlmResponse` field with token usage and model version:

```python
event.llm_response.input_tokens
event.llm_response.output_tokens
event.llm_response.model_version
```
