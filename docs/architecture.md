# Architecture

```mermaid
flowchart TD
    classDef base fill:#f1f5f9,stroke:#94a3b8,stroke-dasharray:5 5,color:#334155
    classDef llm fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef composite fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef tool fill:#f3e8ff,stroke:#a855f7,color:#581c87
    classDef planner fill:#fff7ed,stroke:#f97316,color:#7c2d12
    classDef infra fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef event fill:#fce7f3,stroke:#ec4899,color:#831843

    Runner:::infra --> Session[(SessionService)]:::infra
    Runner --> Ctx[Context\nsession · state · run_config · event_callback]:::base
    Ctx --> BaseAgent:::base

    BaseAgent --> LlmAgent:::llm
    BaseAgent --> ReActAgent:::llm
    BaseAgent --> Sequential([SequentialAgent]):::composite
    BaseAgent --> Parallel([ParallelAgent]):::composite
    BaseAgent --> Loop([LoopAgent]):::composite

    LlmAgent --> Planners:::planner
    Planners --> PRA[PlanReActPlanner]:::planner
    Planners --> TP[TaskPlanner]:::planner

    LlmAgent --> Tools:::tool
    Tools --> FT[function_tool]:::tool
    Tools --> AT[AgentTool]:::tool
    Tools --> TF[make_transfer_tool]:::tool
    Tools --> EL[exit_loop_tool]:::tool
    Tools --> MCP[MCPToolAdapter]:::tool

    LlmAgent --> Events:::event
    Events --> AM[AGENT_MESSAGE\nfinal answer · tool calls · partial]:::event
    Events --> TR[TOOL_RESPONSE]:::event
    Events --> AS[AGENT_START / AGENT_END]:::event
    Events --> UM[USER_MESSAGE]:::event
```

## Request lifecycle

```mermaid
sequenceDiagram
    participant C as Caller
    participant R as Runner
    participant S as SessionService
    participant A as Agent

    C->>R: run_async(user_id, session_id, message)
    R->>S: get_session() or create_session()
    S-->>R: Session
    R->>R: build Context with session reference
    R->>S: append_event(session, user_event)
    note over S: persists USER_MESSAGE event

    R->>A: astream(message, ctx)

    loop for each yielded event
        A-->>R: Event
        R->>S: append_event(session, event)
        note over S: applies state_delta,<br/>appends to history
        R-->>C: yield Event
    end
```

## Key design decisions

- **No LangGraph** — orchestration is plain Python `asyncio` and async generators.
- **`Context.state`** is a shared mutable dict across the call tree. Use `EventActions.state_delta` to persist changes back to the session.
- **`Context.event_callback`** enables real-time event streaming from tools. `LlmAgent` injects an `asyncio.Queue`-based callback before tool execution; `AgentTool` uses it to push sub-agent events as they arrive. Any custom tool can use the same mechanism.
- **`LlmRequest` / `LlmResponse`** isolate LangChain types from the rest of the SDK. Swap the LLM provider without touching agent logic.
- **Planners are per-turn hooks**, not static prompts. They receive the live context and request so they can make dynamic decisions each turn.

## Custom session backend

```python
from langchain_adk.sessions import BaseSessionService, Session

class RedisSessionService(BaseSessionService):
    async def create_session(self, *, app_name, user_id, state=None, session_id=None) -> Session: ...
    async def get_session(self, *, app_name, user_id, session_id) -> Session | None: ...
    async def update_session(self, session_id, *, state) -> Session: ...
    async def delete_session(self, session_id) -> None: ...
    async def list_sessions(self, *, app_name, user_id) -> list[Session]: ...
```

## Custom planner

```python
from langchain_adk import BasePlanner, ReadonlyContext, LlmRequest

class MyPlanner(BasePlanner):
    def build_planning_instruction(self, ctx: ReadonlyContext, request: LlmRequest) -> str | None:
        return "Always verify your answer before responding."
```

Pass it to any `LlmAgent` via `planner=MyPlanner()`.
