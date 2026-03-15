# Agents

All agents inherit from `BaseAgent` and implement a single method:

```python
async def astream(self, input: str, *, ctx: Context) -> AsyncIterator[Event]:
    ...
```

The `_run_with_callbacks()` wrapper fires `before_agent_callback` / `after_agent_callback` hooks and emits `AGENT_START` / `AGENT_END` events around `astream()`.

```mermaid
flowchart LR
    classDef base fill:#f1f5f9,stroke:#94a3b8,stroke-dasharray:5 5,color:#334155
    classDef llm fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef composite fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef infra fill:#fef9c3,stroke:#eab308,color:#713f12

    BaseAgent:::base

    BaseAgent --> LlmAgent:::llm
    BaseAgent --> ReActAgent:::llm
    BaseAgent --> SequentialAgent:::composite
    BaseAgent --> ParallelAgent:::composite
    BaseAgent --> LoopAgent:::composite

    Runner:::infra --> BaseAgent
    Runner --> SessionService:::infra

    LlmAgent -. tools .-> Tools(["function_tool · AgentTool\ntransfer · exit_loop · MCP"])
    LlmAgent -. planner .-> Planners(["PlanReActPlanner\nTaskPlanner"])
```

## LlmAgent

The primary agent. Uses LangChain `BaseChatModel` with a manual tool-call loop.

```mermaid
sequenceDiagram
    participant C as Caller
    participant A as LlmAgent
    participant P as Planner
    participant L as LLM
    participant T as Tool

    C->>A: astream(input, ctx)
    A-->>C: Event(AGENT_START)

    loop ReAct loop (max_iterations)
        A->>P: build_planning_instruction(ctx, request)
        P-->>A: instruction string
        A->>L: llm.bind_tools().invoke(messages)
        L-->>A: AIMessage

        alt tool_calls present
            A-->>C: Event(AGENT_MESSAGE, has_tool_calls=True)
            A->>T: tool._arun(args)
            T-->>A: result
            A-->>C: Event(TOOL_RESPONSE)
        else no tool_calls
            A-->>C: Event(AGENT_MESSAGE, final response)
            note over A: loop exits
        end
    end

    A-->>C: Event(AGENT_END)
```

```python
from langchain_adk import LlmAgent

agent = LlmAgent(
    name="MyAgent",
    llm=llm,
    tools=[search_tool, calculator_tool],
    instructions="You are a research assistant.",   # or a Callable[[ReadonlyContext], str]
    description="Searches and calculates things.",
    planner=my_planner,         # optional BasePlanner
    output_schema=MySchema,     # optional: force structured output
    max_iterations=10,
    before_model_callback=None,
    after_model_callback=None,
    before_tool_callback=None,
    after_tool_callback=None,
)
```

### Dynamic instructions

The `instructions` parameter accepts either a plain string or an instruction provider — a callable that receives a `ReadonlyContext` and returns a string:

```python
def my_instructions(ctx: ReadonlyContext) -> str:
    user_name = ctx.state.get("user_name", "user")
    return f"You are helping {user_name}. Be concise."

agent = LlmAgent(name="Agent", llm=llm, instructions=my_instructions)
```

## ReActAgent

A structured-reasoning variant that forces the LLM to emit explicit thought steps via `with_structured_output()` before acting.

```python
from langchain_adk import ReActAgent

agent = ReActAgent(
    name="ThinkingAgent",
    llm=llm,
    tools=[search_tool],
    max_iterations=10,
)
```

Yields events with types: `AGENT_START` -> `AGENT_MESSAGE` (thoughts/actions) -> `TOOL_RESPONSE` (observations) -> ... -> `AGENT_MESSAGE` (final answer) -> `AGENT_END`.
