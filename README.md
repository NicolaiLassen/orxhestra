<p align="center">
  <img src="docs/assets/logo.svg" width="120" alt="langchain-adk logo">
</p>

<h1 align="center">langchain-adk</h1>

<p align="center">
  <strong>A LangChain-powered Agent Development Kit</strong><br>
  Async event-streaming agents, composable hierarchies, session management, planners, skills, MCP and A2A integration.
</p>

<p align="center">
  <a href="https://pypi.org/project/langchain-adk/"><img src="https://img.shields.io/pypi/v/langchain-adk" alt="PyPI"></a>
  <a href="https://pypi.org/project/langchain-adk/"><img src="https://img.shields.io/pypi/pyversions/langchain-adk" alt="Python"></a>
  <a href="https://github.com/NicolaiLassen/langchain-adk/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NicolaiLassen/langchain-adk" alt="License"></a>
  <a href="https://nicolailassen.github.io/langchain-adk/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
</p>

> **Beta** — This project is under heavy development and may be subject to breaking changes between minor versions. Pin your dependency version if you need stability.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Agents](#agents)
  - [Events](#events)
  - [Runner & Sessions](#runner--sessions)
  - [InvocationContext](#invocationcontext)
- [Tools](#tools)
  - [Long-Running Tools](#long-running-tools)
  - [Tool Confirmation](#tool-confirmation)
- [Agents](#composite-agents)
  - [SequentialAgent](#sequentialagent)
  - [ParallelAgent](#parallelagent)
  - [LoopAgent](#loopagent)
- [Orchestration](#orchestration)
  - [Planners](#planners)
  - [Skills](#skills)
  - [Prompts](#prompts)
- [Runtime](#runtime)
  - [Streaming](#streaming)
  - [Callbacks](#callbacks)
  - [Tracing](#tracing-langfuse-langsmith-etc)
  - [Structured Output](#structured-output)
  - [Human-in-the-Loop](#human-in-the-loop)
- [Integrations](#integrations)
  - [Agent-to-Agent (A2A)](#agent-to-agent-a2a-server)
  - [MCP](#mcp-integration)
- [Examples](#examples)
- [Architecture](#architecture)
- [Development](#development)

---

## Overview

`langchain-adk` gives you a structured way to build production-quality agents on top of any LangChain-compatible LLM. The core ideas:

- **Async event stream**: every agent is an `async def run()` that yields typed `Event` objects — thoughts, tool calls, results, final answers.
- **Composable hierarchy**: agents nest freely. Wrap a sub-agent as a tool (`AgentTool`), chain them (`SequentialAgent`), run them in parallel (`ParallelAgent`), or loop until done (`LoopAgent`).
- **Manual tool-call loop**: `LlmAgent` drives its own ReAct loop using `llm.bind_tools()` — no LangGraph, no hidden graphs.
- **Planners**: inject per-turn planning instructions and post-process responses before the agent acts.
- **Sessions**: pluggable `BaseSessionService` persists every event and state delta automatically via the `Runner`.
- **First-class streaming**: `RunConfig(streaming_mode=StreamingMode.SSE)` switches the LLM call to `astream()` and yields `partial=True` events for real-time UIs.

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

---

## Installation

```bash
pip install langchain-adk

# or with uv
uv add langchain-adk
```

Install with an LLM provider:

```bash
pip install langchain-adk[openai]      # ChatOpenAI
pip install langchain-adk[anthropic]   # ChatAnthropic
pip install langchain-adk[google]      # ChatGoogleGenerativeAI
pip install langchain-adk[mcp]         # MCP integration (fastMCP)
```

**Python >= 3.10 required.**

---

## Quick Start

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from langchain_adk import LlmAgent, Runner, InMemorySessionService
from langchain_adk.events.event import FinalAnswerEvent

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 22 degrees."

agent = LlmAgent(
    name="WeatherAgent",
    llm=ChatAnthropic(model="claude-3-5-haiku-latest"),
    tools=[get_weather],
    instructions="You are a helpful weather assistant.",
)

runner = Runner(
    agent=agent,
    app_name="demo",
    session_service=InMemorySessionService(),
)

async def main():
    async for event in runner.run_async(
        user_id="user-1",
        session_id="session-1",
        new_message="What's the weather in Copenhagen and Berlin?",
    ):
        if isinstance(event, FinalAnswerEvent):
            print(event.answer)

asyncio.run(main())
```

---

## Core Concepts

### Agents

All agents inherit from `BaseAgent` and implement a single method:

```python
async def run(self, input: str, *, ctx: InvocationContext) -> AsyncIterator[Event]:
    ...
```

The `run_with_callbacks()` wrapper fires `before_agent_callback` / `after_agent_callback` hooks and emits `AGENT_START` / `AGENT_END` events around `run()`.

#### LlmAgent

The primary agent. Uses LangChain `BaseChatModel` with a manual tool-call loop.

```mermaid
sequenceDiagram
    participant C as Caller
    participant A as LlmAgent
    participant P as Planner
    participant L as LLM
    participant T as Tool

    C->>A: run(input, ctx)
    A-->>C: AgentStartEvent

    loop ReAct loop (max_iterations)
        A->>P: build_planning_instruction(ctx, request)
        P-->>A: instruction string
        A->>L: llm.bind_tools().invoke(messages)
        L-->>A: AIMessage

        alt tool_calls present
            A-->>C: ToolCallEvent
            A->>T: tool._arun(args)
            T-->>A: result
            A-->>C: ToolResultEvent
        else no tool_calls
            A-->>C: FinalAnswerEvent
            note over A: loop exits
        end
    end

    A-->>C: AgentEndEvent
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

The `instructions` parameter accepts either a plain string or an instruction provider — a callable that receives a `ReadonlyContext` and returns a string. This lets you build dynamic prompts per-turn:

```python
def my_instructions(ctx: ReadonlyContext) -> str:
    user_name = ctx.state.get("user_name", "user")
    return f"You are helping {user_name}. Be concise."

agent = LlmAgent(name="Agent", llm=llm, instructions=my_instructions)
```

#### ReActAgent

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

Yields: `ThoughtEvent` -> `ActionEvent` -> `ObservationEvent` -> ... -> `FinalAnswerEvent`.

---

### Events

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

`FinalAnswerEvent` and `ToolCallEvent` carry an `llm_response: LlmResponse` field with token usage and model version:

```python
event.llm_response.input_tokens
event.llm_response.output_tokens
event.llm_response.model_version
```

---

### Runner & Sessions

`Runner` is the main entry point for session-managed execution. It wires an agent, a session service, and the invocation context together.

```python
from langchain_adk import Runner, InMemorySessionService, RunConfig, StreamingMode

runner = Runner(
    agent=agent,
    app_name="my-app",
    session_service=InMemorySessionService(),
)

# Non-streaming (default)
async for event in runner.run_async(
    user_id="user-1",
    session_id="session-abc",
    new_message="Hello!",
):
    ...

# SSE streaming — yields partial FinalAnswerEvents as text arrives
async for event in runner.run_async(
    user_id="user-1",
    session_id="session-abc",
    new_message="Hello!",
    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
):
    if isinstance(event, FinalAnswerEvent) and event.partial:
        print(event.answer, end="", flush=True)
```

`Runner` automatically:
1. Fetches or creates the session
2. Builds an `InvocationContext` from the session state
3. Persists every event to the session via `append_event()`
4. Applies `EventActions.state_delta` to the session state

#### Sessions directly

```python
from langchain_adk import InMemorySessionService

svc = InMemorySessionService()
session = await svc.create_session(app_name="demo", user_id="user-1")

# All sessions for a user
sessions = await svc.list_sessions(app_name="demo", user_id="user-1")

# Delete
await svc.delete_session(session.id)
```

Implement `BaseSessionService` to back sessions with any database.

---

### InvocationContext

`InvocationContext` is the runtime state passed through every agent in the call tree. It carries the session binding, a mutable shared state dict, and run config.

```python
from langchain_adk import InvocationContext

ctx = InvocationContext(
    session_id="session-1",
    user_id="user-1",
    app_name="my-app",
    agent_name="RootAgent",
    state={"user_name": "Alice"},
)
```

Sub-agents receive a **derived** context with their own `agent_name` and `branch` for isolation, while sharing the same `state` reference:

```python
child_ctx = ctx.derive(agent_name="ChildAgent", branch_suffix="child")
# child_ctx.branch == "RootAgent.child"
# child_ctx.state is ctx.state  <- shared reference
```

Read-only and callback views:

```python
from langchain_adk import ReadonlyContext, CallbackContext

# Planners and instruction providers receive ReadonlyContext
# state is exposed as MappingProxyType — no accidental mutations
def instructions(ctx: ReadonlyContext) -> str:
    return f"Help {ctx.state['user_name']}."

# Callbacks receive CallbackContext — state is mutable, actions available
async def after_model(ctx: CallbackContext, response: LlmResponse) -> None:
    ctx.state["last_model"] = response.model_version
    ctx.actions.state_delta["last_model"] = response.model_version
```

---

## Tools

```mermaid
flowchart LR
    classDef agent fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef tool fill:#f3e8ff,stroke:#a855f7,color:#581c87
    classDef effect fill:#fff7ed,stroke:#f97316,color:#7c2d12
    classDef external fill:#f0fdf4,stroke:#22c55e,color:#14532d

    A([LlmAgent]):::agent

    A --> FT[function_tool]:::tool
    A --> AT[AgentTool]:::tool
    A --> TT[make_transfer_tool]:::tool
    A --> EL[exit_loop_tool]:::tool
    A --> LR[LongRunningFunctionTool]:::tool
    A --> MCP[MCPToolAdapter]:::tool

    FT -->|wraps| Fn["async def fn()"]:::external
    AT -->|invokes| SA([sub-agent]):::agent
    SA -->|derive context\nbranch isolation| IC["InvocationContext\n.derive()"]:::effect
    TT -->|sets| EA1["EventActions\n.transfer_to_agent"]:::effect
    EL -->|sets| EA2["EventActions\n.escalate = True"]:::effect
    MCP -->|fetches from| MS[("MCP Server")]:::external
```

### Function tools

Wrap any async function as a LangChain `BaseTool`:

```python
from langchain_adk import function_tool

async def search_web(query: str) -> str:
    """Search the web and return results."""
    ...

tool = function_tool(search_web)
# or: function_tool(search_web, name="web_search", description="...")
```

Or use LangChain's `@tool` decorator directly — both work with `LlmAgent`.

### AgentTool — sub-agents as tools

Wrap a `BaseAgent` so it can be called as a tool by a parent agent:

```python
from langchain_adk import AgentTool

research_tool = AgentTool(research_agent)
# The parent agent can call "ResearchAgent" as a tool.
# The tool derives a child context with branch isolation automatically.
```

### Transfer tool — explicit agent handoff

```python
from langchain_adk import make_transfer_tool

transfer = make_transfer_tool([billing_agent, support_agent, tech_agent])
# The LLM can call "transfer_to_agent" with the target agent name.
# EventActions.transfer_to_agent is set; the parent routes accordingly.
```

### Exit loop tool

Signal a `LoopAgent` to stop iterating:

```python
from langchain_adk import exit_loop_tool

loop_agent = LoopAgent(
    name="RefineLoop",
    agents=[refine_agent],
    # refine_agent has exit_loop_tool in its tools list
)
```

### ToolContext

Inside a tool's `_arun()`, use `ToolContext` to read/write agent state:

```python
from langchain_adk import ToolContext

class MyStatefulTool(BaseTool):
    _ctx: ToolContext | None = None

    def inject_context(self, ctx: InvocationContext) -> None:
        self._ctx = ToolContext(ctx)

    async def _arun(self, query: str) -> str:
        self._ctx.state["last_query"] = query
        return "done"
```

`LlmAgent` automatically calls `inject_context()` on any tool that exposes it before each tool execution.

### Long-running tools

For tools that take significant time (deployments, data processing), use `LongRunningFunctionTool`. It instructs the LLM not to re-invoke the tool while pending:

```python
from langchain_adk.tools import LongRunningFunctionTool

async def deploy_service(env: str) -> str:
    """Deploy the service to the given environment."""
    await some_long_operation(env)
    return f"Deployed to {env}"

tool = LongRunningFunctionTool(deploy_service)
agent = LlmAgent(name="deployer", llm=llm, tools=[tool.as_tool()])
```

### Tool confirmation

Gate dangerous tool executions with `ToolContext.request_confirmation()`:

```python
from langchain_adk.tools import ToolContext

async def confirm_dangerous(ctx, tool_name: str, tool_args: dict) -> None:
    if tool_name in ("delete_file", "drop_table"):
        tool_ctx = ToolContext(ctx)
        tool_ctx.request_confirmation()
```

---

## Orchestration

### Planners

Planners inject planning instructions into the system prompt before each LLM call and can post-process the response.

```python
from langchain_adk import BasePlanner, ReadonlyContext, LlmRequest, LlmResponse

class MyPlanner(BasePlanner):
    def build_planning_instruction(
        self, ctx: ReadonlyContext, request: LlmRequest
    ) -> str | None:
        return "Think step by step before acting. Plan before calling tools."

    def process_planning_response(
        self, ctx: ReadonlyContext, response: LlmResponse
    ) -> LlmResponse | None:
        return None  # no post-processing needed
```

#### PlanReActPlanner

A built-in planner that enforces structured planning tags — the agent must emit a `/*PLANNING*/` block before reasoning and a `/*FINAL_ANSWER*/` block to conclude.

```python
from langchain_adk import PlanReActPlanner, LlmAgent

agent = LlmAgent(
    name="PlanningAgent",
    llm=llm,
    tools=[...],
    planner=PlanReActPlanner(),
)
```

#### TaskPlanner

Maintains a task board in `ctx.state` and injects its current status into the system prompt. Pairs with `ManageTasksTool` so the agent can create, update, complete, and list tasks.

```python
from langchain_adk import TaskPlanner, LlmAgent

planner = TaskPlanner()

agent = LlmAgent(
    name="ProjectAgent",
    llm=llm,
    tools=[planner.get_manage_tasks_tool()],
    planner=planner,
    instructions=(
        "Track your work with manage_tasks. "
        "Initialize tasks at the start. Mark each complete when done."
    ),
)
```

The agent can call `manage_tasks` with actions: `initialize`, `list`, `create`, `update`, `complete`, `remove`.

---

### Skills

Skills are reusable instruction blocks that an agent can load dynamically at runtime.

```python
from langchain_adk import Skill, InMemorySkillStore
from langchain_adk.skills.load_skill_tool import make_load_skill_tool, make_list_skills_tool

store = InMemorySkillStore([
    Skill(
        name="summarization",
        description="How to write concise summaries.",
        content="Extract the 3-5 most important points. Use bullet points. Be concise.",
    ),
    Skill(
        name="code_review",
        description="How to conduct a thorough code review.",
        content="Check correctness, readability, test coverage, and security.",
    ),
])

agent = LlmAgent(
    name="SkillfulAgent",
    llm=llm,
    tools=[
        make_list_skills_tool(store),   # lets agent discover available skills
        make_load_skill_tool(store),    # lets agent load a skill's full content
    ],
)
```

The agent calls `list_skills` to discover what's available, then `load_skill("summarization")` to get the full instruction text. Implement `BaseSkillStore` for any backend.

---

### Prompts

The prompt catalog builds structured system prompts from a `PromptContext`:

```python
from langchain_adk import PromptContext, build_system_prompt

prompt = build_system_prompt(PromptContext(
    agent_name="WriterAgent",
    goal="Write polished articles from research notes.",
    instructions="Use markdown. Structure with headings. Be thorough.",
    skills=[
        {"name": "AP Style", "content": "Follow AP style guide for all prose."}
    ],
    agents=[
        {"name": "ResearchAgent", "description": "Retrieves research on any topic."}
    ],
    workflow_instructions=(
        "1. Call ResearchAgent to gather background information.\n"
        "2. Outline the article structure.\n"
        "3. Write the full article.\n"
        "4. Review and refine."
    ),
))

agent = LlmAgent(name="WriterAgent", llm=llm, instructions=prompt)
```

Sections are only included when non-empty — no boilerplate padding.

---

## Runtime

### Streaming

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

---

### Callbacks

Attach hooks at the agent, model, and tool level:

```python
from langchain_adk import InvocationContext, LlmAgent
from langchain_adk.models.llm_request import LlmRequest
from langchain_adk.models.llm_response import LlmResponse

async def log_llm_call(ctx: InvocationContext, request: LlmRequest) -> None:
    print(f"[{ctx.agent_name}] LLM call with {len(request.messages)} messages")

async def track_usage(ctx: InvocationContext, response: LlmResponse) -> None:
    print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")

async def handle_llm_error(ctx: InvocationContext, request: LlmRequest, error: Exception) -> LlmResponse | None:
    """Called when an LLM call fails. Return a LlmResponse to recover, or None to propagate the error."""
    print(f"LLM error: {error}")
    return None  # yields ErrorEvent

async def log_tool(ctx: InvocationContext, name: str, args: dict) -> None:
    print(f"[TOOL] {name}({args})")

agent = LlmAgent(
    name="TrackedAgent",
    llm=llm,
    before_model_callback=log_llm_call,
    after_model_callback=track_usage,
    on_model_error_callback=handle_llm_error,
    before_tool_callback=log_tool,
)
```

Agent-level hooks:

```python
async def on_start(ctx: InvocationContext) -> None:
    print(f"Agent {ctx.agent_name} starting")

agent.before_agent_callback = on_start
```

### Tracing (Langfuse, LangSmith, etc.)

`RunConfig` mirrors LangChain's `RunnableConfig` fields — pass `callbacks`, `tags`, `metadata`, `run_name` directly. The entire agent run is wrapped in a single parent trace with all child operations nested automatically.

```python
from langfuse.langchain import CallbackHandler
from langchain_adk import RunConfig

run_config = RunConfig(
    callbacks=[CallbackHandler()],  # Langfuse, LangSmith, or any BaseCallbackHandler
    tags=["production", "user-facing"],
    metadata={"user_id": "u-123"},
    run_name="MyAgent",
)

# Via Runner
async for event in runner.run_async(
    user_id="user-1",
    session_id="session-1",
    new_message="Hello!",
    run_config=run_config,
):
    ...

# Or direct agent usage
ctx = InvocationContext(
    session_id="s1",
    agent_name="Agent",
    run_config=run_config,
)
async for event in agent.run_with_callbacks("Hello!", ctx=ctx):
    ...
```

Set the Langfuse environment variables:

```bash
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

`RunConfig` supports all LangChain `RunnableConfig` keys as first-class fields: `callbacks`, `tags`, `metadata`, `run_name`, `max_concurrency`, `configurable`.

**What gets traced:**
- Each agent run as a single parent trace (named after the agent)
- All LLM calls (`ainvoke` / `astream`) as child spans with token usage
- All tool executions as child spans with inputs/outputs
- Structured output fallback calls
- Works with composite agents — each sub-agent creates its own nested trace

**Across A2A boundaries:** A2A is HTTP, so callbacks can't cross the wire. Each A2A server should create its own handler. Link traces by passing a `trace_id` in task metadata.

---

### Structured Output

Force an agent to return a typed Pydantic object instead of free-form text. Pass `output_schema` to `LlmAgent` — the SDK automatically injects JSON format instructions into the prompt and parses the LLM response using LangChain's `PydanticOutputParser`.

```python
from pydantic import BaseModel, Field
from langchain_adk import LlmAgent
from langchain_adk.events.event import FinalAnswerEvent

class CompanyAnalysis(BaseModel):
    name: str = Field(description="Company name")
    industry: str = Field(description="Primary industry")
    strengths: list[str] = Field(description="Key strengths")
    risks: list[str] = Field(description="Key risks")
    recommendation: str = Field(description="Buy, Hold, or Sell")
    confidence: float = Field(description="Confidence score 0-1")

agent = LlmAgent(
    name="AnalystAgent",
    llm=llm,
    tools=[get_financials, get_news_sentiment],
    output_schema=CompanyAnalysis,
    instructions="You are a financial analyst. Research the company using the available tools.",
)
```

The parsed object is available on `FinalAnswerEvent.structured_output`:

```python
async for event in agent.run("Analyze Apple", ctx=ctx):
    if isinstance(event, FinalAnswerEvent):
        analysis = event.structured_output  # CompanyAnalysis instance
        print(f"{analysis.name}: {analysis.recommendation} ({analysis.confidence:.0%})")
```

**How it works:**
1. `PydanticOutputParser.get_format_instructions()` is appended to the system prompt — the LLM sees the JSON schema.
2. When the LLM responds, `PydanticOutputParser.parse()` extracts and validates JSON (handles markdown fences, partial JSON, etc.).
3. If direct parsing fails, `with_structured_output()` is used as a fallback for API-level schema enforcement.
4. Works with streaming (`StreamingMode.SSE`) and multi-agent compositions (`SequentialAgent`, `ParallelAgent`).

---

### Human-in-the-Loop

Create an `ask_human` tool that lets the agent ask the user questions interactively:

```python
from langchain_core.tools import tool

@tool
def ask_human(question: str) -> str:
    """Ask the human user a question and return their response."""
    print(f"\n  Agent asks: {question}")
    return input("  Your answer: ").strip()

agent = LlmAgent(
    name="Assistant",
    llm=llm,
    tools=[ask_human, book_meeting, send_email],
    instructions=(
        "When you don't have enough information to complete a task, "
        "use the ask_human tool to ask the user for the missing details."
    ),
)
```

The agent will call `ask_human` whenever it needs clarification, then use the response to proceed. See `examples/human_in_the_loop.py` for a full example.

---

## Composite Agents

### SequentialAgent

Chains sub-agents in order. The final answer of each agent becomes the input to the next.

```mermaid
flowchart LR
    classDef agent fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef io fill:#f1f5f9,stroke:#94a3b8,color:#334155

    In([input]):::io --> A1([Agent 1]):::agent
    A1 -->|final answer| A2([Agent 2]):::agent
    A2 -->|final answer| A3([Agent 3]):::agent
    A3 --> Out([output]):::io
    A1 -.->|escalate| Out
```

```python
from langchain_adk import SequentialAgent

pipeline = SequentialAgent(
    name="ResearchWriterPipeline",
    agents=[research_agent, writer_agent, editor_agent],
)

async for event in pipeline.run("Write about quantum computing", ctx=ctx):
    ...
```

Stops early if any event carries `actions.escalate=True`.

### ParallelAgent

Runs sub-agents concurrently. Events from all agents are merged into a single stream.

```mermaid
flowchart LR
    classDef agent fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef io fill:#f1f5f9,stroke:#94a3b8,color:#334155
    classDef merge fill:#dcfce7,stroke:#22c55e,color:#14532d

    In([input]):::io --> A1([Agent 1]):::agent
    In --> A2([Agent 2]):::agent
    In --> A3([Agent 3]):::agent
    A1 --> M([merged\nevent stream]):::merge
    A2 --> M
    A3 --> M
    M --> Out([output]):::io
```

```python
from langchain_adk import ParallelAgent

parallel = ParallelAgent(
    name="MultiSourceResearch",
    agents=[web_agent, academic_agent, news_agent],
)

async for event in parallel.run("Find info about fusion energy", ctx=ctx):
    # Events arrive from all three agents, tagged by agent_name
    print(f"[{event.agent_name}] {event.type}")
```

Each sub-agent gets a derived context with an isolated branch (`parallel.web_agent`, `parallel.academic_agent`, etc.) so their state and events don't collide.

### LoopAgent

Repeats its sub-agents until a termination condition is met.

```mermaid
flowchart LR
    classDef agent fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef decision fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef io fill:#f1f5f9,stroke:#94a3b8,color:#334155

    In([input]):::io --> A([sub-agent]):::agent
    A --> D{escalate or\nmax_iterations?}:::decision
    D -->|No, loop| A
    D -->|Yes| Out([output]):::io
```

```python
from langchain_adk import LoopAgent, exit_loop_tool

refine_agent = LlmAgent(
    name="RefineAgent",
    llm=llm,
    tools=[edit_tool, exit_loop_tool],
    instructions=(
        "Improve the draft. Call exit_loop when the quality is acceptable. "
        "Otherwise call the edit tool and keep refining."
    ),
)

loop = LoopAgent(
    name="RefinementLoop",
    agents=[refine_agent],
    max_iterations=5,
)

async for event in loop.run(draft_text, ctx=ctx):
    ...
```

`LoopAgent` stops when:
- An event has `actions.escalate=True` (set by `exit_loop_tool`)
- `max_iterations` is reached
- Optional `should_continue(event) -> bool` callback returns `False`

---

## Integrations

### Agent-to-Agent (A2A) Server

Expose any agent as a spec-compliant [A2A protocol](https://a2a-protocol.org/) endpoint (v0.3.0).

```mermaid
sequenceDiagram
    participant C as Client
    participant S as A2AServer
    participant A as BaseAgent

    Note over S: GET /.well-known/agent-card.json
    C->>S: Agent Card discovery
    S-->>C: AgentCard (name, skills, capabilities)

    Note over S: POST / (JSON-RPC 2.0)
    C->>S: message/send {message: {role, parts}}
    S->>A: run_with_callbacks(text, ctx)
    A-->>S: SDK Events
    S-->>C: Task {status: completed, artifacts, history}

    C->>S: message/stream {message: {role, parts}}
    S->>A: run_with_callbacks(text, ctx)
    loop SSE stream
        A-->>S: SDK Event
        S-->>C: data: {jsonrpc: "2.0", result: TaskStatusUpdateEvent}
        S-->>C: data: {jsonrpc: "2.0", result: TaskArtifactUpdateEvent}
    end
    S-->>C: data: {jsonrpc: "2.0", result: {status: completed, final: true}}

    C->>S: tasks/get {id: "task-uuid"}
    S-->>C: Task

    C->>S: tasks/cancel {id: "task-uuid"}
    S-->>C: Task {status: canceled}
```

```python
from langchain_adk import LlmAgent, InMemorySessionService
from langchain_adk.a2a import A2AServer, AgentSkill

agent = LlmAgent(name="MyAgent", llm=llm, tools=[...])

server = A2AServer(
    agent=agent,
    session_service=InMemorySessionService(),
    app_name="my-agent-service",
    skills=[
        AgentSkill(
            id="qa", name="Q&A",
            description="Answers general questions.",
            tags=["general"],
        ),
    ],
)

app = server.as_fastapi_app()
# uvicorn my_module:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/.well-known/agent-card.json` | Agent Card discovery |
| `POST` | `/` | JSON-RPC 2.0 dispatch |

**JSON-RPC methods:**

| Method | Description |
|---|---|
| `message/send` | Send message, receive completed Task |
| `message/stream` | Send message, receive SSE stream |
| `tasks/get` | Retrieve task by ID |
| `tasks/cancel` | Cancel a running task |

**Example requests:**

```bash
# Discover agent
curl http://localhost:8000/.well-known/agent-card.json

# Send message (blocking)
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "What is 2+2?"}]
      }
    }
  }'

# Stream response (SSE)
curl -N -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "2",
    "method": "message/stream",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "Tell me a story"}]
      }
    }
  }'
```

---

### MCP Integration

Install with MCP support:

```bash
pip install langchain-adk[mcp]
```

Connect to any MCP-compatible tool server and use its tools inside any agent.

```mermaid
sequenceDiagram
    participant A as LlmAgent
    participant W as WrappedMCPTool
    participant C as MCPClient
    participant S as MCP Server

    Note over C,S: Setup (once)
    C->>S: list_tools()
    S-->>C: [Tool definitions]
    C->>W: MCPToolAdapter wraps each tool

    Note over A,W: Runtime (per LLM tool call)
    A->>W: ainvoke({city: "Copenhagen"})
    W->>C: call_tool("get_weather", {city: "Copenhagen"})
    C->>S: Tool execution request
    S-->>C: CallToolResult [TextContent]
    C-->>W: result
    W-->>A: "Sunny, 22°C"
```

```python
from langchain_adk.integrations.mcp import MCPClient, MCPToolAdapter

# Connect to an MCP server (HTTP or in-memory)
client = MCPClient("http://localhost:8001/mcp")

# Wrap MCP tools as LangChain tools
adapter = MCPToolAdapter(client)
mcp_tools = await adapter.load_tools()

agent = LlmAgent(
    name="MCPAgent",
    llm=llm,
    tools=mcp_tools,
    instructions="Use the available tools to answer questions.",
)
```

For testing, pass a fastMCP server object directly (no HTTP needed):

```python
from fastmcp import FastMCP

server = FastMCP("TestServer")

@server.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

client = MCPClient(server)  # in-memory, no network
adapter = MCPToolAdapter(client)
tools = await adapter.load_tools()
```

`MCPToolAdapter.load_tools()` fetches the tool list from the MCP server and wraps each as a LangChain `BaseTool`.

---

## Examples

The `examples/` directory contains runnable demos for every major feature. Each example includes a `NotImplementedError` placeholder — replace it with your LLM of choice (e.g. `ChatOpenAI`, `ChatAnthropic`).

| Example | What it demonstrates |
|---|---|
| `basic_agent.py` | Minimal agent with a single tool |
| `streaming_agent.py` | SSE streaming with partial events |
| `multi_agent.py` | SequentialAgent pipeline (researcher → writer) |
| `parallel_agent.py` | ParallelAgent running 3 research agents concurrently |
| `loop_agent.py` | LoopAgent with writer/reviewer iterative refinement |
| `transfer_agent.py` | Agent handoff via `make_transfer_tool` (triage → specialists) |
| `planner_agent.py` | PlanReActPlanner with structured chain-of-thought |
| `memory_agent.py` | Persistent memory with `InMemoryMemoryStore` |
| `structured_output.py` | Typed Pydantic output via `output_schema` |
| `human_in_the_loop.py` | Interactive `ask_human` tool for user input |
| `task_orchestration.py` | TaskPlanner with managed task board |
| `mcp_agent.py` | MCP tool server integration |
| `a2a_server.py` | A2A protocol server endpoint |

```bash
# Run any example (after replacing the NotImplementedError with your LLM)
uv run python examples/basic_agent.py
```

---

## Architecture

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
    Runner --> Ctx[InvocationContext\nsession · user · state · run_config]:::base
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
    Events --> TC[ToolCallEvent]:::event
    Events --> TR[ToolResultEvent]:::event
    Events --> FA[FinalAnswerEvent\npartial=True for SSE]:::event
    Events --> ERR[ErrorEvent]:::event
```

```mermaid
sequenceDiagram
    participant C as Caller
    participant R as Runner
    participant S as SessionService
    participant A as Agent

    C->>R: run_async(user_id, session_id, message)
    R->>S: get_session() or create_session()
    S-->>R: Session
    R->>R: build InvocationContext from session.state

    R->>A: run_with_callbacks(message, ctx)

    loop for each yielded event
        A-->>R: Event
        R->>S: append_event(session, event)
        note over S: applies state_delta,\nappends to history
        R-->>C: yield Event
    end
```

**Key design decisions:**

- No LangGraph — orchestration is plain Python `asyncio` and async generators.
- `InvocationContext.state` is a shared mutable dict across the call tree. Use `EventActions.state_delta` to persist changes back to the session.
- `LlmRequest` / `LlmResponse` isolate LangChain types from the rest of the SDK. Swap the LLM provider without touching agent logic.
- Planners are per-turn hooks, not static prompts. They receive the live context and request so they can make dynamic decisions each turn.

---

## Development

```bash
# Clone and install with uv
git clone <repo-url>
cd agent-sdk
uv sync

# Run tests
uv run pytest tests/ -v

# Run an example (set your API key first)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python examples/basic_agent.py
```

**Adding a custom session backend:**

```python
from langchain_adk.sessions import BaseSessionService, Session

class RedisSessionService(BaseSessionService):
    async def create_session(self, *, app_name, user_id, state=None, session_id=None) -> Session: ...
    async def get_session(self, *, app_name, user_id, session_id) -> Session | None: ...
    async def update_session(self, session_id, *, state) -> Session: ...
    async def delete_session(self, session_id) -> None: ...
    async def list_sessions(self, *, app_name, user_id) -> list[Session]: ...
```

**Adding a custom planner:**

```python
from langchain_adk import BasePlanner, ReadonlyContext, LlmRequest

class MyPlanner(BasePlanner):
    def build_planning_instruction(self, ctx: ReadonlyContext, request: LlmRequest) -> str | None:
        # Return a string to append to the system prompt, or None to skip
        return "Always verify your answer before responding."
```

Pass it to any `LlmAgent` via `planner=MyPlanner()`.
