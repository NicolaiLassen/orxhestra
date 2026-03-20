---
name: agent-planners
description: Add planners to langchain-adk agents for structured reasoning. Covers BasePlanner, PlanReActPlanner, and TaskPlanner.
---

# Agent Planners

Planners inject planning instructions into the system prompt before each LLM call.

## Custom planner

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

## PlanReActPlanner

Enforces structured planning tags — the agent must emit `/*PLANNING*/` and `/*FINAL_ANSWER*/` blocks.

```python
from langchain_adk import PlanReActPlanner, LlmAgent

agent = LlmAgent(
    name="PlanningAgent",
    llm=llm,
    tools=[...],
    planner=PlanReActPlanner(),
)
```

## TaskPlanner

Maintains a task board in `ctx.state` and injects status into the system prompt. Pairs with `ManageTasksTool`.

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

The agent calls `manage_tasks` with actions: `initialize`, `list`, `create`, `update`, `complete`, `remove`.
