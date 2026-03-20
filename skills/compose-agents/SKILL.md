---
name: compose-agents
description: Compose multi-agent systems with SequentialAgent, ParallelAgent, and LoopAgent. Use when building agent pipelines, parallel workflows, or iterative loops.
---

# Composing Multi-Agent Systems

Three orchestration primitives for combining agents.

## SequentialAgent — Run agents in order

Each agent's final answer becomes the next agent's input.

```python
from langchain_adk import SequentialAgent, LlmAgent

researcher = LlmAgent(name="researcher", llm=llm, instructions="Research the topic.")
writer = LlmAgent(name="writer", llm=llm, instructions="Write an article from the research.")

pipeline = SequentialAgent(
    name="pipeline",
    agents=[researcher, writer],
)

async for event in pipeline.astream("AI trends 2025"):
    if event.is_final_response():
        print(event.text)
```

## ParallelAgent — Run agents concurrently

All agents run simultaneously. Each gets a derived context with branch isolation.

```python
from langchain_adk import ParallelAgent, LlmAgent

analyst_a = LlmAgent(name="market", llm=llm, instructions="Analyze market trends.")
analyst_b = LlmAgent(name="tech", llm=llm, instructions="Analyze tech trends.")

parallel = ParallelAgent(
    name="analysis",
    agents=[analyst_a, analyst_b],
)
```

## LoopAgent — Repeat until done

Repeats sub-agents until `escalate=True` (via `exit_loop_tool`) or `max_iterations` reached.

```python
from langchain_adk import LoopAgent, LlmAgent
from langchain_adk.tools import exit_loop_tool

writer = LlmAgent(name="writer", llm=llm, instructions="Write a draft.")
reviewer = LlmAgent(
    name="reviewer",
    llm=llm,
    instructions="Review the draft. Call exit_loop if approved.",
    tools=[exit_loop_tool],
)

loop = LoopAgent(
    name="review_loop",
    agents=[writer, reviewer],
    max_iterations=5,
)
```

### Custom stop condition

```python
def quality_check(ctx, last_event):
    score = ctx.state.get("quality_score", 0)
    return score >= 8  # stop if quality is high

loop = LoopAgent(
    name="quality_loop",
    agents=[writer, reviewer],
    should_continue=quality_check,  # return True to continue
    max_iterations=10,
)
```

## Combining patterns

```python
# Research in parallel, then write sequentially, then review in a loop
research = ParallelAgent(name="research", agents=[market_analyst, tech_analyst])
write = LlmAgent(name="writer", llm=llm, instructions="Synthesize research into article.")
review_loop = LoopAgent(name="review", agents=[editor, fact_checker], max_iterations=3)

full_pipeline = SequentialAgent(
    name="content_pipeline",
    agents=[research, write, review_loop],
)
```

## Transfer routing — Agent handoff

```python
from langchain_adk import LlmAgent
from langchain_adk.tools import make_transfer_tool

sales = LlmAgent(name="sales", llm=llm, description="Handles orders.", instructions="...")
support = LlmAgent(name="support", llm=llm, description="Technical help.", instructions="...")

triage = LlmAgent(
    name="triage",
    llm=llm,
    instructions="Route the user to the right specialist.",
    tools=[make_transfer_tool([sales, support])],
)
triage.register_sub_agent(sales)
triage.register_sub_agent(support)
```
