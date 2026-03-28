---
name: yaml-composer
description: Build orxhestra agent trees from declarative YAML orx files. Covers full schema, models, tools, agents, runner, and server.
---

# YAML Composer

Build an entire multi-agent setup from a single YAML file.

```bash
pip install orxhestra[composer]
```

## Minimal example

```yaml
defaults:
  model:
    provider: openai
    name: gpt-4o

agents:
  assistant:
    type: llm
    instructions: "You are a helpful assistant."

main_agent: assistant
```

```python
from orxhestra.composer import Composer

agent = Composer.from_yaml("orx.yaml")
async for event in agent.astream("Hello"):
    print(event.text)
```

## Named models

```yaml
models:
  smart:
    provider: anthropic
    name: claude-opus-4-6
    max_tokens: 8192
  fast:
    provider: openai
    name: gpt-4o-mini

agents:
  researcher:
    type: llm
    model: smart
  writer:
    type: llm
    model: fast
```

Extra keys on a model config are forwarded directly to the LangChain model constructor.

## Tools

```yaml
tools:
  search:
    function: "myapp.tools.search_web"
  weather:
    mcp:
      url: "http://localhost:8001/mcp"
  exit:
    builtin: "exit_loop"

agents:
  agent:
    type: llm
    tools:
      - search
      - weather
      - function: "myapp.tools.inline_tool"  # inline tool def
```

## Agent types

- `llm` — LlmAgent
- `react` — ReActAgent
- `sequential` — SequentialAgent (runs agents in order)
- `parallel` — ParallelAgent (runs agents concurrently)
- `loop` — LoopAgent (repeats until exit_loop or max_iterations)
- `a2a` — A2AAgent (remote agent via A2A protocol)

## Multi-agent with transfer

```yaml
agents:
  triage:
    type: llm
    instructions: "Route to the right specialist."
    tools:
      - transfer:
          targets: [sales, support]
  sales:
    type: llm
    instructions: "Handle sales inquiries."
  support:
    type: llm
    instructions: "Handle support tickets."

main_agent: triage
```

## Sequential pipeline

```yaml
agents:
  researcher:
    type: llm
    tools: [search]
  writer:
    type: llm
  pipeline:
    type: sequential
    agents: [researcher, writer]

main_agent: pipeline
```

## Runner and Server

```yaml
runner:
  app_name: "my-app"
  session_service: "memory"

server:
  app_name: "my-app"
  version: "1.0.0"
  url: "http://localhost:8000"
  skills:
    - id: qa
      name: Q&A
      description: "Answers questions"
```

```python
runner = Composer.runner_from_yaml("orx.yaml")
app = Composer.server_from_yaml("orx.yaml")  # FastAPI app
```
