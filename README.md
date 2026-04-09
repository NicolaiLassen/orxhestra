<p align="center">
  <img src="https://raw.githubusercontent.com/NicolaiLassen/orxhestra/main/assets/logo_text_bottom.svg" width="400" alt="orxhestra logo">
</p>

<p align="center">
  <strong>Multi-agent orchestration framework for Python — turn any agent setup into a CLI or server.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/orxhestra/"><img src="https://img.shields.io/pypi/v/orxhestra" alt="PyPI"></a>
  <a href="https://pypi.org/project/orxhestra/"><img src="https://img.shields.io/pypi/pyversions/orxhestra" alt="Python"></a>
  <a href="https://github.com/NicolaiLassen/orxhestra/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NicolaiLassen/orxhestra" alt="License"></a>
</p>

<br>

Compose multi-agent AI systems with async event streaming, agent hierarchies, and built-in support for MCP and A2A protocols.

## Orx CLI

Turn any `orx.yaml` agent setup into an interactive terminal agent. Ships with a coding agent out of the box — or compose your own.

```bash
pip install orxhestra[cli,openai]
orx
```

```
+-- orx - terminal coding agent ------------------------------------+
|  model: gpt-5.4   workspace: ~/my-project   /help for commands    |
+-------------------------------------------------------------------+

orx> add error handling to the API routes

  > read_file(src/api/routes.py)
  > grep(pattern="raise", path=src/api/)
  > write_todos(3 tasks)

  Tasks
  * Add try/except to all route handlers  [in progress]
  - Add custom error response model
  - Write tests for error cases

  > edit_file(src/api/routes.py)
  > shell_exec(pytest tests/test_api.py)
  4 passed

  Done - added structured error handling to all 4 route handlers
  with a custom ErrorResponse model. All tests pass.
```

### Features

- **29 LLM providers** — OpenAI, Azure OpenAI, Anthropic, Google, Mistral, Cohere, Groq, DeepSeek, Ollama, and 20 more via `--model`
- **Streaming** — real-time token rendering with Markdown formatting
- **Tool approval** — prompts before destructive operations (write, edit, shell)
- **Task planning** — structured todo lists visible in the terminal
- **Sub-agent delegation** — spawn isolated agents for complex subtasks
- **Auto-memory** — persistent per-project memories across sessions (4 types: user, feedback, project, reference)
- **Dark/light theme** — auto-detects terminal, toggle with `/theme`
- **Background tasks** — spawn and monitor async sub-agent tasks
- **Smart file reading** — offset/limit pagination with line numbers, 256KB size guard
- **Local context injection** — auto-detects language, git state, package manager, project tree
- **Context summarization** — auto-compacts long conversations, `/compact` command
- **Orx YAML** — run any orx.yaml agent team: `orx my-agents.yaml`

### Usage

```bash
orx                               # interactive REPL (default model)
orx --model claude-sonnet-4-6     # use a specific model
orx -c "fix the failing tests"    # single-shot command
orx my-agents.yaml                # run a custom orx file
orx --auto-approve                # skip approval prompts
orx orx.yaml --serve -p 9000      # start as A2A server
```

### Commands

| Command | Description |
|---------|-------------|
| `/model <name>` | Switch model mid-session |
| `/clear` | Reset conversation |
| `/compact` | Summarize old messages to free context |
| `/todos` | Show current task list |
| `/memory` | List saved memories |
| `/theme` | Switch dark/light theme |
| `/session` | Session info |
| `/undo` | Remove last turn |
| `/retry` | Re-run last message |
| `/copy` | Copy last response |
| `/help` | Show all commands |
| `/exit` | Exit |

---

## Quickstart (SDK)

```bash
pip install orxhestra
# or
uv add orxhestra
```

```python
from orxhestra import LlmAgent, Runner, InMemorySessionService

agent = LlmAgent(
    name="assistant",
    model="gpt-5.4",
    instructions="You are a helpful assistant.",
)

runner = Runner(agent=agent, session_service=InMemorySessionService())
response = await runner.run(user_id="user1", session_id="s1", new_message="Hello!")

for event in response:
    print(event.content)
```

> [!TIP]
> For persistent database sessions, install the database extra: `pip install orxhestra[database]`

> [!TIP]
> For full documentation, guides, and API reference, visit [docs.orxhestra.com](https://docs.orxhestra.com).

## Features

- **Agent ensemble** - LLM, ReAct, Sequential, Parallel, and Loop agents
- **29 LLM providers** - OpenAI, Azure OpenAI, Anthropic, Google, Mistral, Cohere, Groq, DeepSeek, Ollama, and 20 more
- **Event streaming** - Async event-driven architecture with real-time streaming
- **Composer** - Conduct entire agent orchestras declaratively with YAML
- **Tools** - Function tools, filesystem tools, agent-as-tool, shell, and long-running tool support
- **Planners** - Choreograph task execution with PlanReAct and TaskPlanner strategies
- **Skills** - Reusable, composable agent repertoires
- **MCP** - Model Context Protocol integration for tool servers
- **A2A** - Agent-to-Agent protocol for cross-service harmonization
- **Auto-memory** - Persistent memories with save_memory tool (user, feedback, project, reference)
- **Background tasks** - Async sub-agent task lifecycle with spawn and monitor
- **Deprecation decorators** - `@deprecated` and `@deprecated_param` for clean API evolution
- **Tracing** - Built-in support for Langfuse, LangSmith, and custom callbacks

## Agents at a glance

| Agent | Description |
|-------|-------------|
| `LlmAgent` | Chat model agent with tools, instructions, and structured output |
| `ReActAgent` | Reasoning + acting loop with automatic tool use |
| `SequentialAgent` | Runs sub-agents in order |
| `ParallelAgent` | Runs sub-agents concurrently |
| `LoopAgent` | Repeats a sub-agent until exit condition |
| `A2AAgent` | Connects to remote agents via A2A protocol |

## Composer

Define entire agent orchestras in a single YAML file — no Python wiring needed. Compose LLM agents, loops, pipelines, tools, and review cycles declaratively. The example below builds a coding agent that plans, implements with filesystem + shell access, and self-reviews in a loop:

```yaml
defaults:
  model:
    provider: openai
    name: gpt-5.4

tools:
  exit:
    builtin: "exit_loop"
  filesystem:
    builtin: "filesystem"
  shell:
    builtin: "shell"

agents:
  planner:
    type: llm
    description: "Plans the implementation steps for the coder agent."
    instructions: |
      Output a numbered list of concrete steps the coder
      should execute. Each step must be an actionable file
      operation or shell command.

  coder:
    type: llm
    description: "Implements code changes with filesystem and shell access."
    instructions: |
      Follow the plan from the previous step exactly.
      Use filesystem tools to create files and shell to
      run commands. Never ask the user to do anything.
    tools:
      - filesystem
      - shell

  reviewer:
    type: llm
    description: "Reviews changes and approves or requests fixes."
    instructions: |
      Check files exist and look correct. If done, call
      exit_loop. Otherwise describe what needs fixing.
    tools:
      - exit

  dev_loop:
    type: loop
    agents: [coder, reviewer]
    max_iterations: 10

  coordinator:
    type: sequential
    agents: [planner, dev_loop]

main_agent: coordinator

runner:
  app_name: coding-agent
  session_service: memory
```

Run it as an interactive CLI or expose it as an A2A server:

```bash
orx orx.yaml                    # interactive terminal agent
orx orx.yaml --serve -p 9000    # A2A server on port 9000
```

```bash
# test the server
curl -X POST http://localhost:9000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"text": "Hello!", "mediaType": "text/plain"}]
      }
    }
  }'
```

## Docker

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v ./orx.yaml:/app/orx.yaml \
  nicolaimtlassen/orxhestra
```

## Documentation

- [Getting Started](https://docs.orxhestra.com/getting-started/quickstart) - Installation and first agent
- [Agents](https://docs.orxhestra.com/concepts/agents) - Agent types and configuration
- [Tools](https://docs.orxhestra.com/tools/overview) - Built-in and custom tools
- [Composer](https://docs.orxhestra.com/composer/overview) - YAML-based agent composition
- [Integrations](https://docs.orxhestra.com/integrations/mcp) - MCP and A2A setup
- [Skills](docs/skills/) - Code-level CLI skill references (agent-tools, callbacks, planners, streaming, and more)

---

## Acknowledgments

This project is built on the shoulders of several outstanding open-source projects and research efforts:

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Agent Development Kit (ADK)](https://github.com/google/adk-python)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- [Agent-to-Agent Protocol (A2A)](https://github.com/google/A2A)

Special thanks to the open-source AI community for pushing the boundaries of what's possible with agent frameworks.
