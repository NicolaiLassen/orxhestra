<p align="center">
  <img src="https://raw.githubusercontent.com/NicolaiLassen/orxhestra/main/assets/logo_text_bottom.svg" width="400" alt="orxhestra logo">
</p>

<p align="center">
  <strong>Multi-agent orchestration framework for Python — with a built-in coding CLI.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/orxhestra/"><img src="https://img.shields.io/pypi/v/orxhestra" alt="PyPI"></a>
  <a href="https://pypi.org/project/orxhestra/"><img src="https://img.shields.io/pypi/pyversions/orxhestra" alt="Python"></a>
  <a href="https://github.com/NicolaiLassen/orxhestra/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NicolaiLassen/orxhestra" alt="License"></a>
</p>

<br>

Compose multi-agent AI systems with async event streaming, agent hierarchies, and built-in support for MCP and A2A protocols.

## Orx CLI

A pre-built coding agent in your terminal — similar to Claude Code or Cursor — powered by any LLM. One install command and you're up and running.

```bash
pip install orxhestra[cli,openai]
orx
```

```
  orx - terminal coding agent
  model: gpt-5.4 | workspace: ~/my-project
  type /help for commands, Ctrl+D to exit

orx> add error handling to the API routes

  > read_file(src/api/routes.py)
  > grep(pattern=raise, path=src/api/)
  > write_todos([{"content": "Add try/except to all route handlers", "status": "in_progress"}, ...])

  Tasks:
    ▶ Add try/except to all route handlers
    ○ Add custom error response model
    ○ Write tests for error cases

  > edit_file(src/api/routes.py)
  > shell_exec(pytest tests/test_api.py)

  Done. Added structured error handling to all 4 route handlers with
  a custom `ErrorResponse` model. All tests pass.
```

### Features

- **Any LLM** — OpenAI, Anthropic, Google via `--model gpt-5.4` / `claude-sonnet-4-6` / `gemini-2.0-flash`
- **Streaming** — real-time token rendering with Markdown formatting
- **Tool approval** — prompts before destructive operations (write, edit, shell)
- **Task planning** — structured todo lists visible in the terminal
- **Sub-agent delegation** — spawn isolated agents for complex subtasks
- **AGENTS.md memory** — persistent project context across sessions
- **Local context injection** — auto-detects language, git state, package manager, project tree
- **Context summarization** — auto-compacts long conversations, `/compact` command
- **Compose support** — run any compose YAML: `orx compose.yaml`
- **Multi-agent pipeline** — built-in plan → code → review loop: `orx --coding`

### Usage

```bash
orx                               # interactive REPL (default model)
orx --model claude-sonnet-4-6     # use a specific model
orx -c "fix the failing tests"    # single-shot command
orx compose.yaml                  # run a compose file
orx --coding                      # multi-agent coding pipeline
orx --auto-approve                # skip approval prompts
```

### Commands

| Command | Description |
|---------|-------------|
| `/model <name>` | Switch model mid-session |
| `/clear` | Reset conversation |
| `/compact` | Summarize old messages to free context |
| `/todos` | Show current task list |
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
> For full documentation, guides, and API reference, visit [orxhestra.com](https://docs.orxhestra.com).

## Features

- **Agent ensemble** - LLM, ReAct, Sequential, Parallel, and Loop agents
- **Event streaming** - Async event-driven architecture with real-time streaming
- **Composer** - Conduct entire agent orchestras declaratively with YAML
- **Tools** - Function tools, filesystem tools, agent-as-tool, shell, and long-running tool support
- **Planners** - Choreograph task execution with PlanReAct and TaskPlanner strategies
- **Skills** - Reusable, composable agent repertoires
- **MCP** - Model Context Protocol integration for tool servers
- **A2A** - Agent-to-Agent protocol for cross-service harmonization
- **Memory** - Pluggable memory stores for persistent agent context
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

```bash
orxhestra compose.yaml
```

## Docker

```bash
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v ./compose.yaml:/app/compose.yaml \
  nicolaimtlassen/orxhestra
```

## Documentation

- [Getting Started](https://docs.orxhestra.com/getting-started/quickstart) - Installation and first agent
- [Agents](https://docs.orxhestra.com/concepts/agents) - Agent types and configuration
- [Tools](https://docs.orxhestra.com/tools/overview) - Built-in and custom tools
- [Composer](https://docs.orxhestra.com/composer/overview) - YAML-based agent composition
- [Integrations](https://docs.orxhestra.com/integrations/mcp) - MCP and A2A setup

---

## Acknowledgments

This project is built on the shoulders of several outstanding open-source projects and research efforts:

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Agent Development Kit (ADK)](https://github.com/google/adk-python)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io)
- [Agent-to-Agent Protocol (A2A)](https://github.com/google/A2A)

Special thanks to the open-source AI community for pushing the boundaries of what's possible with agent frameworks.
