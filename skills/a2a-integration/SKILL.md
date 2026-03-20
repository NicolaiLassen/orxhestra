---
name: a2a-integration
description: Expose langchain-adk agents as A2A protocol endpoints or connect to remote A2A agents.
---

# A2A Integration

## A2A Server — expose an agent

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

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/.well-known/agent.json` | Agent Card discovery |
| `POST` | `/` | JSON-RPC 2.0 dispatch |

### JSON-RPC methods

| Method | Description |
|---|---|
| `message/send` | Send message, receive completed Task |
| `message/stream` | Send message, receive SSE stream |
| `tasks/get` | Retrieve task by ID |
| `tasks/cancel` | Cancel a running task |

## A2A Agent — connect to a remote agent

```python
from langchain_adk.agents.a2a_agent import A2AAgent

remote = A2AAgent(
    name="RemoteAgent",
    description="A remote research agent.",
    url="http://localhost:9000",
)

async for event in remote.astream("What is quantum computing?"):
    print(event.text)
```

## In YAML Composer

```yaml
agents:
  remote_researcher:
    type: a2a
    description: "Remote research agent"
    url: "http://localhost:9000"

  orchestrator:
    type: llm
    tools:
      - agent: remote_researcher

main_agent: orchestrator
```
