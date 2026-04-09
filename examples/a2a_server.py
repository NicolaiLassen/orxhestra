"""A2A server example — spec-compliant Agent-to-Agent protocol.

Run with:
    uvicorn examples.a2a_server:app

Then test:
    # Agent Card discovery
    curl http://localhost:8000/.well-known/agent-card.json

    # message/send (JSON-RPC 2.0)
    curl -X POST http://localhost:8000/ \\
      -H "Content-Type: application/json" \\
      -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
          "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "What is 2+2?"}]
          }
        }
      }'

    # message/stream (SSE)
    curl -N -X POST http://localhost:8000/ \\
      -H "Content-Type: application/json" \\
      -d '{
        "jsonrpc": "2.0",
        "id": "2",
        "method": "message/stream",
        "params": {
          "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "What is 2+2?"}]
          }
        }
      }'
"""

from __future__ import annotations

from orxhestra import InMemorySessionService, LlmAgent
from orxhestra.a2a.server import A2AServer
from orxhestra.a2a.types import AgentSkill

# --- Replace with a real LLM ---
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-5.4")
raise NotImplementedError(
    "Replace the llm= line below with a real LangChain chat model "
    "and comment out this raise."
)

agent = LlmAgent(
    name="AssistantAgent",
    llm=llm,  # noqa: F821
    instructions="You are a helpful assistant.",
)

server = A2AServer(
    agent,
    session_service=InMemorySessionService(),
    app_name="demo",
    skills=[
        AgentSkill(
            id="general",
            name="General Assistant",
            description="Answers general questions.",
            tags=["general", "qa"],
        ),
    ],
)

app = server.as_fastapi_app()
