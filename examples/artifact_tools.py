"""Artifact tools example — saving and loading files during agent execution.

Shows how to use make_artifact_tools() with an InMemoryArtifactService
so the agent can save, load, and list artifacts.

Prerequisites::

    pip install orxhestra

Run::

    python examples/artifact_tools.py
"""

from __future__ import annotations

import asyncio

from orxhestra import InMemorySessionService, LlmAgent, Runner
from orxhestra.artifacts.in_memory_artifact_service import InMemoryArtifactService
from orxhestra.events.event import EventType
from orxhestra.tools.artifact_tools import make_artifact_tools


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    artifact_tools = make_artifact_tools()

    agent = LlmAgent(
        name="ArtifactAgent",
        llm=llm,  # noqa: F821
        tools=artifact_tools,
        instructions=(
            "You are a helpful assistant that can save and load artifacts. "
            "When asked to create a file, use save_artifact. "
            "When asked to read a file, use load_artifact. "
            "Use list_artifacts to see what's available."
        ),
    )

    runner = Runner(
        agent=agent,
        app_name="artifact-demo",
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
    )

    print(f"Running agent: {agent.name}\n{'='*40}")

    async for event in runner.astream(
        user_id="user-1",
        session_id="session-1",
        new_message=(
            "Create a Python script called hello.py that prints 'Hello, World!'. "
            "Then list all artifacts and load the script back to verify it."
        ),
    ):
        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text[:200]}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
