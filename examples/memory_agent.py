"""Memory agent example - persistent memory across sessions.

Demonstrates:
  - InMemoryMemoryService for storing and recalling facts
  - Agent remembers information from previous conversations
"""

from __future__ import annotations

import asyncio

from orxhestra import LlmAgent
from orxhestra.memory.in_memory_service import InMemoryMemoryService
from orxhestra.memory.memory import Memory


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # --- Set up memory ---
    memory_service = InMemoryMemoryService()

    # Pre-populate some memories (simulating past conversations)
    key = ("my-app", "user-42")
    memory_service._store[key] = [
        Memory(content="User's name is Alice.", author="agent"),
        Memory(content="Alice is a backend engineer working on microservices.", author="agent"),
        Memory(content="Alice prefers Python over JavaScript.", author="agent"),
        Memory(content="Alice's team is migrating from REST to gRPC.", author="agent"),
    ]

    # Search for relevant memories
    result = await memory_service.search_memory(
        app_name="my-app",
        user_id="user-42",
        query="alice",
    )

    # Build context from memories
    memory_context = "\n".join(
        f"- {m.content}" for m in result.memories
    )

    agent = LlmAgent(
        name="MemoryAgent",
        llm=llm,  # noqa: F821
        instructions=(
            "You are a helpful assistant with memory of past conversations.\n\n"
            "Here is what you remember about this user:\n"
            f"{memory_context}\n\n"
            "Use this context to personalize your responses. "
            "Reference what you know when relevant."
        ),
    )

    print("Asking a personalized question (with memory context)")
    print("=" * 50)

    async for event in agent.astream(
        "What programming language should I use for my new service?",
    ):
        if event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
