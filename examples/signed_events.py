"""Signed events example — cryptographic event attribution.

Shows how to create an agent with a signing key so every event is
signed with Ed25519, and how to verify those signatures.

Prerequisites::

    pip install orxhestra[auth]

Run::

    python examples/signed_events.py
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from orxhestra import LlmAgent
from orxhestra.events.event import EventType

# Import auth utilities for key generation
try:
    from orxhestra.auth.crypto import generate_ed25519_keypair, public_key_to_did_key
except ImportError:
    raise ImportError(
        "This example requires orxhestra[auth]. "
        "Install with: pip install orxhestra[auth]"
    )


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 22°C."


async def main() -> None:
    # --- Replace with a real LLM ---
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-5.4")
    raise NotImplementedError(
        "Replace the llm= line below with a real LangChain chat model "
        "and comment out this raise."
    )

    # Generate a signing key pair for this agent
    signing_key, public_key = generate_ed25519_keypair()
    signing_did = public_key_to_did_key(public_key)

    agent = LlmAgent(
        name="SignedAgent",
        llm=llm,  # noqa: F821
        tools=[get_weather],
        instructions="You are a helpful weather assistant.",
        signing_key=signing_key,
        signing_did=signing_did,
    )

    print(f"Running agent: {agent.name}")
    print(f"Agent DID: {signing_did}\n{'='*40}")

    async for event in agent.astream("What's the weather in Copenhagen?"):
        # Show signature info for each event
        if event.is_signed:
            valid = event.verify_signature()
            status = "VALID" if valid else "INVALID"
            print(f"  [SIG {status}] {event.type.value} by {event.signer_did[:30]}...")

        if event.has_tool_calls:
            print(f"[TOOL CALL] {event.tool_name}({event.tool_input})")
        elif event.type == EventType.TOOL_RESPONSE:
            print(f"[TOOL RESULT] {event.text}")
        elif event.is_final_response():
            print(f"\n[ANSWER] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
