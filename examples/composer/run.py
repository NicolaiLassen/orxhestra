"""Run a composed agent team from a YAML file.

Usage::

    python examples/composer/run.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so "examples.composer.tools" resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_adk.composer import Composer
from langchain_adk.events.event import EventType


async def main() -> None:
    yaml_path = Path(__file__).parent / "compose.yaml"

    # Build the full agent tree + runner from YAML.
    runner = await Composer.runner_from_yaml_async(yaml_path)

    print("Agent team composed from YAML. Streaming...\n")

    async for event in runner.astream(
        user_id="user-1",
        session_id="session-1",
        new_message="Where is my order #12345?",
    ):
        if event.type == EventType.AGENT_MESSAGE and event.partial:
            print(event.text, end="", flush=True)
        elif event.is_final_response():
            print(f"\n\n[{event.agent_name}] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
