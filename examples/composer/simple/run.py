"""Simplest composer example — a single LLM agent from YAML.

Usage::

    python examples/composer/simple/run.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from orxhestra.composer import Composer
from orxhestra.events.event import EventType


async def main() -> None:
    yaml_path = Path(__file__).parent / "orx.yaml"
    agent = await Composer.from_yaml_async(yaml_path)

    print("Simple agent ready. Streaming...\n")

    async for event in agent.astream("What are the benefits of microservices?"):
        if event.type == EventType.AGENT_MESSAGE and event.partial:
            print(event.text, end="", flush=True)
        elif event.is_final_response():
            print(f"\n\n[Done] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
