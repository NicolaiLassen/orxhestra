"""Loop agent: draft → review → revise until approved.

Usage::

    python examples/composer/loop/run.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from orxhestra.composer import Composer
from orxhestra.events.event import EventType


async def main() -> None:
    yaml_path = Path(__file__).parent / "orx.yaml"
    runner = await Composer.runner_from_yaml_async(yaml_path)

    print("Review loop ready. Streaming...\n")

    async for event in runner.astream(
        user_id="user-1",
        session_id="session-1",
        new_message="Write a product description for a smart water bottle.",
    ):
        if event.type == EventType.AGENT_MESSAGE and event.partial:
            print(event.text, end="", flush=True)
        elif event.is_final_response():
            print(f"\n\n[{event.agent_name}] {event.text}")


if __name__ == "__main__":
    asyncio.run(main())
