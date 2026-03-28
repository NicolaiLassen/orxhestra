"""Coding agent: plan → code/review loop.

Usage::

    python examples/composer/coding/run.py
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

    prompt = (
        "Create a simple HTML page and serve it on port 8080 using Python's http.server. "
        "The page should have a centered heading 'Hello from Orxhestra' with a dark background "
        "and the orxhestra gradient colors (blue to purple to sky). "
        "Create the files in /tmp/orxhestra-demo/ and start the server."
    )

    print(f"Prompt: {prompt}\n")
    print("=" * 60)

    async for event in runner.astream(
        user_id="user-1",
        session_id="session-1",
        new_message=prompt,
    ):
        if event.type == EventType.AGENT_MESSAGE and event.partial:
            print(event.text, end="", flush=True)
        elif event.is_final_response():
            print(f"\n\n[{event.agent_name}] {event.text}")
        elif event.type == EventType.TOOL_RESPONSE:
            resp = str(event.text)[:200]
            print(f"< {resp}")


if __name__ == "__main__":
    asyncio.run(main())
