"""Generic entrypoint for the orxhestra Docker image.

Loads orx.yaml from /app and starts either:
  - An A2A server (if `server:` section exists in YAML) on port 8080
  - A Runner with interactive stdin (if `runner:` section exists)
  - A standalone agent that runs a single prompt from $PROMPT env var

Usage:
  # A2A server mode (default when server: is defined)
  docker run -v ./orx.yaml:/app/orx.yaml -p 8080:8080 nicolaimtlassen/orxhestra

  # With custom tools
  docker run -v ./orx.yaml:/app/orx.yaml -v ./tools.py:/app/tools.py -p 8080:8080 nicolaimtlassen/orxhestra

  # Override port
  docker run -e PORT=9000 -p 9000:9000 -v ./orx.yaml:/app/orx.yaml nicolaimtlassen/orxhestra
"""

import asyncio
import os
import sys
from pathlib import Path

# Add /app to sys.path so user-mounted tools.py can be imported
sys.path.insert(0, "/app")


def main() -> None:
    yaml_path = Path("/app/orx.yaml")
    if not yaml_path.exists():
        print("Error: No orx.yaml found at /app/orx.yaml")
        print("Mount your config: docker run -v ./orx.yaml:/app/orx.yaml ...")
        sys.exit(1)

    from orxhestra.composer import Composer

    # Read YAML to check which mode to use
    import yaml

    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")

    if spec.get("server"):
        # A2A server mode
        import uvicorn

        app = Composer.server_from_yaml(yaml_path)
        print(f"Starting A2A server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

    elif spec.get("runner"):
        # Runner mode — run a single prompt from env
        prompt = os.environ.get("PROMPT")
        if not prompt:
            print("Error: Runner mode requires PROMPT env var")
            print('  docker run -e PROMPT="Hello!" -v ./orx.yaml:/app/orx.yaml ...')
            sys.exit(1)

        runner = Composer.runner_from_yaml(yaml_path)

        async def run() -> None:
            async for event in runner.astream(
                user_id="docker-user",
                session_id="docker-session",
                new_message=prompt,
            ):
                if event.is_final_response():
                    print(event.text)

        asyncio.run(run())

    else:
        # Standalone agent mode
        prompt = os.environ.get("PROMPT", "Hello!")
        agent = Composer.from_yaml(yaml_path)

        async def run() -> None:
            async for event in agent.astream(prompt):
                if event.is_final_response():
                    print(event.text)

        asyncio.run(run())


if __name__ == "__main__":
    main()
