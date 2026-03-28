"""A2A server: expose a composed agent as an A2A endpoint.

Usage::

    python examples/composer/a2a-server/run.py

Then discover the agent:

    curl http://localhost:8080/.well-known/agent.json

Send a message:

    curl -X POST http://localhost:8080 -H "Content-Type: application/json" -d '{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": "msg-1",
                "role": "user",
                "parts": [{"text": "What is the weather in Copenhagen?"}]
            }
        }
    }'
"""

import sys
from pathlib import Path

# In Docker, tools.py is in the same directory as orx.yaml.
# Locally, we also need the repo root for orxhestra.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from orxhestra.composer import Composer

yaml_path = Path(__file__).parent / "orx.yaml"
app = Composer.server_from_yaml(yaml_path)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
