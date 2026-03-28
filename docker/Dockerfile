FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install orxhestra with all extras
RUN uv pip install --system --no-cache \
    orxhestra[composer,openai,anthropic,google,a2a,mcp] \
    uvicorn

# Generic entrypoint that loads orx.yaml from /app
COPY entrypoint.py /app/entrypoint.py

EXPOSE 8080

# Users mount their orx.yaml (and optional tools.py) to /app
# docker run -v ./orx.yaml:/app/orx.yaml -e OPENAI_API_KEY=$OPENAI_API_KEY nicolaimtlassen/orxhestra
CMD ["python", "entrypoint.py"]
