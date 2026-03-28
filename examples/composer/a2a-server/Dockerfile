FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system --no-cache orxhestra[composer,openai,a2a] uvicorn

COPY orx.yaml tools.py run.py ./

EXPOSE 8080

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8080"]
