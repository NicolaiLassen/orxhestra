"""CLI configuration constants — defaults, env-var names, provider map.

Centralises the small set of string literals the CLI refers to in
multiple places: default user / app names, the default model name,
and the per-provider API-key environment variables consulted by
:func:`orxhestra.cli.models.create_llm`.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME: str = "orx-cli"
DEFAULT_USER_ID: str = "cli-user"
DEFAULT_MODEL: str = os.environ.get("ORX_MODEL", "gpt-5.4")
HISTORY_DIR: Path = Path.home() / ".orx"
HISTORY_FILE: Path = HISTORY_DIR / "history"

# Provider detection: prefix -> provider name for model registry.
# Order matters — first match wins.
PROVIDER_PREFIXES: list[tuple[str, str]] = [
    # OpenAI
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("o5-", "openai"),
    ("chatgpt-", "openai"),
    ("dall-e-", "openai"),
    # Anthropic
    ("claude-", "anthropic"),
    # Google
    ("gemini-", "google"),
    ("gemma-", "google"),
    # Mistral
    ("mistral-", "mistralai"),
    ("codestral-", "mistralai"),
    ("pixtral-", "mistralai"),
    ("ministral-", "mistralai"),
    ("open-mistral-", "mistralai"),
    ("open-mixtral-", "mistralai"),
    # Cohere
    ("command-", "cohere"),
    ("c4ai-", "cohere"),
    ("embed-", "cohere"),
    # DeepSeek
    ("deepseek-", "deepseek"),
    # Groq
    ("llama-", "groq"),
    ("llama3-", "groq"),
    ("mixtral-", "groq"),
    ("gemma2-", "groq"),
    # xAI
    ("grok-", "xai"),
    # Perplexity
    ("sonar-", "perplexity"),
    # IBM watsonx
    ("ibm-", "ibm"),
    ("granite-", "ibm"),
    # NVIDIA NIM
    ("meta/llama-", "nvidia"),
    ("nvidia/", "nvidia"),
    # Upstage
    ("solar-", "upstage"),
    # Fireworks
    ("accounts/fireworks/", "fireworks"),
    # Together
    ("togethercomputer/", "together"),
    ("meta-llama/", "together"),
    ("Qwen/", "together"),
    # HuggingFace
    ("bigscience/", "huggingface"),
    ("tiiuae/", "huggingface"),
    ("microsoft/", "huggingface"),
    # AWS Bedrock — model names overlap, use explicit provider
    # Azure OpenAI — model names overlap with OpenAI, use explicit provider
    # Ollama — local models, use explicit provider
    # OpenRouter — aggregator, use explicit provider
]

# Env var checked before calling the provider.
PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "azure_ai": "AZURE_AI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "perplexity": "PPLX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "upstage": "UPSTAGE_API_KEY",
    # No env var required for: google_vertexai (ADC), aws_bedrock (boto3),
    # huggingface (HF_TOKEN), ollama (local), ibm (uses IAM).
}

PROVIDER_INSTALL_HINTS: dict[str, str] = {
    "openai": "pip install orxhestra[openai]",
    "azure_openai": "pip install orxhestra[openai]",
    "azure": "pip install orxhestra[openai]",
    "azure_ai": "pip install langchain-azure-ai",
    "anthropic": "pip install orxhestra[anthropic]",
    "anthropic_bedrock": "pip install langchain-aws",
    "google": "pip install orxhestra[google]",
    "google_genai": "pip install orxhestra[google]",
    "google_vertexai": "pip install langchain-google-vertexai",
    "google_anthropic_vertex": "pip install langchain-google-vertexai",
    "vertexai": "pip install langchain-google-vertexai",
    "bedrock": "pip install langchain-aws",
    "bedrock_converse": "pip install langchain-aws",
    "aws_bedrock": "pip install langchain-aws",
    "mistralai": "pip install langchain-mistralai",
    "mistral": "pip install langchain-mistralai",
    "cohere": "pip install langchain-cohere",
    "fireworks": "pip install langchain-fireworks",
    "together": "pip install langchain-together",
    "groq": "pip install langchain-groq",
    "nvidia": "pip install langchain-nvidia-ai-endpoints",
    "huggingface": "pip install langchain-huggingface",
    "deepseek": "pip install langchain-deepseek",
    "ollama": "pip install langchain-ollama",
    "xai": "pip install langchain-xai",
    "ibm": "pip install langchain-ibm",
    "perplexity": "pip install langchain-perplexity",
    "openrouter": "pip install langchain-openrouter",
    "upstage": "pip install langchain-upstage",
}
