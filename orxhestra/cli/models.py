"""LLM factory - create a LangChain chat model from a model name string."""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from orxhestra.cli.config import (
    PROVIDER_ENV_VARS,
    PROVIDER_INSTALL_HINTS,
    PROVIDER_PREFIXES,
)


def detect_provider(model_name: str) -> str:
    """Detect the LLM provider from a model name prefix."""
    for prefix, provider in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return provider
    return "openai"


def create_llm(model_name: str) -> BaseChatModel:
    """Create a LangChain BaseChatModel from a model name string.

    Raises a clear error if the provider package is missing or if
    the API key environment variable is not set.
    """
    provider: str = detect_provider(model_name)

    env_var: str | None = PROVIDER_ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        msg = f"Missing {env_var}. Set it with: export {env_var}=sk-..."
        raise RuntimeError(msg)

    from orxhestra.composer.builders.models import create

    try:
        return create(provider, model_name)
    except Exception as exc:
        hint: str = PROVIDER_INSTALL_HINTS.get(provider, "")
        msg = f"Failed to create model '{model_name}' ({provider}): {exc}"
        if hint:
            msg += f"\nInstall the provider: {hint}"
        raise RuntimeError(msg) from exc
