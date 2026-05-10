"""LLM factory — create a LangChain chat model from a model name string.

Single entry point (:func:`create_llm`) that inspects the prefix of a
model identifier (``"gpt-"``, ``"claude-"``, ``"gemini-"``, ...) and
dispatches to the matching provider's LangChain integration.  Each
provider is imported lazily so the full matrix of optional providers
doesn't have to be installed.

Environment variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...)
are honoured transparently via the underlying LangChain clients.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

from orxhestra.cli.config import (
    PROVIDER_ENV_VARS,
    PROVIDER_INSTALL_HINTS,
    PROVIDER_PREFIXES,
)


def detect_provider(model_name: str) -> str:
    """Detect the LLM provider from a model name prefix.

    Parameters
    ----------
    model_name : str
        Model name string (e.g. ``"claude-sonnet-4-6"``).

    Returns
    -------
    str
        Provider identifier (e.g. ``"anthropic"``, ``"openai"``).
    """
    for prefix, provider in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return provider
    return "openai"


def create_llm(model_name: str, effort: str | None = None) -> BaseChatModel:
    """Create a LangChain BaseChatModel from a model name string.

    Raises a clear error if the provider package is missing or if
    the API key environment variable is not set.

    Parameters
    ----------
    model_name : str
        Model name string used to detect the provider and instantiate the model.
    effort : str or None, optional
        Unified reasoning effort (``"low"``, ``"medium"``, ``"high"``).
        Translated per-provider via :func:`effort_model_kwargs`. ``None``
        leaves the model defaults untouched.

    Returns
    -------
    BaseChatModel
        Configured LangChain chat model instance.
    """
    provider: str = detect_provider(model_name)

    env_var: str | None = PROVIDER_ENV_VARS.get(provider)
    if env_var and not os.environ.get(env_var):
        msg = f"Missing {env_var}. Set it with: export {env_var}=sk-..."
        raise RuntimeError(msg)

    from orxhestra.cli.config import effort_model_kwargs
    from orxhestra.composer.builders.models import create

    extra: dict = effort_model_kwargs(provider, effort) if effort else {}

    try:
        return create(provider, model_name, **extra)
    except Exception as exc:
        hint: str = PROVIDER_INSTALL_HINTS.get(provider, "")
        msg = f"Failed to create model '{model_name}' ({provider}): {exc}"
        if hint:
            msg += f"\nInstall the provider: {hint}"
        raise RuntimeError(msg) from exc
