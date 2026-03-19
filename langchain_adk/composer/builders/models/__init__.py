"""Model provider registry and factory.

Built-in providers (``openai``, ``anthropic``, ``google``) are registered
lazily on first use.  Custom providers can be added via
:func:`register` or by passing a dotted import path as the provider name.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from langchain_adk.composer.errors import ComposerError

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseChatModel]] = {}

# Lazy-import specs: provider name -> (module, class_name)
_LAZY_PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
}


def register(name: str, cls: type[BaseChatModel]) -> None:
    """Register a custom model provider class.

    Example::

        register("my_provider", MyCustomChatModel)
    """
    _REGISTRY[name] = cls


def _resolve_provider(provider: str) -> type[BaseChatModel]:
    """Look up a provider class, lazy-importing built-ins as needed."""
    if provider in _REGISTRY:
        return _REGISTRY[provider]

    if provider in _LAZY_PROVIDERS:
        module_path, class_name = _LAZY_PROVIDERS[provider]
        import importlib

        module = importlib.import_module(module_path)
        cls: type[BaseChatModel] = getattr(module, class_name)
        _REGISTRY[provider] = cls
        return cls

    # Treat as a dotted import path to a custom class.
    from langchain_adk.composer.builders.tools import import_object

    cls = import_object(provider)
    _REGISTRY[provider] = cls
    return cls


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create(provider: str, name: str, **kwargs: Any) -> BaseChatModel:
    """Create a ``BaseChatModel`` from a provider name and model name.

    Parameters
    ----------
    provider:
        A registered name (``"openai"``, ``"anthropic"``, ``"google"``), or
        a fully-qualified class import path.
    name:
        Model name passed to the provider constructor.
    **kwargs:
        Extra keyword arguments forwarded to the model constructor.
    """
    try:
        cls = _resolve_provider(provider)
    except Exception as exc:
        msg = f"Failed to load model provider '{provider}': {exc}"
        raise ComposerError(msg) from exc
    return cls(model=name, **kwargs)
