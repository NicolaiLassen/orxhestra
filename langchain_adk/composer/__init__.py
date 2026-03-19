"""Declarative YAML-based agent composition.

Registries for extending the composer:

- ``register_builder`` — add custom agent types
- ``register_provider`` — add custom model providers
- ``register_builtin_tool`` — add custom builtin tools
"""

from langchain_adk.composer.builders.agents import register as register_builder
from langchain_adk.composer.builders.models import register as register_provider
from langchain_adk.composer.builders.tools import register_builtin as register_builtin_tool
from langchain_adk.composer.composer import Composer
from langchain_adk.composer.errors import CircularReferenceError, ComposerError

__all__ = [
    "Composer",
    "ComposerError",
    "CircularReferenceError",
    "register_builder",
    "register_provider",
    "register_builtin_tool",
]
