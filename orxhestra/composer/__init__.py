"""Declarative YAML-based agent composition.

Registries for extending the composer:

- :func:`register_builder` — add custom agent types (``type:`` keys
  under YAML ``agents:``).
- :func:`register_provider` — add custom LLM model providers
  consumed by ``model: { provider: ... }``.
- :func:`register_builtin_tool` — add custom built-in tools
  referenced by ``tools: { ..., builtin: ... }``.
- :func:`register_tool_resolver` — add a whole new *tool type*,
  extending :class:`~orxhestra.composer.schema.ToolDef.custom`.

See Also
--------
Composer : Top-level builder class.
ComposeSpec : Root YAML schema.
"""

from orxhestra.composer.builders.agents import register as register_builder
from orxhestra.composer.builders.models import register as register_provider
from orxhestra.composer.builders.tools import register_builtin as register_builtin_tool
from orxhestra.composer.builders.tools import register_tool_resolver
from orxhestra.composer.composer import Composer
from orxhestra.composer.errors import CircularReferenceError, ComposerError

__all__ = [
    "CircularReferenceError",
    "Composer",
    "ComposerError",
    "register_builder",
    "register_builtin_tool",
    "register_provider",
    "register_tool_resolver",
]
