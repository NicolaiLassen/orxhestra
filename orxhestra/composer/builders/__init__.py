"""Builder registries for agents, models, and tools.

Three parallel extension points, one per layer of the composer:

- :mod:`orxhestra.composer.builders.agents` — agent-type registry
  (``type:`` keys under YAML ``agents:``).  Register via
  :func:`~orxhestra.composer.register_builder`.
- :mod:`orxhestra.composer.builders.models` — LLM provider registry
  consumed by ``model: { provider: ... }``.  Register via
  :func:`~orxhestra.composer.register_provider`.
- :mod:`orxhestra.composer.builders.tools` — both built-in tool
  factories (``register_builtin_tool``) and custom tool-type
  resolvers (``register_tool_resolver``).

Every builder is a plain async callable, so adding one doesn't
require subclassing anything.  The :class:`Helpers` bag passed to
agent builders bundles the three cross-cutting resolvers
(``resolve_tools``, ``resolve_model``, ``build_agent``) that most
builders need.

See Also
--------
orxhestra.composer : Module-level ``register_*`` entry points.
orxhestra.composer.composer.Composer : Orchestrator that drives all
    three registries.
"""

from orxhestra.composer.builders.agents import Helpers
from orxhestra.composer.builders.agents import register as register_builder

__all__ = ["Helpers", "register_builder"]
