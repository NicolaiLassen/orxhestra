"""Integrations — adapters that plug external tool ecosystems into orxhestra.

Each sub-package wraps a third-party protocol or SDK and exposes it
as first-class orxhestra primitives (tools, agents, resolvers).

Current sub-packages:

- :mod:`~orxhestra.integrations.mcp` — Model Context Protocol: fetch
  a remote tool server's catalogue and expose each entry as a
  LangChain :class:`~langchain_core.tools.BaseTool`.

Integrations are optional and gated behind extras in
``pyproject.toml`` — installing the core package does not pull in
their dependencies.
"""
