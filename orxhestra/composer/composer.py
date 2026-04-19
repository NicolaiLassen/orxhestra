"""Build an agent tree from a declarative YAML specification.

The :class:`Composer` reads a validated :class:`ComposeSpec`, walks
the ``agents:`` map, and dispatches each one to a registered builder
(see :mod:`orxhestra.composer.builders`).  Tools, models, and skills
are resolved lazily as each agent is constructed; circular refs are
detected; and every error is rewrapped as :class:`ComposerError` so
callers catch a single exception type at the composer boundary.

Public surface:

- :meth:`Composer.from_yaml` / :meth:`from_yaml_async` — one-shot
  "parse a YAML file, return the root agent".
- :meth:`Composer.runner_from_yaml` / :meth:`runner_from_yaml_async`
  — same, but wrap in a :class:`~orxhestra.Runner`.
- :meth:`Composer.server_from_yaml` / :meth:`server_from_yaml_async`
  — same, but wrap in an A2A FastAPI server.
- :meth:`Composer.build` / :meth:`build_runner` / :meth:`build_server`
  — public hooks for callers (the CLI) that construct a
  :class:`Composer` manually and need to mutate the spec between
  construction and build.

Usage::

    from orxhestra.composer import Composer

    agent = Composer.from_yaml("orx.yaml")
    async for event in agent.astream("Hello"):
        print(event.text)

See Also
--------
orxhestra.composer.schema.ComposeSpec : Root YAML schema.
orxhestra.composer.register_builder : Add a custom agent type.
orxhestra.composer.register_provider : Add a custom LLM provider.
orxhestra.composer.register_builtin_tool : Add a custom built-in tool.
orxhestra.composer.register_tool_resolver : Add a whole new tool type
    accessible via ``tools: { ..., custom: { type: ... } }``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.builders.agents import Helpers as _AgentHelpers
from orxhestra.composer.builders.agents import get as _get_agent_builder
from orxhestra.composer.builders.agents import registered_types as _registered_agent_types
from orxhestra.composer.builders.tools import (
    import_object,
    resolve_agent_tool,
    resolve_builtin,
    resolve_custom_tool,
    resolve_function,
    resolve_mcp,
    resolve_transfer,
)
from orxhestra.composer.errors import CircularReferenceError, ComposerError
from orxhestra.composer.schema import (
    AgentDef,
    ComposeSpec,
    ModelConfig,
    ToolDef,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from orxhestra.artifacts.base_artifact_service import BaseArtifactService
    from orxhestra.runner import Runner
    from orxhestra.sessions.base_session_service import BaseSessionService


class Composer:
    """Parse a YAML orx file and build a live agent tree.

    Uses the builder registry (``orxhestra.composer.builders``) to
    delegate construction of each agent type. Custom types can be
    added via ``register_builder``.

    See Also
    --------
    ComposeSpec : Pydantic schema of the YAML file.
    BaseAgent : Root interface of every agent the composer produces.
    Runner : Built by :meth:`runner_from_yaml` when the YAML declares
        a ``runner:`` section.
    """

    def __init__(self, spec: ComposeSpec) -> None:
        """Initialize the composer with a validated specification.

        Parameters
        ----------
        spec : ComposeSpec
            The parsed and validated compose specification.
        """
        self._spec = spec
        self._agents: dict[str, BaseAgent] = {}
        # Ordered stack of agent names currently being built, used for
        # cycle detection *and* for error-message context so the user
        # knows which agent (and which parent chain) triggered a
        # failure deep in the resolution pipeline.
        self._building: list[str] = []
        # Default model — cached once rather than reconstructing a
        # Pydantic model on every ``_resolve_model`` call.
        self._default_model: ModelConfig = spec.defaults.model or ModelConfig()
        # Single shared helpers bag — method references are stable, so
        # we don't need to rebuild this object on every ``_build_agent``
        # call.
        self._helpers: _AgentHelpers = _AgentHelpers(
            resolve_tools=self._resolve_tools,
            resolve_model=self._resolve_model,
            build_agent=self._build_agent,
        )


    @classmethod
    def from_yaml(cls, path: str | Path) -> BaseAgent:
        """Parse a YAML file and return the root agent.

        Parameters
        ----------
        path : str | Path
            Path to the YAML orx file.

        Returns
        -------
        BaseAgent
            The root agent of the composed tree.
        """
        return asyncio.run(cls.from_yaml_async(path))

    @classmethod
    async def from_yaml_async(cls, path: str | Path) -> BaseAgent:
        """Parse a YAML file and return the root agent (async).

        Parameters
        ----------
        path : str | Path
            Path to the YAML orx file.

        Returns
        -------
        BaseAgent
            The root agent of the composed tree.
        """
        spec = cls._load_spec(path)
        composer = cls(spec)
        return await composer.build()

    @classmethod
    def runner_from_yaml(cls, path: str | Path) -> Runner:
        """Parse a YAML file and return a ``Runner``.

        Parameters
        ----------
        path : str | Path
            Path to the YAML orx file.

        Returns
        -------
        Runner
            A configured runner with session and artifact services.

        Raises
        ------
        ComposerError
            If the YAML has no ``runner`` section.
        """
        return asyncio.run(cls._runner_async(path))

    @classmethod
    async def runner_from_yaml_async(cls, path: str | Path) -> Runner:
        """Parse a YAML file and return a ``Runner`` (async).

        Parameters
        ----------
        path : str | Path
            Path to the YAML orx file.

        Returns
        -------
        Runner
            A configured runner with session and artifact services.

        Raises
        ------
        ComposerError
            If the YAML has no ``runner`` section.
        """
        spec = cls._load_spec(path)
        if spec.runner is None:
            msg = "No 'runner' section in YAML"
            raise ComposerError(msg)
        composer = cls(spec)
        root = await composer.build()
        return await composer.build_runner(root)

    @classmethod
    def server_from_yaml(cls, path: str | Path) -> Any:
        """Parse a YAML file and return a FastAPI app.

        Parameters
        ----------
        path : str | Path
            Path to the YAML orx file.

        Returns
        -------
        FastAPI
            A FastAPI application wired to the composed agent tree.

        Raises
        ------
        ComposerError
            If the YAML has no ``server`` section.
        """
        return asyncio.run(cls._server_async(path))

    @classmethod
    async def server_from_yaml_async(cls, path: str | Path) -> Any:
        """Parse a YAML file and return a FastAPI app (async).

        Parameters
        ----------
        path : str | Path
            Path to the YAML orx file.

        Returns
        -------
        FastAPI
            A FastAPI application wired to the composed agent tree.

        Raises
        ------
        ComposerError
            If the YAML has no ``server`` section.
        """
        spec = cls._load_spec(path)
        if spec.server is None:
            msg = "No 'server' section in YAML"
            raise ComposerError(msg)
        composer = cls(spec)
        root = await composer.build()
        return await composer.build_server(root)


    @classmethod
    async def _runner_async(cls, path: str | Path) -> Runner:
        return await cls.runner_from_yaml_async(path)

    @classmethod
    async def _server_async(cls, path: str | Path) -> Any:
        return await cls.server_from_yaml_async(path)


    @staticmethod
    def _load_spec(path: str | Path) -> ComposeSpec:
        """Load and validate a YAML orx file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML file.

        Returns
        -------
        ComposeSpec
            The validated spec.

        Raises
        ------
        ComposerError
            When the file is missing, the YAML is not a mapping, or
            Pydantic validation rejects the spec.  Pydantic's own
            :class:`~pydantic.ValidationError` is caught and rewrapped
            so callers can catch a single exception type at the
            composer boundary.
        """
        import sys

        from pydantic import ValidationError

        try:
            import yaml
        except ImportError:
            msg = "PyYAML is required: pip install orxhestra[composer]"
            raise ComposerError(msg) from None

        path = Path(path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise ComposerError(msg)

        # Add the YAML's directory to sys.path so local tools can be imported.
        yaml_dir: str = str(path.parent.resolve())
        if yaml_dir not in sys.path:
            sys.path.insert(0, yaml_dir)

        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            msg = f"Expected a YAML mapping, got {type(raw).__name__}"
            raise ComposerError(msg)
        try:
            return ComposeSpec.model_validate(raw)
        except ValidationError as exc:
            # Preserve the Pydantic diagnostics but expose a single
            # ComposerError type to callers.
            raise ComposerError(f"Invalid orx spec: {exc}") from exc


    async def build(self) -> BaseAgent:
        """Build the full agent tree from the stored spec.

        Public hook for callers (notably :mod:`orxhestra.cli.builder`)
        that need to construct a :class:`Composer` manually and drive
        the build step themselves — e.g. because they want to mutate
        the spec (inject workspace context, wire up extra tools)
        between construction and the build.

        For the common case, prefer :meth:`from_yaml` /
        :meth:`runner_from_yaml` / :meth:`server_from_yaml`.

        Returns
        -------
        BaseAgent
            The root of the composed agent tree (i.e. ``spec.main_agent``).
        """
        await self._build_agent(self._spec.main_agent)
        return self._agents[self._spec.main_agent]

    async def build_runner(self, root: BaseAgent) -> Runner:
        """Build a :class:`Runner` around an already-constructed tree.

        Pairs with :meth:`build`.  Separated so callers can do work
        between building the agent tree and wrapping it in a runner.

        Parameters
        ----------
        root : BaseAgent
            The already-built root agent (typically the return value
            of :meth:`build`).

        Returns
        -------
        Runner

        Raises
        ------
        ComposerError
            When the underlying spec has no ``runner:`` section.
        """
        if self._spec.runner is None:
            msg = "Compose spec has no 'runner' section"
            raise ComposerError(msg)
        return await self._build_runner(root)

    async def build_server(self, root: BaseAgent) -> Any:
        """Build a FastAPI app around an already-constructed tree.

        Public counterpart of :meth:`_build_server`.  Pairs with
        :meth:`build` so callers can compose the agent tree and the
        server wrapper in separate steps.

        Parameters
        ----------
        root : BaseAgent
            The already-built root agent.

        Returns
        -------
        FastAPI

        Raises
        ------
        ComposerError
            When the underlying spec has no ``server:`` section.
        """
        if self._spec.server is None:
            msg = "Compose spec has no 'server' section"
            raise ComposerError(msg)
        return await self._build_server(root)

    # Back-compat alias — some first-party code still reaches for the
    # private name.  Prefer :meth:`build` in new code.
    async def _build(self) -> BaseAgent:
        """Deprecated alias for :meth:`build` — kept for internal callers."""
        return await self.build()

    async def _build_agent(self, name: str) -> BaseAgent:
        """Recursively build an agent via the builder registry.

        Tracks the in-flight build stack in :attr:`_building` so that
        error messages produced anywhere in the resolution pipeline
        can cite the agent that triggered the failure.
        """
        if name in self._agents:
            return self._agents[name]
        if name in self._building:
            msg = (
                f"Circular agent reference detected: {name} "
                f"(chain: {' -> '.join([*self._building, name])})"
            )
            raise CircularReferenceError(msg)

        self._building.append(name)
        try:
            agent_def = self._spec.agents.get(name)
            if agent_def is None:
                msg = (
                    f"Agent '{name}' referenced but not defined in agents: "
                    f"{self._context_suffix()}"
                )
                raise ComposerError(msg)

            builder = _get_agent_builder(agent_def.type)
            if builder is None:
                msg = (
                    f"Unknown agent type: '{agent_def.type}' on agent "
                    f"'{name}' (registered types: "
                    f"{', '.join(_registered_agent_types())})"
                )
                raise ComposerError(msg)

            agent = await builder(
                name, agent_def, self._spec, helpers=self._helpers
            )

            # Register transfer targets as sub-agents so the SDK can route to them.
            self._register_transfer_sub_agents(agent, agent_def)

            self._agents[name] = agent
            return agent
        finally:
            # Stack is always balanced since `append` is the first line
            # of the try block — no need to guard with try/except.
            self._building.pop()

    def _context_suffix(self) -> str:
        """Format the in-flight build stack as a breadcrumb.

        Returns
        -------
        str
            ``"while building agent 'root -> child'"`` when a build is
            in progress, otherwise an empty string.  Used by
            resolution helpers to append YAML context to error
            messages so the user knows *where* the failure originated.
        """
        if not self._building:
            return ""
        path = " -> ".join(self._building)
        return f"while building agent '{path}'"

    def _register_transfer_sub_agents(
        self, agent: BaseAgent, agent_def: AgentDef
    ) -> None:
        """Register transfer targets as sub-agents for routing."""
        if not agent_def.tools:
            return
        for ref in agent_def.tools:
            td = ref if isinstance(ref, ToolDef) else self._spec.tools.get(ref)
            if td is None or td.transfer is None:
                continue
            for target_name in td.transfer.targets:
                target = self._agents.get(target_name)
                if target is not None:
                    agent.register_sub_agent(target)


    def _resolve_model(self, agent_def: AgentDef) -> ModelConfig:
        """Resolve the model config for an agent.

        The agent ``model`` field can be:

        - ``None`` — fall back to the cached default
          (``spec.defaults.model`` or a bare ``ModelConfig()``).
        - ``str`` — look up a named model from the YAML ``models:``
          section.
        - ``ModelConfig`` — inline override merged over the default,
          preserving default values for any keys the override leaves
          unset.

        Parameters
        ----------
        agent_def : AgentDef
            The agent definition whose model is being resolved.

        Returns
        -------
        ModelConfig

        Raises
        ------
        ComposerError
            When a string reference doesn't match any entry in the
            ``models:`` section.
        """
        default = self._default_model

        if agent_def.model is None:
            return default

        # String reference → look up in the models section.
        if isinstance(agent_def.model, str):
            named: ModelConfig | None = self._spec.models.get(agent_def.model)
            if named is None:
                known = ", ".join(sorted(self._spec.models)) or "<none>"
                msg = (
                    f"Model '{agent_def.model}' not found in models section "
                    f"(known: {known}) {self._context_suffix()}"
                )
                raise ComposerError(msg)
            return named

        # Inline ModelConfig → merge non-None fields over defaults.
        overrides: dict[str, Any] = {
            k: v
            for k, v in agent_def.model.model_dump().items()
            if v is not None
        }
        return default.model_copy(update=overrides)


    async def _resolve_tools(self, agent_def: AgentDef) -> list[BaseTool]:
        """Resolve all tool references for an agent."""
        if not agent_def.tools:
            return []

        tools: list[BaseTool] = []
        for ref in agent_def.tools:
            if isinstance(ref, str):
                tool_def = self._spec.tools.get(ref)
                if tool_def is None:
                    known = ", ".join(sorted(self._spec.tools)) or "<none>"
                    msg = (
                        f"Tool '{ref}' not found in tools section "
                        f"(known: {known}) {self._context_suffix()}"
                    )
                    raise ComposerError(msg)
                resolved = await self._resolve_tool_def(tool_def)
            else:
                resolved = await self._resolve_tool_def(ref)

            if isinstance(resolved, list):
                tools.extend(resolved)
            else:
                tools.append(resolved)
        return tools

    async def _resolve_tool_def(
        self, td: ToolDef
    ) -> BaseTool | list[BaseTool]:
        """Resolve a single ``ToolDef`` into ``BaseTool`` instance(s).

        Dispatch order matches the ``ToolDef`` field order: ``function``,
        ``mcp``, ``builtin``, ``agent``, ``transfer``.  Errors are
        enriched with :meth:`_context_suffix` so the user can trace
        which YAML agent referenced the broken tool.

        Parameters
        ----------
        td : ToolDef
            The tool definition to resolve.

        Returns
        -------
        BaseTool or list[BaseTool]
            A single LangChain tool, or a list when the MCP resolver
            expands one ``ToolDef`` into multiple tools.

        Raises
        ------
        ComposerError
            When the ``ToolDef`` has no recognised type.
        """
        if td.function:
            return resolve_function(td.function, td.name, td.description)
        if td.mcp:
            return await resolve_mcp(td.mcp.url, td.mcp.server)
        if td.builtin:
            return resolve_builtin(td.builtin)
        if td.agent:
            agent = await self._build_agent(td.agent)
            return resolve_agent_tool(agent, td.skip_summarization)
        if td.transfer:
            targets: list[BaseAgent] = []
            for target_name in td.transfer.targets:
                targets.append(await self._build_agent(target_name))
            return resolve_transfer(targets)
        if td.custom:
            return await resolve_custom_tool(td.custom)
        msg = (
            f"ToolDef has no recognised type (set one of "
            f"function/mcp/builtin/agent/transfer/custom) "
            f"{self._context_suffix()}"
        )
        raise ComposerError(msg)


    async def _build_runner(self, root: BaseAgent) -> Runner:
        """Build a ``Runner`` from the spec's runner config."""
        from orxhestra.runner import Runner
        from orxhestra.sessions.compaction import CompactionConfig

        cfg = self._spec.runner
        assert cfg is not None
        session_svc = await self._build_session_service(cfg.session_service)

        compaction_config: CompactionConfig | None = None
        if cfg.compaction is not None:
            # Resolve an LLM for summarization from the default model config
            model = None
            if self._spec.defaults.model:
                from orxhestra.composer.builders.models import create

                model_cfg = self._spec.defaults.model
                model = create(model_cfg.provider, model_cfg.name)

            compaction_config = CompactionConfig(
                char_threshold=cfg.compaction.char_threshold,
                retention_chars=cfg.compaction.retention_chars,
                model=model,
            )

        artifact_svc = self._build_artifact_service(cfg.artifact_service)

        # Resolve run config (callbacks, tags, metadata)
        default_config: dict = {}
        if cfg.config is not None:
            from orxhestra.composer.builders.tools import import_object

            if cfg.config.callbacks:
                callbacks = []
                for path in cfg.config.callbacks:
                    obj = import_object(path)
                    if isinstance(obj, type):
                        obj = obj()
                    callbacks.append(obj)
                default_config["callbacks"] = callbacks

            if cfg.config.tags:
                default_config["tags"] = cfg.config.tags

            if cfg.config.metadata:
                default_config["metadata"] = cfg.config.metadata

        signing_key, signer_did = self._resolve_identity()
        middleware = self._build_middleware(signing_key, signer_did)

        runner = Runner(
            agent=root,
            app_name=cfg.app_name,
            session_service=session_svc,
            artifact_service=artifact_svc,
            compaction_config=compaction_config,
            default_config=default_config or None,
            middleware=middleware or None,
        )

        if signing_key is not None and signer_did:
            self._propagate_identity(root, signing_key, signer_did)

        return runner

    def _resolve_identity(self) -> tuple[Any, str]:
        """Load the configured signing key, returning ``(key, did)``.

        Expands ``${VAR}`` env-var references in ``signing_key`` and
        ``encryption_password`` so YAML can stay secrets-free.  When
        ``did_method='key'`` the derived DID overrides any ``did``
        set in the YAML, since ``did:key`` is a pure function of the
        key bytes.

        Returns
        -------
        tuple[Any, str]
            The ``Ed25519PrivateKey`` and its DID.  ``(None, "")`` when
            no ``identity:`` section is configured.
        """
        cfg = self._spec.identity
        if cfg is None:
            return None, ""

        import os

        def _expand(value: str | None) -> str | None:
            if value is None:
                return None
            return os.path.expandvars(value)

        key_file = _expand(cfg.signing_key)
        password = _expand(cfg.encryption_password)

        from orxhestra.security.crypto import load_or_create_signing_key

        assert key_file is not None
        signing_key, derived_did = load_or_create_signing_key(
            key_file, encryption_password=password,
        )

        if cfg.did_method == "web":
            # Trust the user-supplied did:web — its public key lives in
            # the hosted DID document, not in the local key file.
            return signing_key, cfg.did or derived_did
        return signing_key, derived_did

    def _build_middleware(
        self, signing_key: Any, signer_did: str,
    ) -> list[Any]:
        """Construct the opt-in trust + attestation middleware stack.

        Parameters
        ----------
        signing_key : Any
            Identity key resolved by :meth:`_resolve_identity`.
        signer_did : str
            DID corresponding to ``signing_key``.

        Returns
        -------
        list[Middleware]
            Ordered middleware list ready for :class:`Runner`.  Empty
            when no ``trust:`` or ``attestation:`` block is set.
        """
        middleware: list[Any] = []

        trust_cfg = self._spec.trust
        if trust_cfg is not None and signing_key is not None:
            from orxhestra.middleware import TrustMiddleware
            from orxhestra.security.did import (
                CompositeResolver,
                DidKeyResolver,
                DidWebResolver,
            )
            from orxhestra.trust import TrustPolicy

            policy = TrustPolicy(
                mode=trust_cfg.mode,
                trusted_dids=set(trust_cfg.trusted_dids),
                denied_dids=set(trust_cfg.denied_dids),
                require_chain=trust_cfg.require_chain,
                allow_unsigned=trust_cfg.allow_unsigned,
            )
            resolver = CompositeResolver([DidKeyResolver(), DidWebResolver()])
            middleware.append(TrustMiddleware(policy, resolver))

        att_cfg = self._spec.attestation
        if att_cfg is not None:
            from orxhestra.middleware import AttestationMiddleware
            from orxhestra.trust import (
                LocalAttestationProvider,
                NoOpAttestationProvider,
            )

            provider: Any
            if att_cfg.provider == "noop":
                provider = NoOpAttestationProvider()
            elif att_cfg.provider == "local":
                assert att_cfg.path is not None
                import os

                path = os.path.expandvars(att_cfg.path)
                if signing_key is None or not signer_did:
                    raise ComposerError(
                        "attestation.provider='local' requires an identity: block."
                    )
                provider = LocalAttestationProvider(path, signing_key, signer_did)
            else:
                provider = import_object(att_cfg.provider)
                if isinstance(provider, type):
                    provider = provider()
            middleware.append(AttestationMiddleware(provider))

        return middleware

    def _propagate_identity(
        self, root: BaseAgent, signing_key: Any, signer_did: str,
    ) -> None:
        """Stamp ``signing_key``/``signer_did`` onto every agent in the tree.

        Walks ``root`` and its descendants via ``sub_agents`` and
        sets the identity only when the agent does not already have
        one — letting per-agent overrides in user code win.

        Parameters
        ----------
        root : BaseAgent
            Entry point into the agent tree.
        signing_key : Any
            Key to attach.
        signer_did : str
            DID to attach.
        """
        stack: list[BaseAgent] = [root]
        seen: set[int] = set()
        while stack:
            agent = stack.pop()
            if id(agent) in seen:
                continue
            seen.add(id(agent))
            if agent.signing_key is None and not agent.signing_did:
                agent.signing_key = signing_key
                agent.signing_did = signer_did
            stack.extend(agent.sub_agents)

    async def _build_server(self, root: BaseAgent) -> Any:
        """Build a FastAPI app from the spec's server config."""
        from orxhestra.a2a.server import A2AServer
        from orxhestra.a2a.types import AgentSkill

        cfg = self._spec.server
        assert cfg is not None

        svc_name = (
            self._spec.runner.session_service if self._spec.runner else "memory"
        )
        session_svc = await self._build_session_service(svc_name)

        skills = [
            AgentSkill(
                id=s.id, name=s.name, description=s.description, tags=s.tags
            )
            for s in cfg.skills
        ]
        server = A2AServer(
            root,
            session_service=session_svc,
            app_name=cfg.app_name,
            version=cfg.version,
            url=cfg.url,
            skills=skills,
        )
        return server.as_fastapi_app()

    @staticmethod
    async def _build_session_service(name: str) -> BaseSessionService:
        """Instantiate a session service by name or import path."""
        if name == "memory":
            from orxhestra.sessions.in_memory_session_service import (
                InMemorySessionService,
            )

            return InMemorySessionService()
        if name.startswith(("sqlite", "postgresql", "mysql")):
            from orxhestra.sessions.database_session_service import (
                DatabaseSessionService,
            )

            svc = DatabaseSessionService(name)
            await svc.initialize()
            return svc
        cls = import_object(name)
        return cls()

    @staticmethod
    def _build_artifact_service(
        name: str | None,
    ) -> BaseArtifactService | None:
        """Instantiate an artifact service by name or import path.

        Supported names:

        - ``"memory"`` — in-memory (dev/test)
        - ``"file"`` or ``"file:/path/to/dir"`` — filesystem-backed
        - ``None`` — no artifact service
        - dotted import path — custom implementation
        """
        if name is None:
            return None
        if name == "memory":
            from orxhestra.artifacts.in_memory_artifact_service import (
                InMemoryArtifactService,
            )

            return InMemoryArtifactService()
        if name == "file" or name.startswith("file:"):
            from orxhestra.artifacts.file_artifact_service import (
                FileArtifactService,
            )

            root: str = name.removeprefix("file:") if ":" in name else ".artifacts"
            return FileArtifactService(root)
        cls = import_object(name)
        return cls()
