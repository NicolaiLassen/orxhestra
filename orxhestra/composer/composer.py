"""Build an agent tree from a declarative YAML specification.

Usage::

    from orxhestra.composer import Composer

    agent = Composer.from_yaml("orx.yaml")
    async for event in agent.astream("Hello"):
        print(event.text)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orxhestra.agents.base_agent import BaseAgent
from orxhestra.composer.builders.tools import (
    import_object,
    resolve_agent_tool,
    resolve_builtin,
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
        self._building: set[str] = set()


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
        return await composer._build()

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
        root = await composer._build()
        return await composer._build_runner(root)

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
        root = await composer._build()
        return await composer._build_server(root)


    @classmethod
    async def _runner_async(cls, path: str | Path) -> Runner:
        return await cls.runner_from_yaml_async(path)

    @classmethod
    async def _server_async(cls, path: str | Path) -> Any:
        return await cls.server_from_yaml_async(path)


    @staticmethod
    def _load_spec(path: str | Path) -> ComposeSpec:
        """Load and validate a YAML orx file."""
        import sys

        try:
            import yaml
        except ImportError:
            msg = "PyYAML is required: pip install orxhestra[composer]"
            raise ComposerError(msg) from None

        path = Path(path)
        if not path.exists():
            msg = f"File not found: {path}"
            raise ComposerError(msg)

        # Add the YAML's directory to sys.path so local tools can be imported
        yaml_dir: str = str(path.parent.resolve())
        if yaml_dir not in sys.path:
            sys.path.insert(0, yaml_dir)

        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            msg = f"Expected a YAML mapping, got {type(raw).__name__}"
            raise ComposerError(msg)
        return ComposeSpec.model_validate(raw)


    async def _build(self) -> BaseAgent:
        """Build the full agent tree and return the root."""
        await self._build_agent(self._spec.main_agent)
        return self._agents[self._spec.main_agent]

    async def _build_agent(self, name: str) -> BaseAgent:
        """Recursively build an agent via the builder registry."""
        if name in self._agents:
            return self._agents[name]
        if name in self._building:
            msg = f"Circular agent reference detected: {name}"
            raise CircularReferenceError(msg)

        self._building.add(name)
        agent_def = self._spec.agents.get(name)
        if agent_def is None:
            msg = f"Agent '{name}' referenced but not defined"
            raise ComposerError(msg)

        from orxhestra.composer.builders.agents import Helpers, get

        builder = get(agent_def.type)
        if builder is None:
            msg = f"Unknown agent type: '{agent_def.type}'"
            raise ComposerError(msg)

        helpers = Helpers(
            resolve_tools=self._resolve_tools,
            resolve_model=self._resolve_model,
            build_agent=self._build_agent,
        )

        agent = await builder(
            name, agent_def, self._spec, helpers=helpers
        )

        # Register transfer targets as sub-agents so the SDK can route to them.
        self._register_transfer_sub_agents(agent, agent_def)

        self._agents[name] = agent
        self._building.discard(name)
        return agent

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
        - ``None`` — use defaults
        - ``str`` — reference a named model from the ``models:`` section
        - ``ModelConfig`` — inline override merged with defaults
        """
        default: ModelConfig = self._spec.defaults.model or ModelConfig()

        if agent_def.model is None:
            return default

        # String reference → look up in the models section.
        if isinstance(agent_def.model, str):
            named: ModelConfig | None = self._spec.models.get(agent_def.model)
            if named is None:
                msg = f"Model '{agent_def.model}' not found in models section"
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
                    msg = f"Tool '{ref}' not found in tools section"
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
        """Resolve a single ``ToolDef`` into ``BaseTool`` instance(s)."""
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
        msg = f"ToolDef has no recognized type: {td}"
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

        return Runner(
            agent=root,
            app_name=cfg.app_name,
            session_service=session_svc,
            artifact_service=artifact_svc,
            compaction_config=compaction_config,
            default_config=default_config or None,
        )

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
