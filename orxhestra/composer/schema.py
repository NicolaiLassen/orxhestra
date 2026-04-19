"""Pydantic models that validate the YAML orx specification.

The :class:`ComposeSpec` at the bottom of this module is the root
schema — every ``orx.yaml`` file parses into one via Pydantic's
``model_validate``.  Nested schemas (:class:`AgentDef`,
:class:`ToolDef`, :class:`ModelConfig`, ...) encode the field-level
constraints; post-validators (``@model_validator(mode="after")``)
enforce cross-field invariants that aren't expressible as simple
types, such as "a2a agents must set ``url``" and "composite agents
must set a non-empty ``agents`` list".

Errors surfaced here come from Pydantic directly and are rewrapped
as :class:`~orxhestra.composer.errors.ComposerError` by
:meth:`~orxhestra.composer.composer.Composer._load_spec` so callers
catch a single exception type at the composer boundary.

See Also
--------
orxhestra.composer.composer.Composer : Consumes ``ComposeSpec`` to
    build a live agent tree.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _get_version() -> str:
    """Read version without importing orxhestra (avoids circular import)."""
    from pathlib import Path

    init = Path(__file__).resolve().parent.parent / "__init__.py"
    for line in init.read_text().splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return "0.0.0"


_VERSION: str = _get_version()


class ModelConfig(BaseModel):
    """LLM provider + model configuration.

    Any extra keys (``max_tokens``, ``api_key``, ``base_url``, etc.)
    are forwarded directly to the LangChain model constructor.

    Attributes
    ----------
    provider : str
        LLM provider name (e.g. ``"anthropic"``, ``"openai"``, ``"google"``).
    name : str
        Model identifier passed to the provider.
    temperature : float, optional
        Sampling temperature override.  ``None`` uses provider default.
    """

    provider: str = Field(
        default="anthropic",
        description="LLM provider name (e.g. 'anthropic', 'openai', 'google').",
    )
    name: str = Field(
        default="claude-opus-4-6",
        description="Model identifier passed to the provider.",
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature override. None uses provider default.",
    )

    model_config = {"extra": "allow"}


class MCPConfig(BaseModel):
    """MCP server connection — URL or dotted import path to a FastMCP instance.

    Attributes
    ----------
    url : str, optional
        HTTP URL of a running MCP server.
    server : str, optional
        Dotted import path to a FastMCP server object for in-memory usage.
    """

    url: str | None = Field(
        default=None,
        description="HTTP URL of a running MCP server (e.g. 'http://localhost:8080/mcp').",
    )
    server: str | None = Field(
        default=None,
        description="Dotted import path to a FastMCP server object for in-memory usage.",
    )

    @model_validator(mode="after")
    def _require_one(self) -> MCPConfig:
        if not self.url and not self.server:
            msg = "MCP config must have 'url' or 'server'"
            raise ValueError(msg)
        return self


class TransferConfig(BaseModel):
    """Transfer tool target agents.

    Attributes
    ----------
    targets : list[str]
        Agent names that can be transfer targets.
    """

    targets: list[str] = Field(description="Agent names that can be transfer targets.")


class ToolDef(BaseModel):
    """A single tool definition.

    Exactly one of ``function``, ``mcp``, ``builtin``, ``agent``,
    ``transfer``, or ``custom`` must be set.

    Attributes
    ----------
    function : str, optional
        Dotted import path to a Python callable.
    mcp : MCPConfig, optional
        MCP server connection config that provides this tool.
    builtin : str, optional
        Name of a registered built-in tool (e.g. ``"shell"``,
        ``"filesystem"``).
    agent : str, optional
        Name of an agent to wrap as a callable tool.
    transfer : TransferConfig, optional
        Transfer tool config for handing off to other agents.
    custom : dict[str, Any], optional
        Escape hatch for third-party tool resolvers.  The mapping
        must carry a ``"type"`` key identifying a resolver registered
        via
        :func:`~orxhestra.composer.register_tool_resolver`.  All
        other keys are passed through to the resolver.
    skip_summarization : bool
        If ``True``, skip LLM summarization of this tool's result.
    name : str, optional
        Override the tool name exposed to the LLM.
    description : str, optional
        Override the tool description exposed to the LLM.
    """

    function: str | None = Field(
        default=None,
        description="Dotted import path to a Python callable (e.g. 'mymod.tools.search').",
    )
    mcp: MCPConfig | None = Field(
        default=None,
        description="MCP server connection config that provides this tool.",
    )
    builtin: str | None = Field(
        default=None,
        description="Name of a registered built-in tool (e.g. 'shell', 'filesystem').",
    )
    agent: str | None = Field(
        default=None,
        description="Name of an agent to wrap as a callable tool.",
    )
    transfer: TransferConfig | None = Field(
        default=None,
        description="Transfer tool config for handing off to other agents.",
    )
    custom: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Escape hatch for third-party tool types registered via "
            "register_tool_resolver().  Must include a 'type' key."
        ),
    )
    skip_summarization: bool = Field(
        default=False,
        description="If True, skip LLM summarization of this tool's result.",
    )
    name: str | None = Field(
        default=None,
        description="Override the tool name exposed to the LLM.",
    )
    description: str | None = Field(
        default=None,
        description="Override the tool description exposed to the LLM.",
    )

    @model_validator(mode="after")
    def _exactly_one_type(self) -> ToolDef:
        set_fields = [
            f
            for f in (
                "function", "mcp", "builtin", "agent", "transfer", "custom",
            )
            if getattr(self, f) is not None
        ]
        if len(set_fields) != 1:
            msg = (
                "ToolDef must have exactly one of "
                "function/mcp/builtin/agent/transfer/custom, "
                f"got {set_fields}"
            )
            raise ValueError(msg)
        if self.custom is not None and "type" not in self.custom:
            raise ValueError(
                "ToolDef.custom must include a 'type' key naming a "
                "resolver registered via register_tool_resolver()",
            )
        return self


class PlannerDef(BaseModel):
    """Planner configuration attached to an agent.

    Attributes
    ----------
    type : str
        Planner strategy: ``"plan_react"`` or ``"task"``.
    tasks : list[dict], optional
        Pre-defined tasks for the task planner (ignored by plan_react).
    """

    type: Literal["plan_react", "task"] = Field(
        default="plan_react",
        description="Planner strategy: 'plan_react' for ReAct-style or 'task' for task board.",
    )
    tasks: list[dict[str, Any]] | None = Field(
        default=None,
        description="Pre-defined tasks for the task planner (ignored by plan_react).",
    )


class SkillItemDef(BaseModel):
    """A skill definition — inline ``content``, remote ``mcp``, or ``directory``.

    Attributes
    ----------
    name : str
        Unique skill name used for tool registration.
    description : str
        Short description shown to the LLM for skill selection.
    content : str, optional
        Inline Markdown content for the skill.
    mcp : MCPConfig, optional
        MCP server that provides the skill's tools.
    directory : str, optional
        Path to a directory containing a SKILL.md and resources.
    """

    name: str = Field(description="Unique skill name used for tool registration.")
    description: str = Field(
        default="",
        description="Short description shown to the LLM for skill selection.",
    )
    content: str | None = Field(
        default=None,
        description="Inline Markdown content for the skill.",
    )
    mcp: MCPConfig | None = Field(
        default=None,
        description="MCP server that provides the skill's tools.",
    )
    directory: str | None = Field(
        default=None,
        description="Path to a directory containing a SKILL.md and resources.",
    )

    @model_validator(mode="after")
    def _require_one_source(self) -> SkillItemDef:
        sources = [s for s in ("content", "mcp", "directory") if getattr(self, s)]
        if len(sources) != 1:
            msg = "Skill must have exactly one of 'content', 'mcp', or 'directory'"
            raise ValueError(msg)
        return self


class A2ASkillDef(BaseModel):
    """A2A skill advertisement.

    Attributes
    ----------
    id : str
        Unique skill identifier advertised via A2A.
    name : str
        Human-readable skill name.
    description : str
        Description of what the skill does.
    tags : list[str]
        Tags for skill discovery and filtering.
    """

    id: str = Field(description="Unique skill identifier advertised via A2A.")
    name: str = Field(description="Human-readable skill name.")
    description: str = Field(
        default="",
        description="Description of what the skill does.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for skill discovery and filtering.",
    )


class AgentDef(BaseModel):
    """Definition of a single agent in the orx file.

    Attributes
    ----------
    type : str
        Agent type: ``"llm"``, ``"react"``, ``"sequential"``,
        ``"parallel"``, ``"loop"``, or ``"a2a"``.
    description : str
        Short description used by LLMs for routing decisions.
    url : str, optional
        Remote A2A server URL (only for ``type="a2a"``).
    model : ModelConfig or str, optional
        LLM model config or a named model reference.
    instructions : str, optional
        System instructions prepended to the agent's prompt.
    tools : list, optional
        Tools available to this agent (named refs or inline ToolDef).
    skills : list[str]
        Skill names to load as progressive-disclosure tools.
    agents : list[str], optional
        Sub-agent names for composite agent types.
    planner : PlannerDef, optional
        Planning strategy attached to this agent.
    output_key : str, optional
        Session state key where the agent's final output is stored.
    output_schema : str, optional
        Dotted import path to a Pydantic model for structured output.
    include_contents : str, optional
        Content inclusion mode: ``"default"`` or ``"none"``.
    max_iterations : int, optional
        Max tool-call loop iterations before forced stop.
    should_continue : str, optional
        Dotted import path to a callable that decides loop continuation.
    """

    type: Literal["llm", "react", "sequential", "parallel", "loop", "a2a"] = Field(
        default="llm",
        description="Agent type determining execution strategy.",
    )
    description: str = Field(
        default="",
        description="Short description used by LLMs for routing decisions.",
    )
    url: str | None = Field(
        default=None,
        description="Remote A2A server URL (only for type='a2a').",
    )
    model: ModelConfig | str | None = Field(
        default=None,
        description="LLM model config or a named model reference from the models map.",
    )
    instructions: str | None = Field(
        default=None,
        description="System instructions prepended to the agent's prompt.",
    )
    tools: list[str | ToolDef] | None = Field(
        default=None,
        description="Tools available to this agent (named refs or inline ToolDef).",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Skill names to load as progressive-disclosure tools.",
    )
    agents: list[str] | None = Field(
        default=None,
        description="Sub-agent names registered under this agent.",
    )
    planner: PlannerDef | None = Field(
        default=None,
        description="Planning strategy attached to this agent.",
    )
    output_key: str | None = Field(
        default=None,
        description="Session state key where the agent's final output is stored.",
    )
    output_schema: str | None = Field(
        default=None,
        description="Dotted import path to a Pydantic model for structured output.",
    )
    include_contents: str | None = Field(
        default=None,
        description="Content inclusion mode: 'default' or 'none'.",
    )
    max_iterations: int | None = Field(
        default=None,
        description="Max tool-call loop iterations before forced stop.",
    )
    should_continue: str | None = Field(
        default=None,
        description="Dotted import path to a callable that decides loop continuation.",
    )

    #: Built-in agent types that take a non-empty ``agents`` list.
    _COMPOSITE_TYPES: tuple[str, ...] = ("sequential", "parallel", "loop")

    #: Fields that only make sense on LLM-like agent types
    #: (``llm`` / ``react``).  Used by the post-validator to warn when
    #: they're set on an agent type that will silently ignore them.
    _LLM_ONLY_FIELDS: tuple[str, ...] = (
        "model",
        "instructions",
        "tools",
        "skills",
        "planner",
        "output_schema",
        "output_key",
        "include_contents",
    )

    @model_validator(mode="after")
    def _validate_per_type(self) -> AgentDef:
        """Enforce required-per-type fields and warn about meaningless ones.

        Required fields (hard reject):
        - ``a2a`` agents must set ``url``.
        - ``sequential`` / ``parallel`` / ``loop`` agents must set a
          non-empty ``agents`` list.

        Meaningless fields (``warnings.warn``, not rejection):
        - Composite agents ignore ``model``, ``instructions``,
          ``tools``, ``skills``, ``planner``, ``output_schema``, etc.
        - ``a2a`` agents ignore everything except ``url`` and
          ``description``.
        - Non-loop agents ignore ``should_continue``.

        Unknown agent types — ones registered via
        :func:`~orxhestra.composer.register_builder` — are exempt from
        both checks; custom builders are free to consume any field.

        Returns
        -------
        AgentDef
            ``self``, unchanged.  The validator is side-effect-only.
        """
        import warnings

        BUILTIN_TYPES = {"llm", "react", *self._COMPOSITE_TYPES, "a2a"}
        if self.type not in BUILTIN_TYPES:
            return self

        # Hard requirements.
        if self.type == "a2a" and not self.url:
            raise ValueError("a2a agents must set 'url'")
        if self.type in self._COMPOSITE_TYPES and not self.agents:
            raise ValueError(
                f"{self.type} agents must set a non-empty 'agents' list",
            )

        # Soft warnings — set fields that the builder will silently ignore.
        warn = lambda field, reason: warnings.warn(  # noqa: E731
            f"AgentDef(type={self.type!r}) has '{field}' set but {reason}. "
            f"It will be ignored.",
            stacklevel=2,
        )

        if self.type in self._COMPOSITE_TYPES:
            for field_name in self._LLM_ONLY_FIELDS:
                value = getattr(self, field_name)
                if value:
                    warn(field_name, "composite agents only consume 'agents'")
            if self.url:
                warn("url", "only 'a2a' agents consume 'url'")

        if self.type == "a2a":
            for field_name in ("tools", "instructions", "planner",
                                "output_schema", "agents"):
                value = getattr(self, field_name)
                if value:
                    warn(field_name,
                         "'a2a' agents only consume 'url' and 'description'")

        if self.type != "loop" and self.should_continue:
            warn("should_continue", "only 'loop' agents consume it")

        return self


class DefaultsConfig(BaseModel):
    """Global defaults inherited by all agents.

    Attributes
    ----------
    model : ModelConfig, optional
        Default model config inherited by agents without an explicit model.
    max_iterations : int
        Default max tool-call iterations for all agents.
    """

    model: ModelConfig | None = Field(
        default=None,
        description="Default model config inherited by agents without an explicit model.",
    )
    max_iterations: int = Field(
        default=10,
        description="Default max tool-call iterations for all agents.",
    )


class CompactionConfigDef(BaseModel):
    """Compaction configuration for automatic session history management.

    Attributes
    ----------
    char_threshold : int
        Compact when non-compacted event content exceeds this many
        characters.  Default 100,000 (~25k tokens).
    retention_chars : int
        Always keep the most recent events totalling at least this
        many characters as raw.  Default 20,000 (~5k tokens).
    """

    char_threshold: int = Field(
        default=100_000,
        description=(
            "Compact when non-compacted event content exceeds"
            " this many characters (~25k tokens)."
        ),
    )
    retention_chars: int = Field(
        default=20_000,
        description=(
            "Always keep the most recent events totalling at least"
            " this many characters as raw (~5k tokens)."
        ),
    )


class RunConfigDef(BaseModel):
    """LangChain RunnableConfig — passed to every LLM and tool call.

    Attributes
    ----------
    callbacks : list[str]
        Dotted import paths to LangChain callback handler classes.
    tags : list[str]
        Tags propagated to every LLM and tool call for tracing.
    metadata : dict[str, str]
        Arbitrary key-value metadata attached to every run.
    """

    callbacks: list[str] = Field(
        default_factory=list,
        description="Dotted import paths to LangChain callback handler classes.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags propagated to every LLM and tool call for tracing.",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to every run.",
    )


class RunnerConfig(BaseModel):
    """Runner configuration.

    Attributes
    ----------
    app_name : str
        Application name used to namespace sessions.
    session_service : str
        Session backend: ``"memory"`` or ``"database"``.
    artifact_service : str, optional
        Artifact backend: ``"memory"``, ``"file"``, or ``None``.
    compaction : CompactionConfigDef, optional
        Automatic session history compaction settings.
    config : RunConfigDef, optional
        LangChain RunnableConfig passed to every LLM and tool call.
    """

    app_name: str = Field(
        default="agent-app",
        description="Application name used to namespace sessions.",
    )
    session_service: str = Field(
        default="memory",
        description="Session backend: 'memory' for in-memory or 'database' for persistent.",
    )
    artifact_service: str | None = Field(
        default=None,
        description="Artifact backend: 'memory', 'file', or None to disable.",
    )
    compaction: CompactionConfigDef | None = Field(
        default=None,
        description="Automatic session history compaction settings.",
    )
    config: RunConfigDef | None = Field(
        default=None,
        description="LangChain RunnableConfig passed to every LLM and tool call.",
    )


class IdentityConfig(BaseModel):
    """Ed25519 signing identity for the Runner's agents.

    Attributes
    ----------
    signing_key : str
        Path to a JSON key file (created by ``orx identity init`` or
        :func:`orxhestra.security.crypto.load_or_create_signing_key`).
        Supports ``${VAR}`` environment-variable expansion.
    encryption_password : str, optional
        Password used to decrypt the key file when it is stored
        encrypted.  Supports ``${VAR}`` expansion.
    did_method : str
        DID method to use for the public identity — ``"key"``
        (default, offline) or ``"web"`` (institutional, requires
        ``did`` to be set).
    did : str, optional
        Explicit DID to publish.  Required when ``did_method="web"``.
        Ignored for ``did_method="key"`` since the DID is derived
        from the key.
    """

    signing_key: str = Field(
        description="Path to a JSON key file produced by `orx identity init`.",
    )
    encryption_password: str | None = Field(
        default=None,
        description="Password used to decrypt the key file at rest.",
    )
    did_method: Literal["key", "web"] = Field(
        default="key",
        description="DID method — 'key' (offline) or 'web' (institutional).",
    )
    did: str | None = Field(
        default=None,
        description="Explicit DID to publish. Required when did_method='web'.",
    )

    @model_validator(mode="after")
    def _require_did_for_web(self) -> IdentityConfig:
        if self.did_method == "web" and not self.did:
            raise ValueError("identity.did is required when did_method='web'")
        return self


class TrustConfig(BaseModel):
    """Declarative config for the trust middleware.

    See :class:`~orxhestra.middleware.trust.TrustMiddleware` for how
    the runner consumes these fields.

    Attributes
    ----------
    mode : str
        ``"strict"`` drops events that fail verification, ``"permissive"``
        keeps delivering them with a ``metadata["trust"]`` annotation.
    trusted_dids : list[str]
        Allowlist of accepted signer DIDs.  Empty list means any
        valid signer passes (subject to ``denied_dids``).
    denied_dids : list[str]
        DIDs whose events are always rejected.
    require_chain : bool
        When ``True``, enforce hash-chain continuity per branch.
    allow_unsigned : bool
        When ``False``, every event must carry a valid signature.
    """

    mode: Literal["strict", "permissive"] = Field(
        default="permissive",
        description="Strict drops failing events; permissive annotates them.",
    )
    trusted_dids: list[str] = Field(
        default_factory=list,
        description="Allowlist of accepted signer DIDs.",
    )
    denied_dids: list[str] = Field(
        default_factory=list,
        description="DIDs whose events are always rejected.",
    )
    require_chain: bool = Field(
        default=False,
        description="Enforce hash-chain continuity per branch.",
    )
    allow_unsigned: bool = Field(
        default=True,
        description="When False, every event must be signed.",
    )


class AttestationConfig(BaseModel):
    """Declarative config for the attestation middleware.

    See :class:`~orxhestra.middleware.attestation.AttestationMiddleware`
    for how the runner consumes these fields.

    Attributes
    ----------
    provider : str
        Provider type: ``"noop"``, ``"local"``, or a dotted import
        path to a user-supplied
        :class:`~orxhestra.trust.attestation.protocol.AttestationProvider`.
    path : str, optional
        Path passed to the local provider for on-disk audit logs.
        Required when ``provider="local"``.
    """

    provider: str = Field(
        default="noop",
        description="Provider type: 'noop', 'local', or a dotted import path.",
    )
    path: str | None = Field(
        default=None,
        description="Path to on-disk audit log (required for provider='local').",
    )

    @model_validator(mode="after")
    def _require_path_for_local(self) -> AttestationConfig:
        if self.provider == "local" and not self.path:
            raise ValueError("attestation.path is required when provider='local'")
        return self


class ServerConfig(BaseModel):
    """A2A server configuration.

    Attributes
    ----------
    app_name : str
        Application name exposed in the A2A agent card.
    version : str
        Semantic version advertised in the A2A agent card.
    url : str
        Public URL where the A2A server is reachable.
    skills : list[A2ASkillDef]
        Skills advertised in the A2A agent card.
    """

    app_name: str = Field(
        default="agent-app",
        description="Application name exposed in the A2A agent card.",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version advertised in the A2A agent card.",
    )
    url: str = Field(
        default="http://localhost:8000",
        description="Public URL where the A2A server is reachable.",
    )
    skills: list[A2ASkillDef] = Field(
        default_factory=list,
        description="Skills advertised in the A2A agent card.",
    )


class ComposeSpec(BaseModel):
    """Top-level YAML schema for agent composition.

    Attributes
    ----------
    version : str
        Schema version for forward-compatibility checks.
    defaults : DefaultsConfig
        Global defaults inherited by all agents.
    models : dict[str, ModelConfig]
        Named model configs referenceable by agents.
    tools : dict[str, ToolDef]
        Named tool definitions referenceable by agents.
    skills : dict[str, SkillItemDef]
        Named skill definitions available to agents.
    agents : dict[str, AgentDef]
        All agent definitions keyed by unique name.
    main_agent : str
        Name of the root agent that receives user input.
    runner : RunnerConfig, optional
        Runner configuration for session management.
    server : ServerConfig, optional
        A2A server configuration for remote hosting.
    identity : IdentityConfig, optional
        Ed25519 signing identity applied to the Runner.  When set,
        every event emitted by agents running under the Runner is
        signed with this key.
    trust : TrustConfig, optional
        Declarative :class:`~orxhestra.middleware.trust.TrustMiddleware`
        configuration.  Only registered when ``identity`` is also set
        (verification requires keys).
    attestation : AttestationConfig, optional
        Declarative
        :class:`~orxhestra.middleware.attestation.AttestationMiddleware`
        configuration.  Registered on the Runner when set, regardless
        of whether an identity is configured.
    """

    version: str = Field(
        default=_VERSION,
        description="Schema version for forward-compatibility checks.",
    )
    defaults: DefaultsConfig = Field(
        default_factory=DefaultsConfig,
        description="Global defaults inherited by all agents.",
    )
    models: dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="Named model configs referenceable by agents via string key.",
    )
    tools: dict[str, ToolDef] = Field(
        default_factory=dict,
        description="Named tool definitions referenceable by agents.",
    )
    skills: dict[str, SkillItemDef] = Field(
        default_factory=dict,
        description="Named skill definitions available to agents.",
    )
    agents: dict[str, AgentDef] = Field(
        description="All agent definitions keyed by unique name.",
    )
    main_agent: str = Field(
        description="Name of the root agent that receives user input.",
    )
    runner: RunnerConfig | None = Field(
        default=None,
        description="Runner configuration for session management and execution.",
    )
    server: ServerConfig | None = Field(
        default=None,
        description="A2A server configuration for remote agent hosting.",
    )
    identity: IdentityConfig | None = Field(
        default=None,
        description="Ed25519 signing identity applied to the Runner.",
    )
    trust: TrustConfig | None = Field(
        default=None,
        description="TrustMiddleware configuration (requires identity).",
    )
    attestation: AttestationConfig | None = Field(
        default=None,
        description="AttestationMiddleware configuration.",
    )

    @model_validator(mode="after")
    def _validate_main_agent(self) -> ComposeSpec:
        if self.main_agent not in self.agents:
            msg = f"main_agent '{self.main_agent}' not found in agents"
            raise ValueError(msg)
        return self
