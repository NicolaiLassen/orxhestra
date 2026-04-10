"""Pydantic models that validate the YAML orx specification."""

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

    Exactly one of ``function``, ``mcp``, ``builtin``, ``agent``, or
    ``transfer`` must be set.

    Attributes
    ----------
    function : str, optional
        Dotted import path to a Python callable.
    mcp : MCPConfig, optional
        MCP server connection config that provides this tool.
    builtin : str, optional
        Name of a registered built-in tool (e.g. ``"shell"``, ``"filesystem"``).
    agent : str, optional
        Name of an agent to wrap as a callable tool.
    transfer : TransferConfig, optional
        Transfer tool config for handing off to other agents.
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
            for f in ("function", "mcp", "builtin", "agent", "transfer")
            if getattr(self, f) is not None
        ]
        if len(set_fields) != 1:
            msg = (
                "ToolDef must have exactly one of "
                f"function/mcp/builtin/agent/transfer, got {set_fields}"
            )
            raise ValueError(msg)
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

    @model_validator(mode="after")
    def _validate_main_agent(self) -> ComposeSpec:
        if self.main_agent not in self.agents:
            msg = f"main_agent '{self.main_agent}' not found in agents"
            raise ValueError(msg)
        return self
