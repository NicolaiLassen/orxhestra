"""Pydantic models that validate the YAML orx specification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """LLM provider + model configuration.

    Any extra keys (``max_tokens``, ``api_key``, ``base_url``, etc.)
    are forwarded directly to the LangChain model constructor.
    """

    provider: str = Field(default="anthropic")
    name: str = Field(default="claude-opus-4-6")
    temperature: float | None = None

    model_config = {"extra": "allow"}


class MCPConfig(BaseModel):
    """MCP server connection — URL or dotted import path to a FastMCP instance."""

    url: str | None = None
    server: str | None = None

    @model_validator(mode="after")
    def _require_one(self) -> MCPConfig:
        if not self.url and not self.server:
            msg = "MCP config must have 'url' or 'server'"
            raise ValueError(msg)
        return self


class TransferConfig(BaseModel):
    """Transfer tool target agents."""

    targets: list[str]


class ToolDef(BaseModel):
    """A single tool definition.

    Exactly one of ``function``, ``mcp``, ``builtin``, ``agent``, or
    ``transfer`` must be set.
    """

    function: str | None = None
    mcp: MCPConfig | None = None
    builtin: str | None = None
    agent: str | None = None
    transfer: TransferConfig | None = None
    skip_summarization: bool = Field(default=False)
    name: str | None = None
    description: str | None = None

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
    """Planner configuration attached to an agent."""

    type: Literal["plan_react", "task"] = Field(default="plan_react")
    tasks: list[dict[str, Any]] | None = None


class SkillItemDef(BaseModel):
    """A skill definition — inline ``content``, remote ``mcp``, or ``directory``."""

    name: str
    description: str = Field(default="")
    content: str | None = None
    mcp: MCPConfig | None = None
    directory: str | None = None

    @model_validator(mode="after")
    def _require_one_source(self) -> SkillItemDef:
        sources = [s for s in ("content", "mcp", "directory") if getattr(self, s)]
        if len(sources) != 1:
            msg = "Skill must have exactly one of 'content', 'mcp', or 'directory'"
            raise ValueError(msg)
        return self


class A2ASkillDef(BaseModel):
    """A2A skill advertisement."""

    id: str
    name: str
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)


class AgentDef(BaseModel):
    """Definition of a single agent in the orx file."""

    type: Literal["llm", "react", "sequential", "parallel", "loop", "a2a"] = Field(default="llm")
    description: str = Field(default="")
    url: str | None = None
    model: ModelConfig | str | None = None
    instructions: str | None = None
    tools: list[str | ToolDef] | None = None
    skills: list[str] = Field(default_factory=list)
    agents: list[str] | None = None
    planner: PlannerDef | None = None
    output_key: str | None = None
    output_schema: str | None = None
    include_contents: str | None = None  # "default" or "none"
    max_iterations: int | None = None
    should_continue: str | None = None


class DefaultsConfig(BaseModel):
    """Global defaults inherited by all agents."""

    model: ModelConfig | None = None
    max_iterations: int = Field(default=10)


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

    char_threshold: int = Field(default=100_000)
    retention_chars: int = Field(default=20_000)


class RunConfigDef(BaseModel):
    """LangChain RunnableConfig — passed to every LLM and tool call."""

    callbacks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class RunnerConfig(BaseModel):
    """Runner configuration."""

    app_name: str = Field(default="agent-app")
    session_service: str = Field(default="memory")
    artifact_service: str | None = None
    compaction: CompactionConfigDef | None = None
    config: RunConfigDef | None = None


class ServerConfig(BaseModel):
    """A2A server configuration."""

    app_name: str = Field(default="agent-app")
    version: str = Field(default="1.0.0")
    url: str = Field(default="http://localhost:8000")
    skills: list[A2ASkillDef] = Field(default_factory=list)


class ComposeSpec(BaseModel):
    """Top-level YAML schema for agent composition."""

    version: str = Field(default="0.0.8")
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    tools: dict[str, ToolDef] = Field(default_factory=dict)
    skills: dict[str, SkillItemDef] = Field(default_factory=dict)
    agents: dict[str, AgentDef]
    main_agent: str
    runner: RunnerConfig | None = None
    server: ServerConfig | None = None

    @model_validator(mode="after")
    def _validate_main_agent(self) -> ComposeSpec:
        if self.main_agent not in self.agents:
            msg = f"main_agent '{self.main_agent}' not found in agents"
            raise ValueError(msg)
        return self
