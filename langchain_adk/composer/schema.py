"""Pydantic models that validate the YAML compose specification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """LLM provider + model configuration."""

    provider: str = "anthropic"
    name: str = "claude-opus-4-6"
    temperature: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class MCPConfig(BaseModel):
    """MCP server connection — either a URL or a dotted import path to a FastMCP instance."""

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
    skip_summarization: bool = False
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

    type: Literal["plan_react", "task"] = "plan_react"
    tasks: list[dict[str, Any]] | None = None


class SkillItemDef(BaseModel):
    """A skill definition for the agent's skill store."""

    name: str
    description: str = ""
    content: str


class A2ASkillDef(BaseModel):
    """A2A skill advertisement."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class AgentDef(BaseModel):
    """Definition of a single agent in the compose file."""

    type: Literal["llm", "react", "sequential", "parallel", "loop"] = "llm"
    description: str = ""
    model: ModelConfig | None = None
    instructions: str | None = None
    tools: list[str | ToolDef] | None = None
    agents: list[str] | None = None
    planner: PlannerDef | None = None
    output_schema: str | None = None
    max_iterations: int | None = None
    should_continue: str | None = None


class DefaultsConfig(BaseModel):
    """Global defaults inherited by all agents."""

    model: ModelConfig | None = None
    max_iterations: int = 10


class RunnerConfig(BaseModel):
    """Runner configuration."""

    app_name: str = "agent-app"
    session_service: str = "memory"


class ServerConfig(BaseModel):
    """A2A server configuration."""

    app_name: str = "agent-app"
    version: str = "1.0.0"
    url: str = "http://localhost:8000"
    skills: list[A2ASkillDef] = Field(default_factory=list)


class ComposeSpec(BaseModel):
    """Top-level YAML schema for agent composition."""

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    tools: dict[str, ToolDef] = Field(default_factory=dict)
    skills: list[SkillItemDef] = Field(default_factory=list)
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
