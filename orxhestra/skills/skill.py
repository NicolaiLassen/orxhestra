"""Skill models — Agent Skills Protocol (agentskills.io) compatible.

A skill is a named knowledge package with progressive disclosure:
  - L1 (Catalog): ``name`` + ``description`` — loaded at session start.
  - L2 (Instructions): Full SKILL.md body — loaded on activation.
  - L3 (Resources): Scripts, references, assets — loaded on demand.

Backward compatible: ``Skill(name=..., content=...)`` still works for
inline skills defined in YAML or code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class SkillFrontmatter(BaseModel):
    """YAML frontmatter parsed from a SKILL.md file.

    Attributes
    ----------
    name : str
        Kebab-case identifier (must match directory name).
    description : str
        What the skill does and when to use it.
    license : str, optional
        SPDX license identifier.
    compatibility : str, optional
        Environment requirements.
    allowed_tools : list[str]
        Tools pre-approved when this skill is active.
    metadata : dict[str, Any]
        Arbitrary key-value pairs (author, version, etc.).
    """

    name: str
    description: str = ""
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillResource(BaseModel):
    """A resource file within a skill directory.

    Attributes
    ----------
    path : str
        Relative path within the skill directory (e.g. ``scripts/run.py``).
    category : str
        One of ``script``, ``reference``, or ``asset``.
    """

    path: str
    category: Literal["script", "reference", "asset"]


class Skill(BaseModel):
    """A named knowledge package with optional resources.

    Supports three sources:
      - ``inline``: Content provided directly (backward compatible).
      - ``directory``: Loaded from a SKILL.md directory on disk.
      - ``mcp``: Fetched from a remote MCP server.

    Attributes
    ----------
    id : str
        Unique skill identifier.
    name : str
        Short identifier used by the LLM to reference this skill.
    description : str
        One-line summary shown in the agent's skill catalog (L1).
    content : str
        The full instruction text (L2). For directory skills this is
        the SKILL.md body below the frontmatter.
    metadata : dict[str, Any]
        Optional tags/version/source info.
    frontmatter : SkillFrontmatter, optional
        Parsed YAML frontmatter from SKILL.md.
    resources : list[SkillResource]
        Available resource files (L3) within the skill directory.
    base_path : Path, optional
        Filesystem path to the skill directory (for L3 loading).
    source : str
        How this skill was loaded: ``inline``, ``directory``, or ``mcp``.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    frontmatter: SkillFrontmatter | None = None
    resources: list[SkillResource] = Field(default_factory=list)
    base_path: Path | None = None
    source: Literal["inline", "directory", "mcp"] = "inline"

    @model_validator(mode="after")
    def _sync_frontmatter(self) -> Skill:
        """Sync description from frontmatter if not set directly."""
        if self.frontmatter and not self.description:
            self.description = self.frontmatter.description
        return self
