"""Skill loader — parse SKILL.md and scan skill directories.

Implements the Agent Skills Protocol (agentskills.io) loading model:
  - ``parse_skill_md``: Extract YAML frontmatter + markdown body from SKILL.md.
  - ``scan_skill_directory``: Load a single skill directory.
  - ``discover_skills``: Scan ``.agents/skills/`` for skill directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from orxhestra.skills.skill import Skill, SkillFrontmatter, SkillResource

logger = logging.getLogger(__name__)

_RESOURCE_CATEGORIES: dict[str, str] = {
    "scripts": "script",
    "references": "reference",
    "assets": "asset",
}


def parse_skill_md(text: str) -> tuple[SkillFrontmatter | None, str]:
    """Parse a SKILL.md file into frontmatter and body.

    Parameters
    ----------
    text : str
        Raw contents of a SKILL.md file.

    Returns
    -------
    tuple[SkillFrontmatter | None, str]
        Parsed frontmatter (or None if absent) and the markdown body.
    """
    stripped = text.strip()
    if not stripped.startswith("---"):
        return None, stripped

    # Find closing fence
    end_idx = stripped.find("---", 3)
    if end_idx == -1:
        return None, stripped

    yaml_block = stripped[3:end_idx].strip()
    body = stripped[end_idx + 3:].strip()

    try:
        data: dict[str, Any] = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError:
        logger.warning("Failed to parse SKILL.md frontmatter as YAML")
        return None, stripped

    # Map kebab-case keys to snake_case for Pydantic
    if "allowed-tools" in data:
        raw = data.pop("allowed-tools")
        data["allowed_tools"] = raw.split() if isinstance(raw, str) else raw

    frontmatter = SkillFrontmatter(**data)
    return frontmatter, body


def _scan_resources(skill_dir: Path) -> list[SkillResource]:
    """Scan scripts/, references/, assets/ subdirectories for resource files."""
    resources: list[SkillResource] = []
    for subdir_name, category in _RESOURCE_CATEGORIES.items():
        subdir = skill_dir / subdir_name
        if not subdir.is_dir():
            continue
        for file_path in sorted(subdir.rglob("*")):
            if file_path.is_file():
                rel = file_path.relative_to(skill_dir)
                resources.append(SkillResource(path=str(rel), category=category))
    return resources


def scan_skill_directory(path: Path) -> Skill:
    """Load a skill from a directory containing SKILL.md.

    Parameters
    ----------
    path : Path
        Path to the skill directory.

    Returns
    -------
    Skill
        The loaded skill with frontmatter, content, and resource list.

    Raises
    ------
    FileNotFoundError
        If SKILL.md is not found in the directory.
    """
    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        msg = f"SKILL.md not found in {path}"
        raise FileNotFoundError(msg)

    text = skill_md.read_text(encoding="utf-8")
    frontmatter, body = parse_skill_md(text)
    resources = _scan_resources(path)

    name = frontmatter.name if frontmatter else path.name
    description = frontmatter.description if frontmatter else ""

    return Skill(
        name=name,
        description=description,
        content=body,
        frontmatter=frontmatter,
        resources=resources,
        base_path=path.resolve(),
        source="directory",
        metadata=frontmatter.metadata if frontmatter else {},
    )


def discover_skills(root: Path) -> list[Skill]:
    """Scan for skill directories under ``.agents/skills/``.

    Parameters
    ----------
    root : Path
        Project root directory to scan.

    Returns
    -------
    list[Skill]
        All discovered skills.
    """
    skills_dir = root / ".agents" / "skills"
    if not skills_dir.is_dir():
        return []

    skills: list[Skill] = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            skill = scan_skill_directory(skill_md.parent)
            skills.append(skill)
        except Exception as exc:
            logger.warning("Failed to load skill from %s: %s", skill_md.parent, exc)
    return skills
