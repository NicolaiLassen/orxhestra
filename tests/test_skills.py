"""Tests for Skill, InMemorySkillStore, skill tools, and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from orxhestra.skills import (
    InMemorySkillStore,
    Skill,
    SkillFrontmatter,
    SkillResource,
    discover_skills,
    make_list_skills_tool,
    make_load_skill_resource_tool,
    make_load_skill_tool,
    parse_skill_md,
    scan_skill_directory,
)


@pytest.fixture
def store() -> InMemorySkillStore:
    return InMemorySkillStore(
        skills=[
            Skill(name="python", description="Python coding", content="Write clean Python."),
            Skill(name="sql", description="SQL queries", content="Write efficient SQL."),
        ]
    )


# --- Skill model ---


def test_skill_defaults() -> None:
    skill = Skill(name="test", content="content")
    assert skill.name == "test"
    assert skill.description == ""
    assert skill.id is not None
    assert skill.source == "inline"
    assert skill.frontmatter is None
    assert skill.resources == []
    assert skill.base_path is None


def test_skill_backward_compat() -> None:
    """Existing Skill(name=..., content=...) usage must still work."""
    skill = Skill(name="legacy", content="do stuff", description="A skill")
    assert skill.content == "do stuff"
    assert skill.description == "A skill"
    assert skill.source == "inline"


def test_skill_frontmatter_syncs_description() -> None:
    fm = SkillFrontmatter(name="test", description="From frontmatter")
    skill = Skill(name="test", content="body", frontmatter=fm)
    assert skill.description == "From frontmatter"


def test_skill_frontmatter_does_not_override_explicit_description() -> None:
    fm = SkillFrontmatter(name="test", description="From frontmatter")
    skill = Skill(name="test", content="body", description="Explicit", frontmatter=fm)
    assert skill.description == "Explicit"


# --- SkillFrontmatter ---


def test_skill_frontmatter_defaults() -> None:
    fm = SkillFrontmatter(name="my-skill")
    assert fm.description == ""
    assert fm.license is None
    assert fm.compatibility is None
    assert fm.allowed_tools == []
    assert fm.metadata == {}


# --- SkillResource ---


def test_skill_resource_model() -> None:
    r = SkillResource(path="scripts/run.py", category="script")
    assert r.path == "scripts/run.py"
    assert r.category == "script"


# --- InMemorySkillStore ---


def test_store_add_duplicate_raises() -> None:
    store = InMemorySkillStore()
    skill = Skill(name="s", content="c")
    store.add(skill)
    with pytest.raises(ValueError, match="already registered"):
        store.add(Skill(name="s", content="c2"))


@pytest.mark.asyncio
async def test_store_get_by_name(store: InMemorySkillStore) -> None:
    skill = await store.get_by_name("python")
    assert skill is not None
    assert skill.name == "python"


@pytest.mark.asyncio
async def test_store_get_by_name_not_found(store: InMemorySkillStore) -> None:
    skill = await store.get_by_name("nonexistent")
    assert skill is None


@pytest.mark.asyncio
async def test_store_list_skills(store: InMemorySkillStore) -> None:
    skills = await store.list_skills()
    assert len(skills) == 2


# --- Skill tools ---


@pytest.mark.asyncio
async def test_load_skill_tool_found(store: InMemorySkillStore) -> None:
    tool = make_load_skill_tool(store)
    result = await tool.ainvoke({"name": "python"})
    assert "python" in result.lower()
    assert "Write clean Python" in result


@pytest.mark.asyncio
async def test_load_skill_tool_not_found(store: InMemorySkillStore) -> None:
    tool = make_load_skill_tool(store)
    result = await tool.ainvoke({"name": "nonexistent"})
    assert "not found" in result.lower()
    assert "python" in result


@pytest.mark.asyncio
async def test_load_skill_includes_resources_list() -> None:
    """load_skill should list available resources for directory skills."""
    skill = Skill(
        name="code-review",
        content="Review code carefully.",
        resources=[
            SkillResource(path="scripts/lint.sh", category="script"),
            SkillResource(path="references/style.md", category="reference"),
        ],
        frontmatter=SkillFrontmatter(
            name="code-review",
            allowed_tools=["bash", "read_file"],
            compatibility="Python 3.10+",
        ),
    )
    store = InMemorySkillStore([skill])
    tool = make_load_skill_tool(store)
    result = await tool.ainvoke({"name": "code-review"})
    assert "scripts/lint.sh" in result
    assert "references/style.md" in result
    assert "bash" in result
    assert "Python 3.10+" in result


@pytest.mark.asyncio
async def test_list_skills_tool(store: InMemorySkillStore) -> None:
    tool = make_list_skills_tool(store)
    result = await tool.ainvoke({})
    assert "python" in result
    assert "sql" in result


@pytest.mark.asyncio
async def test_list_skills_tool_empty() -> None:
    tool = make_list_skills_tool(InMemorySkillStore())
    result = await tool.ainvoke({})
    assert "No skills" in result


# --- load_skill_resource tool ---


@pytest.mark.asyncio
async def test_load_skill_resource_tool_found(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.py").write_text("print('hello')")

    skill = Skill(
        name="my-skill",
        content="body",
        resources=[SkillResource(path="scripts/run.py", category="script")],
        base_path=skill_dir,
        source="directory",
    )
    store = InMemorySkillStore([skill])
    tool = make_load_skill_resource_tool(store)
    result = await tool.ainvoke({"skill_name": "my-skill", "resource_path": "scripts/run.py"})
    assert "print('hello')" in result


@pytest.mark.asyncio
async def test_load_skill_resource_tool_not_found() -> None:
    skill = Skill(name="my-skill", content="body", resources=[])
    store = InMemorySkillStore([skill])
    tool = make_load_skill_resource_tool(store)
    result = await tool.ainvoke({"skill_name": "my-skill", "resource_path": "nope.txt"})
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_load_skill_resource_path_traversal(tmp_path: Path) -> None:
    """Path traversal attempts must be blocked."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    # Create a file outside the skill dir
    (tmp_path / "secret.txt").write_text("TOP SECRET")

    skill = Skill(
        name="my-skill",
        content="body",
        resources=[SkillResource(path="../secret.txt", category="reference")],
        base_path=skill_dir,
        source="directory",
    )
    store = InMemorySkillStore([skill])
    result = await store.get_skill_resource("my-skill", "../secret.txt")
    assert result is None


# --- parse_skill_md ---


def test_parse_skill_md_with_frontmatter() -> None:
    text = """\
---
name: code-review
description: Reviews code for quality
license: MIT
allowed-tools: bash read_file
metadata:
  author: test
  version: "1.0"
---

## Instructions

Review all code changes carefully.
"""
    fm, body = parse_skill_md(text)
    assert fm is not None
    assert fm.name == "code-review"
    assert fm.description == "Reviews code for quality"
    assert fm.license == "MIT"
    assert fm.allowed_tools == ["bash", "read_file"]
    assert fm.metadata["author"] == "test"
    assert "Review all code changes" in body


def test_parse_skill_md_no_frontmatter() -> None:
    text = "# Just Markdown\n\nSome instructions."
    fm, body = parse_skill_md(text)
    assert fm is None
    assert "Just Markdown" in body
    assert "Some instructions" in body


def test_parse_skill_md_allowed_tools_as_list() -> None:
    text = """\
---
name: test
allowed-tools:
  - bash
  - read_file
---
body"""
    fm, body = parse_skill_md(text)
    assert fm is not None
    assert fm.allowed_tools == ["bash", "read_file"]


# --- scan_skill_directory ---


def test_scan_skill_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "code-review"
    skill_dir.mkdir()

    # Write SKILL.md
    (skill_dir / "SKILL.md").write_text("""\
---
name: code-review
description: Reviews code for quality
---

Review all code carefully.
""")

    # Create resource directories
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "lint.sh").write_text("#!/bin/bash\nruff check .")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "style.md").write_text("# Style Guide")
    (skill_dir / "assets").mkdir()
    (skill_dir / "assets" / "template.json").write_text("{}")

    skill = scan_skill_directory(skill_dir)
    assert skill.name == "code-review"
    assert skill.description == "Reviews code for quality"
    assert skill.source == "directory"
    assert skill.base_path == skill_dir.resolve()
    assert "Review all code carefully" in skill.content
    assert skill.frontmatter is not None
    assert skill.frontmatter.name == "code-review"

    resource_paths = {r.path for r in skill.resources}
    assert "scripts/lint.sh" in resource_paths
    assert "references/style.md" in resource_paths
    assert "assets/template.json" in resource_paths

    categories = {r.path: r.category for r in skill.resources}
    assert categories["scripts/lint.sh"] == "script"
    assert categories["references/style.md"] == "reference"
    assert categories["assets/template.json"] == "asset"


def test_scan_skill_directory_no_skill_md(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
        scan_skill_directory(tmp_path)


def test_scan_skill_directory_no_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "simple"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("Just some instructions.")

    skill = scan_skill_directory(skill_dir)
    assert skill.name == "simple"  # Falls back to directory name
    assert skill.frontmatter is None
    assert "Just some instructions" in skill.content


# --- discover_skills ---


def test_discover_skills(tmp_path: Path) -> None:
    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    # Skill A
    (skills_dir / "skill-a").mkdir()
    (skills_dir / "skill-a" / "SKILL.md").write_text("""\
---
name: skill-a
description: First skill
---
Instructions for A.""")

    # Skill B
    (skills_dir / "skill-b").mkdir()
    (skills_dir / "skill-b" / "SKILL.md").write_text("""\
---
name: skill-b
description: Second skill
---
Instructions for B.""")

    skills = discover_skills(tmp_path)
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"skill-a", "skill-b"}


def test_discover_skills_no_agents_dir(tmp_path: Path) -> None:
    skills = discover_skills(tmp_path)
    assert skills == []


# --- Composer schema ---


def test_skill_item_def_directory() -> None:
    from orxhestra.composer.schema import SkillItemDef

    item = SkillItemDef(name="test", directory=".agents/skills/test")
    assert item.directory == ".agents/skills/test"
    assert item.content is None
    assert item.mcp is None


def test_skill_item_def_rejects_no_source() -> None:
    from orxhestra.composer.schema import SkillItemDef

    with pytest.raises(ValueError, match="exactly one"):
        SkillItemDef(name="test")


def test_skill_item_def_rejects_multiple_sources() -> None:
    from orxhestra.composer.schema import SkillItemDef

    with pytest.raises(ValueError, match="exactly one"):
        SkillItemDef(name="test", content="x", directory="y")
