"""Persistent project context — loads instructions from multiple AI tool conventions.

Loads context from multiple sources in a defined hierarchy:

**orx-native (user-level):**

1. ``~/.orx/AGENTS.md`` — user-level preferences
2. ``~/.orx/rules/*.md`` — user-level modular rules

**orx-native (project-level):**

3. ``<workspace>/.orx/AGENTS.md`` — project-level context
4. ``<workspace>/.orx/AGENTS.local.md`` — project-level local (gitignored)
5. ``<workspace>/.orx/rules/*.md`` — project-level modular rules
6. ``<workspace>/AGENTS.md`` — project root context
7. ``<workspace>/AGENTS.local.md`` — project root local (gitignored)

**Third-party AI tool conventions (project-level, read-only):**

8.  ``<workspace>/CLAUDE.md`` — Claude Code (Anthropic)
9.  ``<workspace>/.cursorrules`` — Cursor IDE (legacy)
10. ``<workspace>/.cursor/rules/*.md`` — Cursor IDE (modern)
11. ``<workspace>/.windsurfrules`` — Windsurf / Codeium (legacy)
12. ``<workspace>/.windsurf/rules/*.md`` — Windsurf / Codeium (modern)
13. ``<workspace>/.clinerules`` — Cline (single file)
14. ``<workspace>/.clinerules/*.md`` — Cline (directory)
15. ``<workspace>/.github/copilot-instructions.md`` — GitHub Copilot
16. ``<workspace>/CONVENTIONS.md`` — Aider / general
17. ``<workspace>/codex.md`` — OpenAI Codex CLI

The agent can update orx-native files via filesystem tools to persist
learnings across sessions. Third-party files are loaded read-only.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

_USER_DIR: Path = Path.home() / ".orx"
_PROJECT_DIR: str = ".orx"
_AGENTS_FILENAME: str = "AGENTS.md"
_LOCAL_FILENAME: str = "AGENTS.local.md"
_RULES_DIR: str = "rules"


def _load_file(path: Path) -> str | None:
    """Load a file if it exists, return None otherwise."""
    if path.exists() and path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            logger.debug("Failed to read %s", path, exc_info=True)
            return None
    return None


def _load_rules_dir(rules_path: Path) -> list[tuple[str, str]]:
    """Load all *.md files from a rules directory, sorted alphabetically.

    Returns a list of ``(filename, content)`` tuples.
    """
    if not rules_path.is_dir():
        return []
    results: list[tuple[str, str]] = []
    for md_file in sorted(rules_path.glob("*.md")):
        content: str | None = _load_file(md_file)
        if content:
            results.append((md_file.stem, content))
    return results


def load_agents_md(workspace: str) -> str:
    """Load context from all standard locations.

    Sources are loaded in order from most general (user-level) to most
    specific (project root local). All non-empty sources are concatenated
    into a single string injected into agent instructions.

    Parameters
    ----------
    workspace : str
        Workspace directory path to search for project-level files.

    Returns
    -------
    str
        Concatenated memory content, or empty string if none found.
    """
    sections: list[str] = []

    user_content: str | None = _load_file(_USER_DIR / _AGENTS_FILENAME)
    if user_content:
        sections.append(f"## User Preferences\n{user_content}")

    for name, content in _load_rules_dir(_USER_DIR / _RULES_DIR):
        sections.append(f"## User Rule: {name}\n{content}")

    ws: Path = Path(workspace)

    project_orx: str | None = _load_file(
        ws / _PROJECT_DIR / _AGENTS_FILENAME
    )
    if project_orx:
        sections.append(f"## Project Notes\n{project_orx}")

    project_local: str | None = _load_file(
        ws / _PROJECT_DIR / _LOCAL_FILENAME
    )
    if project_local:
        sections.append(f"## Project Local Notes\n{project_local}")

    for name, content in _load_rules_dir(ws / _PROJECT_DIR / _RULES_DIR):
        sections.append(f"## Project Rule: {name}\n{content}")

    project_root: str | None = _load_file(ws / _AGENTS_FILENAME)
    if project_root:
        sections.append(f"## Project Context\n{project_root}")

    root_local: str | None = _load_file(ws / _LOCAL_FILENAME)
    if root_local:
        sections.append(f"## Project Local Context\n{root_local}")

    # ── Third-party AI tool conventions (read-only) ──────────────
    _THIRD_PARTY: list[tuple[str, str | Path]] = [
        ("Claude Code", "CLAUDE.md"),
        ("Cursor Rules", ".cursorrules"),
        ("GitHub Copilot", ".github/copilot-instructions.md"),
        ("Conventions", "CONVENTIONS.md"),
        ("Codex", "codex.md"),
    ]
    for label, rel_path in _THIRD_PARTY:
        content = _load_file(ws / rel_path)
        if content:
            sections.append(f"## {label}\n{content}")

    # Directory-based third-party conventions
    _THIRD_PARTY_DIRS: list[tuple[str, str]] = [
        ("Cursor Rule", ".cursor/rules"),
        ("Windsurf Rule", ".windsurf/rules"),
        ("Cline Rule", ".clinerules"),
    ]
    for label, rel_dir in _THIRD_PARTY_DIRS:
        dir_path: Path = ws / rel_dir
        if dir_path.is_dir():
            for name, content in _load_rules_dir(dir_path):
                sections.append(f"## {label}: {name}\n{content}")
        elif dir_path.is_file():
            # .clinerules can be a single file
            content = _load_file(dir_path)
            if content:
                sections.append(f"## {label}\n{content}")

    # Legacy single-file conventions
    windsurfrules: str | None = _load_file(ws / ".windsurfrules")
    if windsurfrules:
        sections.append(f"## Windsurf Rules\n{windsurfrules}")

    # ── Auto-memory (persistent per-project memories) ────────────
    from orxhestra.memory.file_memory_service import get_memory_dir

    mem_dir = get_memory_dir(workspace)
    mem_index: str | None = _load_file(mem_dir / "MEMORY.md")
    if mem_index:
        sections.append(f"## Auto Memory\n{mem_index}")

    if not sections:
        return ""

    return "# Memory (AGENTS.md)\n\n" + "\n\n".join(sections)


def get_memory_instructions() -> str:
    """Return instructions for the agent about the memory system.

    Returns
    -------
    str
        Markdown-formatted instructions describing the memory hierarchy.
    """
    return """\
# Memory System

You have two memory systems: **context files** (loaded at startup) and
**auto-memory** (persistent memories you save with the `save_memory` tool).

## Auto-Memory (use save_memory tool)

Save memories that will be useful in **future** sessions. There are 4 types:

- **user** — the user's role, preferences, knowledge level
- **feedback** — corrections and confirmations about how to work
- **project** — ongoing work, goals, deadlines, decisions
- **reference** — pointers to external systems (Linear, Slack, Grafana)

When to save: user corrections, confirmed approaches, project decisions,
external system references. Include **Why:** and **How to apply:** lines.

What NOT to save: code patterns (derivable from code), git history
(use git log), debugging recipes (the fix is in the code), ephemeral
task state, API keys or secrets.

Memory records can become stale. Before recommending from a memory,
verify the file/function/flag still exists. Trust current code over
old memories.

## Context Files (loaded at startup, read-only)

- **AGENTS.md** files — project context at various paths
- **CLAUDE.md**, **.cursorrules**, etc. — third-party AI tool conventions

To persist project conventions, use `save_memory` with type "feedback"
or "project" instead of editing AGENTS.md directly.
"""
