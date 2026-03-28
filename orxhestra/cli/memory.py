"""AGENTS.md memory system - persistent project context across sessions.

Loads AGENTS.md files from the workspace and user home directory.
The agent can update these files via the filesystem tools to persist
learnings across sessions.
"""

from __future__ import annotations

from pathlib import Path

# Standard locations for AGENTS.md files
_USER_AGENTS_MD: Path = Path.home() / ".orx" / "AGENTS.md"
_PROJECT_DIR: str = ".orx"
_AGENTS_FILENAME: str = "AGENTS.md"


def _load_file(path: Path) -> str | None:
    """Load a file if it exists, return None otherwise."""
    if path.exists() and path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return None


def load_agents_md(workspace: str) -> str:
    """Load AGENTS.md content from all standard locations.

    Sources (later overrides earlier):
    1. User-level: ~/.orx/AGENTS.md
    2. Project-level: <workspace>/.orx/AGENTS.md
    3. Project root: <workspace>/AGENTS.md
    """
    sections: list[str] = []

    # User-level
    user_content: str | None = _load_file(_USER_AGENTS_MD)
    if user_content:
        sections.append(f"## User Preferences\n{user_content}")

    # Project .orx directory
    project_orx: str | None = _load_file(
        Path(workspace) / _PROJECT_DIR / _AGENTS_FILENAME
    )
    if project_orx:
        sections.append(f"## Project Notes\n{project_orx}")

    # Project root
    project_root: str | None = _load_file(Path(workspace) / _AGENTS_FILENAME)
    if project_root:
        sections.append(f"## Project Context\n{project_root}")

    if not sections:
        return ""

    return "# Memory (AGENTS.md)\n\n" + "\n\n".join(sections)


def get_memory_instructions() -> str:
    """Return instructions for the agent about the memory system."""
    return """\
# Memory System
You have access to persistent memory via AGENTS.md files:
- **~/.orx/AGENTS.md** - Your personal preferences and notes (user-level)
- **.orx/AGENTS.md** - Project-specific context (architecture, conventions)
- **AGENTS.md** - Project root context

When you learn something important about the project (build commands, code
conventions, architecture decisions, user preferences), update the appropriate
AGENTS.md file using write_file or edit_file. This persists across sessions.

What to save: build/test commands, code style, architecture notes, tool preferences.
What NOT to save: API keys, passwords, ephemeral task state.
"""
