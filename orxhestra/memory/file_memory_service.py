"""File-based memory service — individual markdown files with YAML frontmatter.

Stores memories as individual ``.md`` files in a per-project directory::

    ~/.orx/projects/<sanitized-cwd>/memory/
    ├── MEMORY.md              ← auto-maintained index
    ├── user_role.md
    ├── feedback_testing.md
    └── project_deadline.md

Each file has YAML frontmatter (name, description, type, created) and
free-form markdown body content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orxhestra.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from orxhestra.memory.memory import Memory

_ORX_DIR: Path = Path.home() / ".orx"
_PROJECTS_DIR: str = "projects"
_MEMORY_DIR: str = "memory"
_INDEX_FILE: str = "MEMORY.md"
_MAX_INDEX_LINES: int = 200

MEMORY_TYPES: list[str] = ["user", "feedback", "project", "reference"]


def sanitize_workspace(workspace: str) -> str:
    """Sanitize a workspace path for use as a directory name."""
    return re.sub(r"[^\w\-.]", "-", workspace.strip("/"))


def get_memory_dir(workspace: str) -> Path:
    """Return the per-project memory directory path.

    Parameters
    ----------
    workspace : str
        Workspace directory path.

    Returns
    -------
    Path
        ``~/.orx/projects/<sanitized-cwd>/memory/``
    """
    sanitized: str = sanitize_workspace(workspace)
    return _ORX_DIR / _PROJECTS_DIR / sanitized / _MEMORY_DIR


def _slugify(name: str) -> str:
    """Convert a memory name to a safe filename slug."""
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return slug[:80] or "memory"


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from markdown text.

    Returns
    -------
    tuple[dict[str, str], str]
        (frontmatter dict, body content).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    fm_text = text[3:end].strip()
    body = text[end + 3:].strip()

    frontmatter: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()

    return frontmatter, body


@dataclass
class MemoryHeader:
    """Parsed header from a memory file's frontmatter."""

    filename: str
    filepath: Path
    name: str
    description: str
    memory_type: str
    created: str
    mtime: float


def scan_memory_files(memory_dir: Path) -> list[MemoryHeader]:
    """Scan a memory directory and return headers from all .md files.

    Parameters
    ----------
    memory_dir : Path
        Directory to scan.

    Returns
    -------
    list[MemoryHeader]
        Headers sorted newest-first.
    """
    if not memory_dir.is_dir():
        return []

    headers: list[MemoryHeader] = []
    for md_file in memory_dir.glob("*.md"):
        if md_file.name == _INDEX_FILE:
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except OSError:
            continue

        fm, _ = _parse_frontmatter(text)
        headers.append(
            MemoryHeader(
                filename=md_file.name,
                filepath=md_file,
                name=fm.get("name", md_file.stem),
                description=fm.get("description", ""),
                memory_type=fm.get("type", ""),
                created=fm.get("created", ""),
                mtime=md_file.stat().st_mtime,
            )
        )

    headers.sort(key=lambda h: h.mtime, reverse=True)
    return headers


def update_memory_index(memory_dir: Path) -> None:
    """Rewrite MEMORY.md from current memory files.

    Caps the index at 200 lines.
    """
    headers = scan_memory_files(memory_dir)
    lines: list[str] = ["# Memory Index", ""]

    for h in headers:
        type_tag = f"[{h.memory_type}] " if h.memory_type else ""
        desc = f" — {h.description}" if h.description else ""
        lines.append(f"- {type_tag}**{h.name}**{desc}")

        if len(lines) >= _MAX_INDEX_LINES:
            lines.append(
                f"(truncated — {len(headers)} memories, "
                f"showing first {_MAX_INDEX_LINES})"
            )
            break

    index_path = memory_dir / _INDEX_FILE
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_memory_file(
    memory_dir: Path,
    *,
    name: str,
    content: str,
    memory_type: str = "",
    description: str = "",
) -> Path:
    """Write a memory file with frontmatter and update the index.

    Parameters
    ----------
    memory_dir : Path
        Memory directory.
    name : str
        Human-readable memory name.
    content : str
        Memory body content (markdown).
    memory_type : str
        One of: user, feedback, project, reference.
    description : str
        One-line description for the index.

    Returns
    -------
    Path
        Path to the created file.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)

    slug = _slugify(name)
    if memory_type:
        slug = f"{memory_type}_{slug}"
    filepath = memory_dir / f"{slug}.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fm_lines = [
        "---",
        f"name: {name}",
        f"description: {description}",
        f"type: {memory_type}",
        f"created: {now}",
        "---",
        "",
        content,
        "",
    ]
    filepath.write_text("\n".join(fm_lines), encoding="utf-8")
    update_memory_index(memory_dir)
    return filepath


def delete_memory_file(memory_dir: Path, name: str) -> bool:
    """Delete a memory file by name and update the index.

    Parameters
    ----------
    memory_dir : Path
        Memory directory.
    name : str
        Memory name to delete (matches frontmatter name or filename stem).

    Returns
    -------
    bool
        True if a file was deleted.
    """
    headers = scan_memory_files(memory_dir)
    for h in headers:
        if h.name.lower() == name.lower() or h.filename == name:
            h.filepath.unlink(missing_ok=True)
            update_memory_index(memory_dir)
            return True
    return False


class FileMemoryService(BaseMemoryService):
    """File-based memory service using individual markdown files.

    Parameters
    ----------
    memory_dir : Path
        Directory for storing memory files.
    """

    def __init__(self, memory_dir: Path) -> None:
        self._dir = memory_dir

    async def add_session_to_memory(self, session: Any) -> None:
        """Not implemented — use save_memory_file() directly."""

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search memories by keyword matching on name and description."""
        headers = scan_memory_files(self._dir)
        query_lower = query.lower()
        matches: list[Memory] = []

        for h in headers:
            searchable = f"{h.name} {h.description}".lower()
            if query_lower in searchable:
                try:
                    text = h.filepath.read_text(encoding="utf-8")
                except OSError:
                    continue
                _, body = _parse_frontmatter(text)
                matches.append(
                    Memory(
                        content=body,
                        id=h.filename,
                        metadata={
                            "name": h.name,
                            "type": h.memory_type,
                            "description": h.description,
                        },
                    )
                )

        return SearchMemoryResponse(memories=matches)
