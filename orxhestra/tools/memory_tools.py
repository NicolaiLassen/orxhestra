"""Memory tools — save, list, and delete persistent memories.

These tools let agents write and read memories that persist across
sessions. Memories are stored as individual markdown files with
YAML frontmatter in ``~/.orx/projects/<workspace>/memory/``.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from orxhestra.memory.file_memory_service import (
    MEMORY_TYPES,
    delete_memory_file,
    save_memory_file,
    scan_memory_files,
)


def make_memory_tools(memory_dir: Path) -> list[BaseTool]:
    """Create tools for reading and writing persistent memories.

    Parameters
    ----------
    memory_dir : Path
        Directory for storing memory files.

    Returns
    -------
    list[BaseTool]
        Three tools: ``save_memory``, ``list_memories``, ``delete_memory``.
    """

    async def save_memory(
        name: str,
        content: str,
        memory_type: str = "project",
        description: str = "",
    ) -> str:
        """Save a persistent memory that survives across sessions.

        Args:
            name: Short name for the memory (e.g. 'testing policy').
            content: The memory content (markdown). For feedback/project
                types, structure as: rule/fact, then **Why:** and
                **How to apply:** lines.
            memory_type: One of: user, feedback, project, reference.
                - user: user's role, preferences, knowledge
                - feedback: what to do/avoid (corrections and confirmations)
                - project: ongoing work, goals, deadlines
                - reference: pointers to external systems
            description: One-line description used for relevance matching
                in future sessions. Be specific.
        """
        if memory_type not in MEMORY_TYPES:
            return (
                f"Error: type must be one of {MEMORY_TYPES}, "
                f"got '{memory_type}'"
            )

        path = save_memory_file(
            memory_dir,
            name=name,
            content=content,
            memory_type=memory_type,
            description=description,
        )
        return f"Memory saved: {name} ({memory_type}) → {path.name}"

    async def list_memories() -> str:
        """List all saved memories with their type and description."""
        headers = scan_memory_files(memory_dir)
        if not headers:
            return "No memories saved yet."

        lines: list[str] = [f"{len(headers)} memories:"]
        for h in headers:
            type_tag = f"[{h.memory_type}]" if h.memory_type else "[?]"
            desc = f" — {h.description}" if h.description else ""
            lines.append(f"  {type_tag} {h.name}{desc}")
        return "\n".join(lines)

    async def delete_memory(name: str) -> str:
        """Delete a saved memory by name.

        Args:
            name: The name of the memory to delete.
        """
        deleted = delete_memory_file(memory_dir, name)
        if deleted:
            return f"Deleted memory: {name}"
        return f"Error: memory '{name}' not found."

    return [
        StructuredTool.from_function(
            coroutine=save_memory,
            name="save_memory",
            description=(
                "Save a persistent memory that survives across sessions. "
                "Use for user preferences, feedback, project context, "
                "or references to external systems. "
                "Types: user, feedback, project, reference."
            ),
        ),
        StructuredTool.from_function(
            coroutine=list_memories,
            name="list_memories",
            description="List all saved persistent memories.",
        ),
        StructuredTool.from_function(
            coroutine=delete_memory,
            name="delete_memory",
            description="Delete a saved memory by name.",
        ),
    ]
