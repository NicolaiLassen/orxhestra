"""File-based memory system — save, scan, and delete persistent memories.

Demonstrates the FileMemoryService and low-level memory helpers that
store memories as individual markdown files with YAML frontmatter.

Run::

    python examples/memory_system.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from orxhestra.memory.file_memory_service import (
    FileMemoryService,
    delete_memory_file,
    save_memory_file,
    scan_memory_files,
)


async def main() -> None:
    """Demonstrate the file-based memory lifecycle."""
    # Use a temp directory so the example is self-contained
    memory_dir: Path = Path(tempfile.mkdtemp()) / "memory"

    # --- Save memories ---
    path1: Path = save_memory_file(
        memory_dir,
        name="testing policy",
        content="Always write unit tests before merging.",
        memory_type="feedback",
        description="Team rule on testing",
    )
    print(f"Saved: {path1.name}")

    path2: Path = save_memory_file(
        memory_dir,
        name="Alice's role",
        content="Alice is a senior backend engineer on the payments team.",
        memory_type="user",
        description="User role and team",
    )
    print(f"Saved: {path2.name}")

    # --- Scan all memories ---
    headers = scan_memory_files(memory_dir)
    print(f"\n{len(headers)} memories on disk:")
    for h in headers:
        print(f"  [{h.memory_type}] {h.name} — {h.description}")

    # --- Search via FileMemoryService ---
    service = FileMemoryService(memory_dir=memory_dir)
    result = await service.search_memory(
        app_name="demo", user_id="user-1", query="alice",
    )
    print(f"\nSearch 'alice': {len(result.memories)} match(es)")
    for m in result.memories:
        print(f"  {m.content}")

    # --- Delete a memory ---
    deleted: bool = delete_memory_file(memory_dir, "testing policy")
    print(f"\nDeleted 'testing policy': {deleted}")
    print(f"Remaining: {len(scan_memory_files(memory_dir))} memories")


if __name__ == "__main__":
    asyncio.run(main())
