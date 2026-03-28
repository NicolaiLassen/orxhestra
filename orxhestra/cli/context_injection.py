"""Local context injection - detect project environment and inject into prompt.

Detects language, package manager, git state, test commands, project tree,
and injects all context into the system prompt.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


async def _run_cmd(cmd: str, cwd: str, timeout: float = 5.0) -> str:
    """Run a shell command and return stdout, or empty string on failure."""
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(errors="replace").strip()
    except (asyncio.TimeoutError, OSError):
        return ""


def _detect_languages(workspace: str) -> list[str]:
    """Detect programming languages from file extensions and config files."""
    ws = Path(workspace)
    langs: list[str] = []
    markers: dict[str, list[str]] = {
        "Python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
        "JavaScript/TypeScript": ["package.json", "tsconfig.json"],
        "Rust": ["Cargo.toml"],
        "Go": ["go.mod"],
        "Java": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "C#": ["*.csproj", "*.sln"],
        "Ruby": ["Gemfile"],
        "PHP": ["composer.json"],
        "Swift": ["Package.swift"],
        "Elixir": ["mix.exs"],
    }
    for lang, files in markers.items():
        for pattern in files:
            if "*" in pattern:
                if list(ws.glob(pattern)):
                    langs.append(lang)
                    break
            elif (ws / pattern).exists():
                langs.append(lang)
                break
    return langs


def _detect_package_manager(workspace: str) -> str | None:
    """Detect the package manager from lock files and configs."""
    ws = Path(workspace)
    managers: list[tuple[str, str]] = [
        ("uv.lock", "uv"),
        ("poetry.lock", "poetry"),
        ("Pipfile.lock", "pipenv"),
        ("bun.lockb", "bun"),
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
        ("package-lock.json", "npm"),
        ("Cargo.lock", "cargo"),
        ("go.sum", "go modules"),
    ]
    for lockfile, manager in managers:
        if (ws / lockfile).exists():
            return manager
    if (ws / "pyproject.toml").exists():
        return "pip/uv"
    if (ws / "package.json").exists():
        return "npm"
    return None


def _detect_test_command(workspace: str) -> str | None:
    """Detect the likely test command for the project."""
    ws = Path(workspace)
    if (ws / "Makefile").exists():
        try:
            content: str = (ws / "Makefile").read_text(errors="replace")
            if "test:" in content:
                return "make test"
        except OSError:
            pass
    if (ws / "pyproject.toml").exists():
        try:
            content = (ws / "pyproject.toml").read_text(errors="replace")
            if "pytest" in content:
                return "pytest"
        except OSError:
            pass
    if (ws / "package.json").exists():
        try:
            content = (ws / "package.json").read_text(errors="replace")
            if '"test"' in content:
                return "npm test"
        except OSError:
            pass
    if (ws / "Cargo.toml").exists():
        return "cargo test"
    if (ws / "go.mod").exists():
        return "go test ./..."
    return None


def _get_directory_listing(workspace: str, max_items: int = 25) -> list[str]:
    """Get top-level directory listing, excluding common noise."""
    ws = Path(workspace)
    ignore: set[str] = {
        "__pycache__", ".git", "node_modules", ".venv", "venv", ".tox",
        ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
        ".egg-info", ".eggs", "target", ".next", ".nuxt", ".cache",
        "coverage", ".coverage", ".DS_Store",
    }
    entries: list[str] = []
    try:
        for item in sorted(ws.iterdir()):
            if item.name in ignore or item.name.endswith(".egg-info"):
                continue
            suffix: str = "/" if item.is_dir() else ""
            entries.append(f"{item.name}{suffix}")
            if len(entries) >= max_items:
                break
    except OSError:
        pass
    return entries


async def collect_local_context(workspace: str) -> str:
    """Collect local project context and return formatted string for prompt injection."""
    sections: list[str] = []

    # Languages
    langs: list[str] = _detect_languages(workspace)
    if langs:
        sections.append(f"Languages: {', '.join(langs)}")

    # Package manager
    pkg_mgr: str | None = _detect_package_manager(workspace)
    if pkg_mgr:
        sections.append(f"Package manager: {pkg_mgr}")

    # Test command
    test_cmd: str | None = _detect_test_command(workspace)
    if test_cmd:
        sections.append(f"Test command: `{test_cmd}`")

    # Git info
    git_branch: str = await _run_cmd("git branch --show-current", workspace)
    if git_branch:
        sections.append(f"Git branch: {git_branch}")

        git_status: str = await _run_cmd(
            "git status --porcelain | wc -l | tr -d ' '", workspace
        )
        if git_status and git_status != "0":
            sections.append(f"Uncommitted changes: {git_status} files")

        main_branch: str = await _run_cmd(
            "git branch --list main master 2>/dev/null | head -1 | tr -d '* '",
            workspace,
        )
        if main_branch:
            sections.append(f"Main branch: {main_branch}")

    # Runtime versions
    python_ver: str = await _run_cmd("python3 --version 2>/dev/null | cut -d' ' -f2", workspace)
    if python_ver:
        sections.append(f"Python: {python_ver}")

    node_ver: str = await _run_cmd("node --version 2>/dev/null", workspace)
    if node_ver:
        sections.append(f"Node.js: {node_ver}")

    # Directory listing
    entries: list[str] = _get_directory_listing(workspace)
    if entries:
        listing: str = ", ".join(entries)
        sections.append(f"Files: {listing}")

    # Project tree (if tree command available, max 3 levels)
    tree_output: str = await _run_cmd(
        "tree -L 2 --dirsfirst -I '__pycache__|node_modules|.git|.venv|venv|dist|build|.tox|.mypy_cache|.pytest_cache|target|.next|coverage' 2>/dev/null | head -40",
        workspace,
    )
    if tree_output:
        sections.append(f"Project tree:\n```\n{tree_output}\n```")

    if not sections:
        return ""

    return "# Project Context\n" + "\n".join(sections)
