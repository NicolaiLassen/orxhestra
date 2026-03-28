"""CLI configuration constants."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME: str = "orx-cli"
DEFAULT_USER_ID: str = "cli-user"
DEFAULT_MODEL: str = os.environ.get("ORX_MODEL", "gpt-5.4")
HISTORY_DIR: Path = Path.home() / ".orx"
HISTORY_FILE: Path = HISTORY_DIR / "history"

# Provider detection: prefix -> (provider name for models registry)
PROVIDER_PREFIXES: list[tuple[str, str]] = [
    ("gpt-", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
    ("o5-", "openai"),
    ("chatgpt-", "openai"),
    ("claude-", "anthropic"),
    ("gemini-", "google"),
]

# Env var hints per provider
PROVIDER_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}

PROVIDER_INSTALL_HINTS: dict[str, str] = {
    "openai": "pip install orxhestra[openai]",
    "anthropic": "pip install orxhestra[anthropic]",
    "google": "pip install orxhestra[google]",
}
