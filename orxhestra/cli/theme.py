"""Orx CLI theme — centralised colors, styles, and Rich configuration."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

ACCENT: str = "#6C8EBF"
MUTED: str = "#6c6c6c"
SUCCESS: str = "#98C379"
WARNING: str = "#E5C07B"
ERROR: str = "#E06C75"
INFO: str = "#61AFEF"
SUBTLE: str = "#4a4a4a"

ORX_THEME = Theme(
    {
        "orx.accent": f"bold {ACCENT}",
        "orx.muted": MUTED,
        "orx.subtle": SUBTLE,
        "orx.label": MUTED,
        "orx.tool.read": MUTED,
        "orx.tool.write": WARNING,
        "orx.tool.shell": "bold",
        "orx.tool.default": MUTED,
        "orx.success": SUCCESS,
        "orx.warning": WARNING,
        "orx.error": f"bold {ERROR}",
        "orx.info": INFO,
        "orx.summary": MUTED,
        "orx.agent": f"bold {ACCENT}",
        "orx.prompt": f"bold {ACCENT}",
        "orx.prompt.input": "bold",
        "orx.help.cmd": f"bold {ACCENT}",
        "orx.help.desc": MUTED,
        "orx.banner.version": MUTED,
        "orx.banner.label": MUTED,
        "orx.approval": f"bold {WARNING}",
        "orx.approval.cmd": "bold",
        "orx.denied": MUTED,
        "orx.interrupted": MUTED,
        "orx.goodbye": MUTED,
        "orx.status": MUTED,
    }
)

TOOL_TOP: str = "\u250c"
TOOL_MID: str = "\u2502"
TOOL_BOT: str = "\u2514"
TURN_DOT: str = "\u27e1"
SEP: str = " \u00b7 "


def make_console() -> Console:
    """Create a themed Rich Console for the orx CLI."""
    return Console(theme=ORX_THEME)
