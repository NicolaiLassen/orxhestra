"""Orx CLI theme — dark/light color palettes and Rich configuration."""

from __future__ import annotations

import os

from rich.console import Console
from rich.theme import Theme

# ── Color palettes ───────────────────────────────────────────

# Dark theme (default)
_DARK = {
    "accent": "#6C8EBF",
    "muted": "#6c6c6c",
    "success": "#98C379",
    "warning": "#E5C07B",
    "error": "#E06C75",
    "info": "#61AFEF",
    "subtle": "#4a4a4a",
}

# Light theme
_LIGHT = {
    "accent": "#2563EB",
    "muted": "#737373",
    "success": "#16A34A",
    "warning": "#CA8A04",
    "error": "#DC2626",
    "info": "#2563EB",
    "subtle": "#a3a3a3",
}

# Exported for use in stream.py spinner styling.
ACCENT: str = _DARK["accent"]


def _build_theme(palette: dict[str, str]) -> Theme:
    """Build a Rich Theme from a color palette."""
    a = palette["accent"]
    m = palette["muted"]
    s = palette["subtle"]
    w = palette["warning"]
    e = palette["error"]
    i = palette["info"]
    sc = palette["success"]
    return Theme(
        {
            "orx.accent": f"bold {a}",
            "orx.muted": m,
            "orx.subtle": s,
            "orx.label": m,
            "orx.tool.read": m,
            "orx.tool.write": w,
            "orx.tool.shell": "bold",
            "orx.tool.default": m,
            "orx.success": sc,
            "orx.warning": w,
            "orx.error": f"bold {e}",
            "orx.info": i,
            "orx.summary": m,
            "orx.agent": f"bold {a}",
            "orx.prompt": f"bold {a}",
            "orx.prompt.input": "bold",
            "orx.help.cmd": f"bold {a}",
            "orx.help.desc": m,
            "orx.banner.version": m,
            "orx.banner.label": m,
            "orx.approval": f"bold {w}",
            "orx.approval.cmd": "bold",
            "orx.denied": m,
            "orx.interrupted": m,
            "orx.goodbye": m,
            "orx.thinking": f"dim italic {m}",
            "orx.status": m,
            # Rich Markdown overrides
            "markdown.item.bullet": f"bold {a}",
            "markdown.block_quote": a,
            "markdown.h1": f"bold {a}",
            "markdown.h2": f"bold {a}",
            "markdown.h3": f"bold {a}",
            "markdown.h4": f"bold {a}",
            "markdown.link": f"underline {a}",
            "markdown.link_url": a,
        }
    )


DARK_THEME: Theme = _build_theme(_DARK)
LIGHT_THEME: Theme = _build_theme(_LIGHT)

TOOL_TOP: str = "\u250c"
TOOL_MID: str = "\u2502"
TOOL_BOT: str = "\u2514"
TURN_DOT: str = "\u27e1"
SEP: str = " \u00b7 "
RESPONSE_CONNECTOR: str = "\u23bf"  # ⎿
BLOCKQUOTE_BAR: str = "\u258e"  # ▎

# Custom spinner for the orxhestra CLI.
ORX_SPINNER: dict = {
    "interval": 200,
    "frames": ["\u2669", "\u266a", "\u266b", "\u266c", "\u266b", "\u266a"],
    # ♩ ♪ ♫ ♬ ♫ ♪
}


def detect_terminal_theme() -> str:
    """Detect terminal dark/light mode.

    Checks ``$ORX_THEME`` (explicit override), then ``$COLORFGBG``
    (set by many terminals). Falls back to ``"dark"``.

    Returns
    -------
    str
        ``"dark"`` or ``"light"``.
    """
    explicit = os.environ.get("ORX_THEME", "").lower()
    if explicit in ("dark", "light"):
        return explicit

    # $COLORFGBG is "fg;bg" — bg < 8 means dark, bg >= 8 means light.
    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            bg = int(colorfgbg.rsplit(";", 1)[1])
            return "light" if bg >= 8 else "dark"
        except ValueError:
            pass

    return "dark"


def make_console(theme: str = "auto") -> Console:
    """Create a themed Rich Console for the orx CLI.

    Parameters
    ----------
    theme : str
        ``"dark"``, ``"light"``, or ``"auto"`` (detect from terminal).

    Returns
    -------
    Console
        A Rich Console with the selected theme applied.
    """
    if theme == "auto":
        theme = detect_terminal_theme()

    global ACCENT  # noqa: PLW0603
    palette = _LIGHT if theme == "light" else _DARK
    ACCENT = palette["accent"]

    rich_theme = LIGHT_THEME if theme == "light" else DARK_THEME
    return Console(theme=rich_theme)
