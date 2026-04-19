"""Orx CLI theme — dark/light color palettes and Rich configuration."""

from __future__ import annotations

import os

from rich.console import Console
from rich.theme import Theme

# ── Brand palette ────────────────────────────────────────────
#
# Single source of truth for every hardcoded colour in the CLI.
# Anything outside this module that needs a colour must import one
# of these names — never inline a hex string.

BRAND_INK     = "#0F0E13"   # warm near-black base
BRAND_PAPER   = "#F5F2EB"   # warm cream (mark colour, light text on dark)
BRAND_SIGNAL  = "#3FE0A8"   # mint-teal — primary brand accent
BRAND_WHISPER = "#6B6872"   # muted grey (secondary text)
BRAND_LINE    = "#2A2732"   # subtle dividers
BRAND_AMBER   = "#F5C06B"   # warm amber (warnings, prompts)
BRAND_CORAL   = "#F5A0A0"   # soft coral (errors)
BRAND_BLUE    = "#8AB4F8"   # soft blue (info)

# Deeper variants for the light theme — signal is too pastel against
# white, so we need WCAG AA-compliant darker shades.
BRAND_SIGNAL_DEEP = "#0F8F66"   # deep mint for light-bg text
BRAND_AMBER_DEEP  = "#A66A00"   # darker amber for light bg
BRAND_CORAL_DEEP  = "#C4302B"   # crimson for light bg
BRAND_BLUE_DEEP   = "#1F5BD9"   # muted blue for light bg
BRAND_GREY_LIGHT  = "#737373"   # mid grey (light-theme muted)
BRAND_GREY_SUBTLE = "#D4D0C7"   # warm grey (light-theme subtle)

# ── Theme palettes ───────────────────────────────────────────

# Dark theme (default).
_DARK = {
    "accent":  BRAND_SIGNAL,
    "muted":   BRAND_WHISPER,
    "success": BRAND_SIGNAL,
    "warning": BRAND_AMBER,
    "error":   BRAND_CORAL,
    "info":    BRAND_BLUE,
    "subtle":  BRAND_LINE,
}

# Light theme — deeper shades for white-bg readability.
_LIGHT = {
    "accent":  BRAND_SIGNAL_DEEP,
    "muted":   BRAND_GREY_LIGHT,
    "success": BRAND_SIGNAL_DEEP,
    "warning": BRAND_AMBER_DEEP,
    "error":   BRAND_CORAL_DEEP,
    "info":    BRAND_BLUE_DEEP,
    "subtle":  BRAND_GREY_SUBTLE,
}

# Exported for use in stream.py spinner styling.
ACCENT: str = _DARK["accent"]


def _build_theme(palette: dict[str, str]) -> Theme:
    """Build a Rich Theme from a color palette.

    Parameters
    ----------
    palette : dict[str, str]
        Mapping of color names to hex values.

    Returns
    -------
    Theme
        Configured Rich theme.
    """
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
            "orx.tool.shell": f"bold {sc}",
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
RESPONSE_CONNECTOR: str = "\u25b8"  # ▸
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

    # Check persisted theme preference (~/.orx/theme).
    saved = _load_saved_theme()
    if saved in ("dark", "light"):
        return saved

    # $COLORFGBG is "fg;bg" — bg < 8 means dark, bg >= 8 means light.
    colorfgbg = os.environ.get("COLORFGBG", "")
    if ";" in colorfgbg:
        try:
            bg = int(colorfgbg.rsplit(";", 1)[1])
            return "light" if bg >= 8 else "dark"
        except ValueError:
            pass

    return "dark"


def _load_saved_theme() -> str:
    """Read the persisted theme from ``~/.orx/theme``.

    Returns
    -------
    str
        ``"dark"``, ``"light"``, or ``""`` if no saved preference.
    """
    from pathlib import Path

    theme_file = Path.home() / ".orx" / "theme"
    try:
        return theme_file.read_text().strip().lower()
    except (OSError, ValueError):
        return ""


def save_theme(theme: str) -> None:
    """Persist the theme preference to ``~/.orx/theme``.

    Parameters
    ----------
    theme : str
        ``"dark"`` or ``"light"``.
    """
    from pathlib import Path

    orx_dir = Path.home() / ".orx"
    orx_dir.mkdir(parents=True, exist_ok=True)
    (orx_dir / "theme").write_text(theme + "\n")


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
