"""Tool output utilities — truncation, formatting, and size management.

Provides reusable functions for managing tool output size to prevent
context window bloat.  These are framework-level utilities that any
tool or agent can use.

Example::

    from orxhestra.tools.output import truncate_output

    result = truncate_output(long_text, max_chars=2000)
"""

from __future__ import annotations


def truncate_output(
    text: str,
    max_chars: int,
    *,
    suffix: str | None = None,
) -> str:
    """Truncate text to a maximum character count.

    If the text exceeds ``max_chars``, it is cut at that boundary and
    a suffix showing the truncation count is appended.  Tries to cut
    at the last newline before the limit to avoid splitting mid-sentence.

    Parameters
    ----------
    text : str
        The text to potentially truncate.
    max_chars : int
        Maximum number of characters to keep.
    suffix : str, optional
        Custom suffix.  If ``None`` (default), an auto-generated suffix
        showing the number of truncated characters is used.

    Returns
    -------
    str
        The original text if within limit, or the truncated text
        with suffix appended.
    """
    if len(text) <= max_chars:
        return text

    original_len = len(text)

    # Try to cut at a newline boundary for cleaner output
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.5:
        cut = cut[:last_nl]

    truncated_chars = original_len - len(cut)

    if suffix is None:
        suffix = f"\n\n[... truncated {truncated_chars:,} chars]"

    return cut + suffix
