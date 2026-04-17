"""Unified content block parser for LangChain message formats.

Handles plain strings (Chat Completions), LangChain v1 standardized
content blocks (``message.content_blocks``), and raw provider formats
that may appear on ``message.content`` before translation:

* Anthropic raw: ``{"type": "thinking", "thinking": "..."}``
* OpenAI Responses raw: ``{"type": "reasoning", "summary": [...]}``
* Bedrock Converse raw: ``{"type": "reasoning_content", ...}``
* LangChain v1 standard: ``{"type": "reasoning", "reasoning": "..."}``

Prefer passing ``message.content_blocks`` when available — it pulls
reasoning from ``additional_kwargs`` (Groq, Ollama, DeepSeek, XAI)
into v1 reasoning blocks that this parser can see.
"""

from __future__ import annotations

from typing import Any


def parse_content_blocks(content: str | list[Any]) -> tuple[str, str]:
    """Extract text and thinking from any LangChain content format.

    Parameters
    ----------
    content : str or list
        The ``content`` field from an ``AIMessage`` or ``AIMessageChunk``.
        May be a plain string (Chat Completions) or a list of typed
        dicts (Anthropic extended thinking, OpenAI Response API).

    Returns
    -------
    tuple[str, str]
        ``(text, thinking)`` — concatenated text parts and concatenated
        thinking/reasoning parts.  Both default to ``""`` when absent.
    """
    if isinstance(content, str):
        return content, ""

    if not content:
        return "", ""

    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        btype = block.get("type", "")

        if btype == "text":
            text_parts.append(block.get("text", ""))

        elif btype == "thinking":
            # Anthropic extended thinking: {"type": "thinking", "thinking": "..."}
            thinking_parts.append(block.get("thinking", ""))

        elif btype == "reasoning":
            if "reasoning" in block:
                # LangChain v1 standard (all providers, post-translation)
                thinking_parts.append(block.get("reasoning", ""))
            else:
                # OpenAI Responses raw: nested summary list
                # {"type": "reasoning", "summary": [{"type": "summary_text", ...}]}
                for summary in block.get("summary", []):
                    if isinstance(summary, dict):
                        thinking_parts.append(summary.get("text", ""))

        elif btype == "reasoning_content":
            # Bedrock Converse raw:
            # {"type": "reasoning_content", "reasoning_content": {"text": "..."}}
            rc = block.get("reasoning_content", {})
            if isinstance(rc, dict):
                thinking_parts.append(rc.get("text", ""))

        # Skip non-text block types:
        # function_call, web_search_call, code_interpreter_call,
        # file_search_call, refusal, tool_call, image, audio, file, etc.

    return "".join(text_parts), "".join(thinking_parts)
