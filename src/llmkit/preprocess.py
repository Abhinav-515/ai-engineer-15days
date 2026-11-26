"""Text cleanup and length bounding before extraction."""

from __future__ import annotations

import re

TRUNCATION_MARKER = "\n[... truncated ...]\n"


def clean_text(text: str) -> str:
    """Normalize line endings, drop NUL bytes and trailing spaces, collapse blank runs."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def truncate_text(text: str, max_chars: int = 6000) -> tuple[str, bool]:
    """Keep the head and tail of long text; invoice totals usually sit at the end.

    The result is max_chars plus the length of the truncation marker.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if len(text) <= max_chars:
        return text, False
    head = (max_chars * 2) // 3
    tail = max_chars - head
    return text[:head] + TRUNCATION_MARKER + text[-tail:], True
