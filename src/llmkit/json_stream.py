"""Incremental extraction of the ``answer`` string from a streamed JSON object."""

from __future__ import annotations

import re

_REPLACEMENT = "\ufffd"


class AnswerFieldExtractor:
    """Decode the value of the top-level ``"answer"`` field as chunks arrive.

    Feed raw JSON text chunks to :meth:`feed`; it returns only the newly decoded
    characters of the answer string. Chunk boundaries may fall anywhere, including
    inside an escape sequence. Output is always valid Unicode.
    """

    _START = re.compile(r'"answer"\s*:\s*"')
    _SIMPLE = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }

    def __init__(self) -> None:
        self._prefix = ""
        self._in_string = False
        self._done = False
        self._pending = ""

    @property
    def done(self) -> bool:
        """True once the closing quote of the answer string has been seen."""
        return self._done

    def feed(self, chunk: str) -> str:
        """Consume a chunk and return newly decoded answer text (possibly empty)."""
        if self._done:
            return ""

        if not self._in_string:
            self._prefix += chunk
            match = self._START.search(self._prefix)
            if match is None:
                return ""
            self._in_string = True
            chunk = self._prefix[match.end() :]
            self._prefix = ""

        data = self._pending + chunk
        self._pending = ""
        out: list[str] = []
        i = 0

        while i < len(data):
            ch = data[i]
            if ch == '"':
                self._done = True
                break
            if ch != "\\":
                out.append(ch)
                i += 1
                continue

            decoded = self._read_escape(data, i)
            if decoded is None:
                self._pending = data[i:]
                break
            text, consumed = decoded
            out.append(text)
            i += consumed

        return "".join(out)

    @classmethod
    def _read_escape(cls, data: str, i: int) -> tuple[str, int] | None:
        """Decode the escape at ``data[i]``; ``None`` means more input is needed."""
        if i + 1 >= len(data):
            return None
        esc = data[i + 1]
        if esc != "u":
            return cls._SIMPLE.get(esc, esc), 2
        if i + 6 > len(data):
            return None

        try:
            code = int(data[i + 2 : i + 6], 16)
        except ValueError:
            return _REPLACEMENT, 6

        if 0xD800 <= code <= 0xDBFF:
            tail = data[i + 6 : i + 12]
            if len(tail) < 6 and "\\u".startswith(tail[:2]):
                return None  # a low surrogate may still arrive
            if tail.startswith("\\u"):
                try:
                    low = int(tail[2:6], 16)
                except ValueError:
                    low = -1
                if 0xDC00 <= low <= 0xDFFF:
                    return chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)), 12
            return _REPLACEMENT, 6

        if 0xDC00 <= code <= 0xDFFF:
            return _REPLACEMENT, 6
        return chr(code), 6
