"""Log parsing: regex first, LLM only for lines the regex cannot handle."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from llmkit.client import BaseLLMClient
from llmkit.loaders import load_document
from llmkit.resilience import FallbackError, extract_with_fallback


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime | None
    level: LogLevel
    message: str = Field(min_length=1)


LEVEL_ALIASES = {
    "DEBUG": LogLevel.DEBUG,
    "INFO": LogLevel.INFO,
    "WARN": LogLevel.WARNING,
    "WARNING": LogLevel.WARNING,
    "ERR": LogLevel.ERROR,
    "ERROR": LogLevel.ERROR,
    "CRITICAL": LogLevel.CRITICAL,
    "FATAL": LogLevel.CRITICAL,
}
LOG_COLUMNS = ["line_no", "timestamp", "level", "message", "source"]

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)"
    r"\s+\[?(?P<level>[A-Za-z]+)\]?:?\s+(?P<msg>\S.*)$"
)


@dataclass
class ParsedLine:
    line_no: int
    event: LogEvent
    source: str


@dataclass
class LogParseResult:
    parsed: list[ParsedLine] = field(default_factory=list)
    unparsed: list[tuple[int, str]] = field(default_factory=list)
    recovered: int = 0


def _parse_line(line: str) -> LogEvent | None:
    match = LINE_RE.match(line)
    if not match:
        return None
    level = LEVEL_ALIASES.get(match.group("level").upper())
    if level is None:
        return None
    try:
        timestamp = datetime.fromisoformat(match.group("ts").replace(",", ".").replace(" ", "T"))
    except ValueError:
        return None
    return LogEvent(timestamp=timestamp, level=level, message=match.group("msg"))


def parse_log(text: str) -> LogParseResult:
    """Parse timestamped lines by regex; indented lines continue the previous event."""
    result = LogParseResult()
    last_was_event = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        event = _parse_line(line.strip())
        if event is not None:
            result.parsed.append(ParsedLine(line_no, event, "regex"))
            last_was_event = True
        elif line[0].isspace() and last_was_event:
            result.parsed[-1].event.message += "\n" + line.strip()
        else:
            result.unparsed.append((line_no, line.strip()))
            last_was_event = False
    return result


async def recover_unparsed(
    result: LogParseResult,
    primary: BaseLLMClient,
    fallback: BaseLLMClient,
    *,
    max_lines: int = 20,
    **options,
) -> LogParseResult:
    """Ask the LLM about unparsed lines, up to max_lines; failures stay unparsed."""
    remaining: list[tuple[int, str]] = []
    for position, (line_no, line) in enumerate(result.unparsed):
        if position >= max_lines:
            remaining.append((line_no, line))
            continue
        try:
            extracted = await extract_with_fallback(primary, fallback, line, LogEvent, **options)
        except FallbackError:
            remaining.append((line_no, line))
            continue
        result.parsed.append(ParsedLine(line_no, extracted.data, "llm"))
        result.recovered += 1
    result.unparsed = remaining
    result.parsed.sort(key=lambda item: item.line_no)
    return result


async def process_log_file(
    path: str | Path,
    primary: BaseLLMClient,
    fallback: BaseLLMClient,
    *,
    max_lines: int = 20,
    **options,
) -> LogParseResult:
    """Load a log file, parse it by regex, then let the LLM try the leftovers."""
    raw = await load_document(path)
    result = parse_log(raw.text)
    return await recover_unparsed(result, primary, fallback, max_lines=max_lines, **options)


def events_to_frame(result: LogParseResult) -> pd.DataFrame:
    rows = [
        {
            "line_no": item.line_no,
            "timestamp": item.event.timestamp.isoformat() if item.event.timestamp else None,
            "level": item.event.level.value,
            "message": item.event.message,
            "source": item.source,
        }
        for item in sorted(result.parsed, key=lambda item: item.line_no)
    ]
    return pd.DataFrame(rows, columns=LOG_COLUMNS)
