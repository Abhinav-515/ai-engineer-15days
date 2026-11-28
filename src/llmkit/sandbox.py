"""Guarded tool execution: validation, timeout, output cap and error containment.

This is an in-process guardrail, not an OS-level sandbox. A timeout can only interrupt
a handler at an ``await`` point; a CPU-bound handler that never yields cannot be stopped.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from llmkit.tools import ToolArgumentError, ToolRegistry, UnknownToolError

TRUNCATION_NOTICE = "\n[output truncated]"


class ToolExecutionError(Exception):
    """A tool failure whose message is safe to show to the model."""


@dataclass(frozen=True)
class ToolOutcome:
    """Result of one guarded tool call; ``output`` holds the result or the error message."""

    tool: str
    ok: bool
    output: str
    duration_s: float
    truncated: bool = False
    error_kind: str | None = (
        None  # unknown_tool, invalid_arguments, timeout, tool_error, internal_error
    )


def _serialize(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


async def execute_tool(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, Any],
    *,
    timeout_s: float = 10.0,
    max_output_chars: int = 4_000,
) -> ToolOutcome:
    """Validate and run one tool call. Tool problems become outcomes, never exceptions."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_output_chars <= 0:
        raise ValueError("max_output_chars must be positive")

    start = time.perf_counter()

    def finish(ok: bool, text: str, kind: str | None = None) -> ToolOutcome:
        truncated = len(text) > max_output_chars
        if truncated:
            text = text[:max_output_chars] + TRUNCATION_NOTICE
        return ToolOutcome(name, ok, text, time.perf_counter() - start, truncated, kind)

    try:
        tool = registry.get(name)
        parsed = registry.parse_arguments(name, arguments)
    except UnknownToolError as exc:
        return finish(False, str(exc), "unknown_tool")
    except ToolArgumentError as exc:
        return finish(False, str(exc), "invalid_arguments")

    try:
        value = await asyncio.wait_for(tool.handler(parsed), timeout=timeout_s)
    except TimeoutError:
        return finish(False, f"tool {name!r} timed out after {timeout_s:g}s", "timeout")
    except ToolExecutionError as exc:
        return finish(False, str(exc), "tool_error")
    except Exception as exc:
        return finish(False, f"tool failed with {type(exc).__name__}", "internal_error")

    return finish(True, _serialize(value))
