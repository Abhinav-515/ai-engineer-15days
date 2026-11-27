"""Tests for guarded tool execution."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, ConfigDict

from llmkit.sandbox import ToolExecutionError, execute_tool
from llmkit.tools import Tool, ToolRegistry


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


def registry_with(handler) -> ToolRegistry:
    return ToolRegistry([Tool("echo", "Echo text.", EchoArgs, handler)])


async def test_success_returns_string_output() -> None:
    async def handler(args: EchoArgs) -> str:
        return args.text.upper()

    outcome = await execute_tool(registry_with(handler), "echo", {"text": "hi"})

    assert outcome.ok
    assert outcome.output == "HI"
    assert outcome.error_kind is None
    assert not outcome.truncated
    assert outcome.duration_s >= 0


async def test_non_string_output_is_serialized_as_json() -> None:
    async def handler(args: EchoArgs) -> dict:
        return {"text": args.text, "n": 2}

    outcome = await execute_tool(registry_with(handler), "echo", {"text": "hi"})

    assert outcome.output == '{"text": "hi", "n": 2}'


async def test_unknown_tool_is_reported_not_raised() -> None:
    async def handler(args: EchoArgs) -> str:
        return "x"

    outcome = await execute_tool(registry_with(handler), "missing", {})

    assert not outcome.ok
    assert outcome.error_kind == "unknown_tool"
    assert "available tools: echo" in outcome.output


async def test_invalid_arguments_do_not_run_the_handler() -> None:
    calls = 0

    async def handler(args: EchoArgs) -> str:
        nonlocal calls
        calls += 1
        return "x"

    outcome = await execute_tool(registry_with(handler), "echo", {"text": 5, "extra": 1})

    assert not outcome.ok
    assert outcome.error_kind == "invalid_arguments"
    assert calls == 0


async def test_tool_execution_error_message_is_shown() -> None:
    async def handler(args: EchoArgs) -> str:
        raise ToolExecutionError("file not found")

    outcome = await execute_tool(registry_with(handler), "echo", {"text": "a"})

    assert not outcome.ok
    assert outcome.error_kind == "tool_error"
    assert outcome.output == "file not found"


async def test_unexpected_exception_hides_details() -> None:
    async def handler(args: EchoArgs) -> str:
        raise RuntimeError("secret path /etc/passwd")

    outcome = await execute_tool(registry_with(handler), "echo", {"text": "a"})

    assert not outcome.ok
    assert outcome.error_kind == "internal_error"
    assert outcome.output == "tool failed with RuntimeError"
    assert "secret" not in outcome.output


async def test_timeout_cancels_slow_handler() -> None:
    cancelled = False

    async def handler(args: EchoArgs) -> str:
        nonlocal cancelled
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return "late"

    outcome = await execute_tool(registry_with(handler), "echo", {"text": "a"}, timeout_s=0.05)

    assert not outcome.ok
    assert outcome.error_kind == "timeout"
    assert cancelled


async def test_long_output_is_truncated() -> None:
    async def handler(args: EchoArgs) -> str:
        return "x" * 100

    outcome = await execute_tool(registry_with(handler), "echo", {"text": "a"}, max_output_chars=10)

    assert outcome.ok
    assert outcome.truncated
    assert outcome.output.startswith("x" * 10)
    assert "x" * 11 not in outcome.output
    assert outcome.output.endswith("[output truncated]")


async def test_outer_cancellation_is_not_swallowed() -> None:
    async def handler(args: EchoArgs) -> str:
        await asyncio.sleep(5)
        return "late"

    task = asyncio.create_task(execute_tool(registry_with(handler), "echo", {"text": "a"}))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(("timeout_s", "max_chars"), [(0, 10), (-1, 10), (1, 0)])
async def test_rejects_non_positive_limits(timeout_s: float, max_chars: int) -> None:
    async def handler(args: EchoArgs) -> str:
        return "x"

    with pytest.raises(ValueError):
        await execute_tool(
            registry_with(handler),
            "echo",
            {"text": "a"},
            timeout_s=timeout_s,
            max_output_chars=max_chars,
        )
