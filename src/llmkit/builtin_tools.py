"""Safe example tools: an AST-based calculator and a directory-restricted file reader."""

from __future__ import annotations

import ast
import asyncio
import math
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llmkit.sandbox import ToolExecutionError
from llmkit.tools import Tool

MAX_EXPRESSION_CHARS = 200
MAX_ABS_RESULT = 10**100
MAX_EXPONENT = 100

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        min_length=1,
        max_length=MAX_EXPRESSION_CHARS,
        description="Arithmetic expression using numbers and + - * / // % ** and parentheses.",
    )


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value: Any = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolExecutionError(f"exponent must not exceed {MAX_EXPONENT}")
        value = _BINARY_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        value = _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    else:
        raise ToolExecutionError("unsupported expression; use numbers and + - * / // % ** only")

    if isinstance(value, complex):
        raise ToolExecutionError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolExecutionError("result is not a finite number")
    if abs(value) > MAX_ABS_RESULT:
        raise ToolExecutionError("result is too large")
    return value


async def _calculate(args: CalculatorArgs) -> str:
    try:
        tree = ast.parse(args.expression.strip(), mode="eval")
        return str(_evaluate(tree))
    except SyntaxError:
        raise ToolExecutionError("invalid expression syntax") from None
    except ZeroDivisionError:
        raise ToolExecutionError("division by zero") from None
    except (OverflowError, RecursionError):
        raise ToolExecutionError("expression is too large or too deeply nested") from None


CALCULATOR_TOOL = Tool(
    name="calculator",
    description="Evaluate an arithmetic expression exactly. Use it for any calculation.",
    args_model=CalculatorArgs,
    handler=_calculate,
)


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=300,
        description="File path relative to the allowed directory.",
    )


def make_read_file_tool(root: Path, *, max_bytes: int = 20_000) -> Tool:
    """Create a read-only file tool confined to ``root`` (symlinks and ``..`` cannot escape)."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    base = root.resolve()
    if not base.is_dir():
        raise ValueError("root must be an existing directory")

    def read(relative: str) -> str:
        try:
            target = (base / relative).resolve()
        except (OSError, ValueError):
            raise ToolExecutionError("invalid path") from None
        if not target.is_relative_to(base):
            raise ToolExecutionError("path is outside the allowed directory")
        if not target.exists():
            raise ToolExecutionError("file not found")
        if not target.is_file():
            raise ToolExecutionError("path is not a file")
        try:
            with target.open("rb") as handle:
                data = handle.read(max_bytes + 1)
        except OSError:
            raise ToolExecutionError("file could not be read") from None
        text = data[:max_bytes].decode("utf-8", errors="replace")
        if len(data) > max_bytes:
            text += "\n[file truncated]"
        return text

    async def handler(args: ReadFileArgs) -> str:
        return await asyncio.to_thread(read, args.path)

    return Tool(
        name="read_file",
        description="Read a UTF-8 text file from the allowed directory (read-only).",
        args_model=ReadFileArgs,
        handler=handler,
    )
