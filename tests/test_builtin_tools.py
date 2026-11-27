"""Tests for the calculator and file-reader tools, run through the guarded executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmkit.builtin_tools import CALCULATOR_TOOL, make_read_file_tool
from llmkit.sandbox import execute_tool
from llmkit.tools import ToolRegistry


async def calc(expression: str):
    return await execute_tool(
        ToolRegistry([CALCULATOR_TOOL]), "calculator", {"expression": expression}
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+3*4", "14"),
        ("(1+2)**3", "27"),
        ("-5 + 2", "-3"),
        ("7//2", "3"),
        ("7%4", "3"),
        ("1/4", "0.25"),
        ("  2 * 21  ", "42"),
    ],
)
async def test_calculator_valid_expressions(expression: str, expected: str) -> None:
    outcome = await calc(expression)

    assert outcome.ok
    assert outcome.output == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "abs(-1)",
        "True + 1",
        "'a' * 5",
        "x + 1",
        "[1, 2]",
        "2 ** 101",
        "10**100 * 10",
        "1 / 0",
        "2 +",
        "(-8) ** 0.5",
    ],
)
async def test_calculator_rejects_unsafe_or_invalid_expressions(expression: str) -> None:
    outcome = await calc(expression)

    assert not outcome.ok
    assert outcome.error_kind == "tool_error"


async def test_calculator_rejects_overlong_expression_at_validation() -> None:
    outcome = await calc("1+" * 150 + "1")

    assert outcome.error_kind == "invalid_arguments"


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "notes.txt").write_text("hello")
    (root / "sub" / "a.txt").write_text("nested")
    outside = tmp_path / "outside.txt"
    outside.write_text("top secret")
    return root, outside


async def read(root: Path, path: str, **kwargs):
    registry = ToolRegistry([make_read_file_tool(root, **kwargs)])
    return await execute_tool(registry, "read_file", {"path": path})


async def test_read_file_inside_root(files: tuple[Path, Path]) -> None:
    root, _ = files

    assert (await read(root, "notes.txt")).output == "hello"
    assert (await read(root, "sub/a.txt")).output == "nested"


async def test_read_file_blocks_parent_traversal(files: tuple[Path, Path]) -> None:
    root, _ = files

    outcome = await read(root, "../outside.txt")

    assert not outcome.ok
    assert outcome.output == "path is outside the allowed directory"
    assert "top secret" not in outcome.output


async def test_read_file_blocks_absolute_paths(files: tuple[Path, Path]) -> None:
    root, outside = files

    outcome = await read(root, str(outside))

    assert not outcome.ok
    assert outcome.output == "path is outside the allowed directory"


async def test_read_file_blocks_symlink_escape(files: tuple[Path, Path]) -> None:
    root, outside = files
    (root / "link.txt").symlink_to(outside)

    outcome = await read(root, "link.txt")

    assert not outcome.ok
    assert outcome.output == "path is outside the allowed directory"


async def test_read_file_missing_directory_and_bad_path(files: tuple[Path, Path]) -> None:
    root, _ = files

    assert (await read(root, "nope.txt")).output == "file not found"
    assert (await read(root, "sub")).output == "path is not a file"
    assert (await read(root, "a\x00b")).output == "invalid path"


async def test_read_file_errors_do_not_leak_absolute_paths(files: tuple[Path, Path]) -> None:
    root, _ = files

    for path in ["nope.txt", "../outside.txt", "sub"]:
        assert str(root) not in (await read(root, path)).output


async def test_read_file_truncates_large_files(files: tuple[Path, Path]) -> None:
    root, _ = files
    (root / "big.txt").write_text("a" * 100)

    outcome = await read(root, "big.txt", max_bytes=10)

    assert outcome.ok
    assert outcome.output.startswith("a" * 10)
    assert "a" * 11 not in outcome.output
    assert outcome.output.endswith("[file truncated]")


async def test_read_file_replaces_invalid_utf8(files: tuple[Path, Path]) -> None:
    root, _ = files
    (root / "bin.dat").write_bytes(b"ok\xff\xfe")

    outcome = await read(root, "bin.dat")

    assert outcome.ok
    assert outcome.output.startswith("ok")


def test_read_file_factory_validates_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        make_read_file_tool(tmp_path / "missing")
    with pytest.raises(ValueError, match="max_bytes"):
        make_read_file_tool(tmp_path, max_bytes=0)
