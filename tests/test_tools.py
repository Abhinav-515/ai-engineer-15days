"""Tests for tool definitions, registry and argument validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from llmkit.tools import (
    Tool,
    ToolArgumentError,
    ToolRegistry,
    UnknownToolError,
)


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: int
    b: int = Field(ge=0)


class LooseArgs(BaseModel):
    value: str


async def add_handler(args: AddArgs) -> int:
    return args.a + args.b


def make_tool(name: str = "add", args_model: type[BaseModel] = AddArgs) -> Tool:
    return Tool(name, "Add two numbers.", args_model, add_handler)


def test_register_and_lookup() -> None:
    registry = ToolRegistry([make_tool()])

    assert "add" in registry
    assert registry.names == ["add"]
    assert len(registry) == 1
    assert registry.get("add").name == "add"


def test_names_are_sorted() -> None:
    registry = ToolRegistry([make_tool("zeta"), make_tool("alpha")])

    assert registry.names == ["alpha", "zeta"]


def test_duplicate_names_rejected() -> None:
    registry = ToolRegistry([make_tool()])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_tool())


@pytest.mark.parametrize("name", ["", "Add", "1add", "add-tool", "add tool", "a" * 41])
def test_invalid_names_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="invalid tool name"):
        ToolRegistry([make_tool(name)])


def test_empty_description_rejected() -> None:
    tool = Tool("add", "   ", AddArgs, add_handler)

    with pytest.raises(ValueError, match="description"):
        ToolRegistry([tool])


def test_args_model_must_forbid_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra='forbid'"):
        ToolRegistry([make_tool("loose", LooseArgs)])


def test_unknown_tool_lists_available_tools() -> None:
    registry = ToolRegistry([make_tool()])

    with pytest.raises(UnknownToolError, match="available tools: add"):
        registry.get("subtract")


def test_unknown_tool_with_empty_registry() -> None:
    with pytest.raises(UnknownToolError, match="available tools: none"):
        ToolRegistry().get("anything")


def test_parse_valid_arguments_returns_model_instance() -> None:
    registry = ToolRegistry([make_tool()])

    parsed = registry.parse_arguments("add", {"a": 1, "b": 2})

    assert isinstance(parsed, AddArgs)
    assert (parsed.a, parsed.b) == (1, 2)


def test_parse_rejects_missing_wrong_type_and_constraint_errors() -> None:
    registry = ToolRegistry([make_tool()])

    with pytest.raises(ToolArgumentError, match="a: ") as missing:
        registry.parse_arguments("add", {"b": 1})
    assert "invalid arguments for tool 'add'" in str(missing.value)

    with pytest.raises(ToolArgumentError, match="a: "):
        registry.parse_arguments("add", {"a": "not a number", "b": 1})

    with pytest.raises(ToolArgumentError, match="b: "):
        registry.parse_arguments("add", {"a": 1, "b": -5})


def test_parse_rejects_extra_arguments() -> None:
    registry = ToolRegistry([make_tool()])

    with pytest.raises(ToolArgumentError, match="c: Extra inputs are not permitted"):
        registry.parse_arguments("add", {"a": 1, "b": 2, "c": 3})


def test_validation_message_does_not_echo_input_values() -> None:
    registry = ToolRegistry([make_tool()])

    with pytest.raises(ToolArgumentError) as exc_info:
        registry.parse_arguments("add", {"a": "SECRET-VALUE", "b": 1})

    assert "SECRET-VALUE" not in str(exc_info.value)


def test_parse_unknown_tool_raises_unknown_tool_error() -> None:
    with pytest.raises(UnknownToolError):
        ToolRegistry().parse_arguments("add", {})


def test_describe_lists_tools_and_schemas() -> None:
    registry = ToolRegistry([make_tool("zeta"), make_tool("alpha")])

    text = registry.describe()

    assert text.index("- alpha:") < text.index("- zeta:")
    assert "Add two numbers." in text
    assert '"properties"' in text
    assert "arguments schema" in text


def test_empty_registry_describes_nothing() -> None:
    assert ToolRegistry().describe() == ""


async def test_handler_receives_validated_model() -> None:
    registry = ToolRegistry([make_tool()])
    tool = registry.get("add")

    parsed = registry.parse_arguments("add", {"a": 2, "b": 3})

    assert await tool.handler(parsed) == 5
