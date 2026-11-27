"""Tool definitions and registry with validated arguments for tool-calling agents."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

ToolHandler = Callable[[Any], Awaitable[Any]]


class ToolError(ValueError):
    """Base class for tool registry and validation errors."""


class UnknownToolError(ToolError):
    """The requested tool is not registered."""


class ToolArgumentError(ToolError):
    """Arguments failed validation against the tool's schema."""


@dataclass(frozen=True)
class Tool:
    """A callable capability exposed to the agent.

    ``handler`` receives an instance of ``args_model`` (already validated), never raw
    model-generated dictionaries.
    """

    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def json_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()


def _format_validation_error(exc: ValidationError) -> str:
    """Condense Pydantic errors to short ``field: message`` lines without echoing input."""
    lines = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        lines.append(f"{location}: {error['msg']}")
    return "; ".join(lines)


class ToolRegistry:
    """Holds the tools available to one agent and validates calls against them."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not TOOL_NAME_PATTERN.match(tool.name):
            raise ValueError(
                f"invalid tool name {tool.name!r}: use lowercase letters, digits and underscores"
            )
        if not tool.description.strip():
            raise ValueError(f"tool {tool.name!r} must have a description")
        if tool.args_model.model_config.get("extra") != "forbid":
            raise ValueError(f"args model for tool {tool.name!r} must set extra='forbid'")
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(self.names) or "none"
            raise UnknownToolError(f"unknown tool {name!r}; available tools: {available}") from None

    def parse_arguments(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        """Validate ``arguments`` for tool ``name``; raise a readable error on failure."""
        tool = self.get(name)
        try:
            return tool.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentError(
                f"invalid arguments for tool {name!r}: {_format_validation_error(exc)}"
            ) from exc

    def describe(self) -> str:
        """Render tool names, descriptions and argument schemas for an LLM prompt."""
        sections = []
        for name in self.names:
            tool = self._tools[name]
            schema = json.dumps(tool.json_schema())
            sections.append(f"- {name}: {tool.description.strip()}\n  arguments schema: {schema}")
        return "\n".join(sections)
