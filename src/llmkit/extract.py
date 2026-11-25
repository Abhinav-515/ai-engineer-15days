"""Schema-driven extraction: document text -> validated Pydantic object."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from llmkit.client import BaseLLMClient, LLMError, Messages

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = (
    "You extract structured data from documents. Reply with ONLY one JSON object that "
    "matches this JSON Schema. Dates must be ISO format (YYYY-MM-DD). Use null when a value "
    "is genuinely absent. Use 0 for tax if none is charged. Never invent values.\n\n"
    "JSON Schema:\n{schema}"
)


class ExtractionError(ValueError):
    """The model's reply could not be turned into a valid object."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class ExtractionResult(Generic[T]):  # noqa: UP046
    data: T
    raw_text: str
    used_native_schema: bool
    latency_s: float
    cost_usd: float
    provider: str = "unknown"


def clean_json_text(text: str) -> str:
    """Strip code fences and surrounding chatter; return the outermost {...} block."""
    t = re.sub(r"```[a-zA-Z]*", "", text).strip()
    start, end = t.find("{"), t.rfind("}")
    return t[start : end + 1] if start != -1 and end > start else t


async def extract(
    client: BaseLLMClient, document: str, model_cls: type[T], *, max_tokens: int = 1500
) -> ExtractionResult[T]:
    schema = model_cls.model_json_schema()

    messages: Messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(schema=json.dumps(schema)),
        },
        {
            "role": "user",
            "content": f"Document:\n\n{document}",
        },
    ]

    native = True

    try:
        result = await client.generate(
            messages,
            max_tokens=max_tokens,
            json_schema=schema,
        )
    except LLMError as e:
        if "HTTP 400" not in str(e):
            raise

        native = False
        result = await client.generate(
            messages,
            max_tokens=max_tokens,
        )

    try:
        data = model_cls.model_validate_json(clean_json_text(result.text))
    except ValidationError as e:
        raise ExtractionError(
            f"validation failed: {e}",
            raw_text=result.text,
        ) from e

    return ExtractionResult(
        data=data,
        raw_text=result.text,
        used_native_schema=native,
        latency_s=result.total_s,
        cost_usd=result.cost_usd,
        provider=getattr(client, "provider", "unknown"),
    )
