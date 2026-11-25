"""Retry and provider-fallback orchestration for structured extraction."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel

from llmkit.client import BaseLLMClient, LLMError
from llmkit.extract import ExtractionError, ExtractionResult, extract

T = TypeVar("T", bound=BaseModel)
Sleep = Callable[[float], Awaitable[None]]


class FallbackError(RuntimeError):
    """Both the primary and fallback providers failed."""

    def __init__(self, primary_error: Exception, fallback_error: Exception) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        super().__init__(
            f"Primary provider failed: {primary_error}. Fallback provider failed: {fallback_error}"
        )


def _is_retryable(error: Exception) -> bool:
    """Identify temporary provider or network failures."""
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(error, LLMError):
        match = re.search(r"HTTP (\d{3})", str(error))
        if match:
            status = int(match.group(1))
            return status == 429 or 500 <= status <= 599

    return False


async def extract_with_fallback(
    primary_client: BaseLLMClient,
    fallback_client: BaseLLMClient,
    document: str,
    model_cls: type[T],
    *,
    max_attempts: int = 2,
    base_delay_s: float = 0.5,
    sleep: Sleep = asyncio.sleep,
    max_tokens: int = 1500,
) -> ExtractionResult[T]:
    """Retry transient primary failures, then use the fallback provider."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if base_delay_s < 0:
        raise ValueError("base_delay_s cannot be negative")

    primary_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await extract(
                primary_client,
                document,
                model_cls,
                max_tokens=max_tokens,
            )
        except (ExtractionError, LLMError, httpx.TransportError) as error:
            primary_error = error

            if not _is_retryable(error) or attempt == max_attempts - 1:
                break

            delay = base_delay_s * (2**attempt)
            await sleep(delay)

    assert primary_error is not None

    try:
        return await extract(
            fallback_client,
            document,
            model_cls,
            max_tokens=max_tokens,
        )
    except (ExtractionError, LLMError, httpx.TransportError) as fallback_error:
        raise FallbackError(primary_error, fallback_error) from fallback_error
