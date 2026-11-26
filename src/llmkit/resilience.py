"""Retry and provider-fallback orchestration for structured extraction."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TypeVar

import httpx
from pydantic import BaseModel

from llmkit.attempts import AttemptRecord
from llmkit.client import BaseLLMClient, LLMError, http_status
from llmkit.extract import ExtractionError, ExtractionResult, extract

T = TypeVar("T", bound=BaseModel)
Sleep = Callable[[float], Awaitable[None]]
Rng = Callable[[], float]


class FallbackError(RuntimeError):
    """Both the primary and fallback providers failed."""

    def __init__(
        self,
        primary_error: Exception,
        fallback_error: Exception,
        *,
        attempts: list[AttemptRecord] | None = None,
    ) -> None:
        self.primary_error = primary_error
        self.fallback_error = fallback_error
        self.attempts = list(attempts) if attempts else []
        super().__init__(
            f"Primary provider failed: {primary_error}. Fallback provider failed: {fallback_error}"
        )


def _is_retryable(error: Exception) -> bool:
    """Identify temporary provider or network failures."""
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(error, LLMError):
        status = http_status(error)
        if status is not None:
            return status == 429 or 500 <= status <= 599

    return False


def compute_delay(
    attempt: int,
    *,
    base_delay_s: float,
    max_delay_s: float = 30.0,
    jitter_ratio: float = 0.0,
    retry_after_s: float | None = None,
    rng: Rng = random.random,
) -> float:
    """Exponential backoff, raised to Retry-After if given, jittered, then capped."""
    delay = base_delay_s * (2**attempt)
    if retry_after_s is not None:
        delay = max(delay, retry_after_s)
    delay = min(delay, max_delay_s)
    if jitter_ratio > 0:
        delay *= 1 + jitter_ratio * (2 * rng() - 1)
        delay = min(delay, max_delay_s)
    return max(delay, 0.0)


async def _run_provider(
    client: BaseLLMClient,
    role: str,
    document: str,
    model_cls: type[T],
    *,
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    jitter_ratio: float,
    sleep: Sleep,
    rng: Rng,
    max_tokens: int,
    log: list[AttemptRecord],
) -> tuple[ExtractionResult[T] | None, Exception | None]:
    provider = getattr(client, "provider", "unknown")
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            result = await extract(client, document, model_cls, max_tokens=max_tokens)
        except (ExtractionError, LLMError, httpx.TransportError) as error:
            last_error = error
            delay: float | None = None
            if _is_retryable(error) and attempt < max_attempts - 1:
                delay = compute_delay(
                    attempt,
                    base_delay_s=base_delay_s,
                    max_delay_s=max_delay_s,
                    jitter_ratio=jitter_ratio,
                    retry_after_s=getattr(error, "retry_after_s", None),
                    rng=rng,
                )
            log.append(
                AttemptRecord(
                    role=role,
                    provider=provider,
                    attempt=attempt + 1,
                    outcome="error",
                    error_type=type(error).__name__,
                    status_code=http_status(error) if isinstance(error, LLMError) else None,
                    delay_s=delay,
                )
            )
            if delay is None:
                break
            await sleep(delay)
        else:
            log.append(
                AttemptRecord(
                    role=role,
                    provider=provider,
                    attempt=attempt + 1,
                    outcome="success",
                )
            )
            return result, None

    return None, last_error


async def extract_with_fallback(
    primary_client: BaseLLMClient,
    fallback_client: BaseLLMClient,
    document: str,
    model_cls: type[T],
    *,
    max_attempts: int = 2,
    fallback_max_attempts: int = 1,
    base_delay_s: float = 0.5,
    max_delay_s: float = 30.0,
    jitter_ratio: float = 0.0,
    sleep: Sleep = asyncio.sleep,
    rng: Rng = random.random,
    max_tokens: int = 1500,
) -> ExtractionResult[T]:
    """Retry transient primary failures, then use the fallback provider."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if fallback_max_attempts < 1:
        raise ValueError("fallback_max_attempts must be at least 1")
    if base_delay_s < 0:
        raise ValueError("base_delay_s cannot be negative")
    if max_delay_s < 0:
        raise ValueError("max_delay_s cannot be negative")
    if not 0 <= jitter_ratio <= 1:
        raise ValueError("jitter_ratio must be between 0 and 1")

    log: list[AttemptRecord] = []

    result, primary_error = await _run_provider(
        primary_client,
        "primary",
        document,
        model_cls,
        max_attempts=max_attempts,
        base_delay_s=base_delay_s,
        max_delay_s=max_delay_s,
        jitter_ratio=jitter_ratio,
        sleep=sleep,
        rng=rng,
        max_tokens=max_tokens,
        log=log,
    )
    if result is not None:
        return replace(result, attempts=list(log))
    assert primary_error is not None

    result, fallback_error = await _run_provider(
        fallback_client,
        "fallback",
        document,
        model_cls,
        max_attempts=fallback_max_attempts,
        base_delay_s=base_delay_s,
        max_delay_s=max_delay_s,
        jitter_ratio=jitter_ratio,
        sleep=sleep,
        rng=rng,
        max_tokens=max_tokens,
        log=log,
    )
    if result is not None:
        return replace(result, attempts=list(log))
    assert fallback_error is not None

    raise FallbackError(primary_error, fallback_error, attempts=log) from fallback_error
