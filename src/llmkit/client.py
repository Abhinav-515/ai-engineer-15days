"""Unified async LLM client for Ollama (local) and OpenAI-compatible APIs (cloud)."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Self

import httpx

from llmkit.config import Settings, get_settings

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    """Raised when a provider returns an HTTP error."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes reasoning tokens for reasoning models
    reasoning_tokens: int = 0


@dataclass
class StreamEvent:
    """One decoded piece of a streamed response."""

    content: str = ""
    reasoning: str = ""
    usage: Usage | None = None


@dataclass
class LLMResult:
    provider: str
    model: str
    text: str
    reasoning_text: str
    usage: Usage
    ttft_s: float | None  # to first visible answer token
    ttft_any_s: float | None  # to first token of any kind (incl. hidden reasoning)
    total_s: float
    tokens_per_s: float | None  # decode throughput after the first token
    cost_usd: float
    usage_estimated: bool  # True if the provider sent no usage and we counted chunks


async def _check(resp: httpx.Response) -> None:
    """Raise LLMError with the response body (raise_for_status hides it on streams)."""
    if resp.is_error:
        body = (await resp.aread()).decode(errors="replace")[:500]
        raise LLMError(f"HTTP {resp.status_code} from {resp.request.url}: {body}")


class BaseLLMClient(ABC):
    provider = "base"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 120.0,
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self._http = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    @abstractmethod
    def stream(
        self, messages: Messages, *, max_tokens: int, temperature: float
    ) -> AsyncIterator[StreamEvent]:
        """Yield StreamEvents as the provider streams its response."""

    async def generate(
        self, messages: Messages, *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResult:
        """Consume the stream, collect text, and compute latency/throughput/cost."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: Usage | None = None
        n_chunks = 0
        first_any: float | None = None
        first_content: float | None = None

        start = time.perf_counter()
        async for ev in self.stream(messages, max_tokens=max_tokens, temperature=temperature):
            now = time.perf_counter() - start
            if ev.content or ev.reasoning:
                n_chunks += 1
                if first_any is None:
                    first_any = now
            if ev.content:
                if first_content is None:
                    first_content = now
                content_parts.append(ev.content)
            if ev.reasoning:
                reasoning_parts.append(ev.reasoning)
            if ev.usage is not None:
                usage = ev.usage
        total = time.perf_counter() - start

        estimated = usage is None
        if usage is None:
            usage = Usage(completion_tokens=n_chunks)  # rough: 1 chunk ~ 1 token

        window = total - first_any if first_any is not None else 0.0
        tps = (
            (usage.completion_tokens - 1) / window
            if window > 0 and usage.completion_tokens > 1
            else None
        )
        cost = (
            usage.prompt_tokens * self.price_in_per_m
            + usage.completion_tokens * self.price_out_per_m
        ) / 1_000_000

        return LLMResult(
            provider=self.provider,
            model=self.model,
            text="".join(content_parts),
            reasoning_text="".join(reasoning_parts),
            usage=usage,
            ttft_s=first_content,
            ttft_any_s=first_any,
            total_s=total,
            tokens_per_s=tps,
            cost_usd=cost,
            usage_estimated=estimated,
        )


class OllamaClient(BaseLLMClient):
    """Local Ollama server. Streams newline-delimited JSON from /api/chat."""

    provider = "ollama"

    async def stream(
        self, messages: Messages, *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> AsyncIterator[StreamEvent]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        async with self._http.stream("POST", "/api/chat", json=payload) as resp:
            await _check(resp)
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                msg = data.get("message") or {}
                usage = None
                if data.get("done"):
                    usage = Usage(
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        completion_tokens=data.get("eval_count", 0),
                    )
                yield StreamEvent(
                    content=msg.get("content", ""),
                    reasoning=msg.get("thinking", ""),
                    usage=usage,
                )


def _parse_usage(raw: dict[str, Any] | None) -> Usage | None:
    if not raw:
        return None
    details = raw.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=raw.get("prompt_tokens", 0),
        completion_tokens=raw.get("completion_tokens", 0),
        reasoning_tokens=details.get("reasoning_tokens", 0),
    )


class OpenAICompatClient(BaseLLMClient):
    """Any OpenAI-compatible API (Groq, OpenAI, ...). Streams Server-Sent Events."""

    provider = "cloud"

    def __init__(self, *args: Any, reasoning_effort: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort

    async def stream(
        self, messages: Messages, *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
            await _check(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                # Standard OpenAI puts usage on the last chunk; Groq also uses x_groq.usage.
                usage = _parse_usage(data.get("usage") or (data.get("x_groq") or {}).get("usage"))
                choices = data.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                yield StreamEvent(
                    content=delta.get("content") or "",
                    reasoning=delta.get("reasoning") or "",
                    usage=usage,
                )


def make_client(provider: str, settings: Settings | None = None) -> BaseLLMClient:
    """Factory: provider is 'ollama' (local) or 'cloud'."""
    s = settings or get_settings()
    if provider == "ollama":
        return OllamaClient(s.ollama_model, base_url=s.ollama_base_url)
    if provider == "cloud":
        return OpenAICompatClient(
            s.openai_model,
            base_url=s.openai_base_url,
            headers={"Authorization": f"Bearer {s.openai_api_key.get_secret_value()}"},
            price_in_per_m=s.cloud_price_in_per_m,
            price_out_per_m=s.cloud_price_out_per_m,
            reasoning_effort=s.cloud_reasoning_effort or None,
        )
    raise ValueError(f"Unknown provider: {provider!r} (use 'ollama' or 'cloud')")
