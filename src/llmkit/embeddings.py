"""Ollama embedding client for dense text representations."""

from __future__ import annotations

import math
from collections.abc import Sequence

import httpx

from llmkit.config import Settings, get_settings


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails or returns invalid data."""


class OllamaEmbedder:
    """Generate text embeddings through Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
        settings: Settings | None = None,
    ) -> None:
        config = settings or get_settings()
        self.model = model or config.ollama_embedding_model
        self._http = httpx.AsyncClient(
            base_url=base_url or config.ollama_base_url,
            timeout=timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> OllamaEmbedder:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts and validate the returned vectors."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings")

        try:
            response = await self._http.post(
                "/api/embed",
                json={"model": self.model, "input": list(texts)},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise EmbeddingError(
                f"Ollama embedding request failed with HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc
        except ValueError as exc:
            raise EmbeddingError("Ollama returned invalid JSON") from exc

        vectors = data.get("embeddings") if isinstance(data, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama returned an invalid number of embeddings")

        validated: list[list[float]] = []
        dimension: int | None = None

        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("Ollama returned an empty or invalid embedding")

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise EmbeddingError("Embedding contains non-finite or non-numeric values")

            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingError("Embedding vectors have inconsistent dimensions")

            validated.append([float(value) for value in vector])

        return validated
