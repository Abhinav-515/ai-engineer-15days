"""Cross-encoder reranking of retrieved chunks."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from llmkit.vector_store import SearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerError(RuntimeError):
    """Raised when reranking cannot be performed or returns invalid scores."""


class Reranker(Protocol):
    """Interface for components that reorder candidate chunks for a query."""

    def rerank(
        self,
        query: str,
        documents: Sequence[SearchResult],
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Return documents reordered by relevance to the query."""


class CrossEncoderReranker:
    """Rerank chunks with a sentence-transformers cross-encoder (lazy-loaded)."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self._model = model

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "sentence-transformers is not installed. "
                    'Install it with: pip install -e ".[rerank]"'
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[SearchResult],
        *,
        limit: int | None = None,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Score each (query, chunk) pair and return chunks best first."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        if not documents:
            return []

        model = self._load()
        pairs = [(query, document.text) for document in documents]

        try:
            raw_scores = list(model.predict(pairs))
        except Exception as exc:
            raise RerankerError(f"Cross-encoder scoring failed: {exc}") from exc

        if len(raw_scores) != len(documents):
            raise RerankerError("Cross-encoder returned an unexpected number of scores")

        scores = [float(score) for score in raw_scores]
        if any(not math.isfinite(score) for score in scores):
            raise RerankerError("Cross-encoder returned non-finite scores")

        reranked = [
            SearchResult(
                id=document.id,
                text=document.text,
                score=score,
                metadata=document.metadata,
            )
            for document, score in zip(documents, scores, strict=True)
            if min_score is None or score >= min_score
        ]
        reranked.sort(key=lambda result: (-result.score, result.id))

        return reranked[:limit] if limit is not None else reranked
