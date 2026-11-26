"""Hybrid retrieval: dense vectors + BM25 + optional cross-encoder reranking."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal, Protocol

from llmkit.fusion import reciprocal_rank_fusion, weighted_score_fusion
from llmkit.lexical import BM25Index
from llmkit.reranker import Reranker
from llmkit.vector_store import QdrantVectorStore, SearchResult

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid+rerank"]
FusionMethod = Literal["rrf", "weighted"]

RETRIEVAL_MODES: tuple[str, ...] = ("dense", "bm25", "hybrid", "hybrid+rerank")
FUSION_METHODS: tuple[str, ...] = ("rrf", "weighted")


class QueryEmbedder(Protocol):
    """Minimal async interface needed to embed a search query."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert texts into embedding vectors."""


class HybridRetriever:
    """Retrieve chunks with dense, BM25, hybrid or hybrid+rerank strategies."""

    def __init__(
        self,
        store: QdrantVectorStore,
        embedder: QueryEmbedder,
        *,
        reranker: Reranker | None = None,
        candidate_pool: int = 20,
        fusion: FusionMethod = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if candidate_pool <= 0:
            raise ValueError("candidate_pool must be positive")
        if fusion not in FUSION_METHODS:
            raise ValueError(f"fusion must be one of {FUSION_METHODS}")

        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self.candidate_pool = candidate_pool
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._bm25: BM25Index | None = None

    def _build_index(self) -> BM25Index:
        return BM25Index(self._store.scroll_all(), k1=self._bm25_k1, b=self._bm25_b)

    def refresh(self) -> int:
        """Rebuild the BM25 index from the vector store; returns the document count."""
        self._bm25 = self._build_index()
        return len(self._bm25)

    def _index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = self._build_index()
        return self._bm25

    async def _dense(self, query: str, limit: int) -> list[SearchResult]:
        vectors = await self._embedder.embed([query])
        return self._store.search(vectors[0], limit=limit)

    def _fuse(
        self,
        dense: list[SearchResult],
        lexical: list[SearchResult],
        limit: int,
    ) -> list[SearchResult]:
        if self.fusion == "rrf":
            return reciprocal_rank_fusion([dense, lexical], k=self.rrf_k, limit=limit)
        return weighted_score_fusion(dense, lexical, alpha=self.alpha, limit=limit)

    async def search(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid+rerank",
        limit: int = 5,
    ) -> list[SearchResult]:
        """Return the top `limit` chunks for the query using the chosen mode."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"mode must be one of {RETRIEVAL_MODES}")
        if mode == "hybrid+rerank" and self._reranker is None:
            raise ValueError("hybrid+rerank mode requires a reranker")

        if mode == "bm25":
            return self._index().search(query, limit=limit)
        if mode == "dense":
            return await self._dense(query, limit)

        pool = max(self.candidate_pool, limit)
        dense = await self._dense(query, pool)
        lexical = self._index().search(query, limit=pool)
        fused = self._fuse(dense, lexical, pool)

        if mode == "hybrid":
            return fused[:limit]

        assert self._reranker is not None
        return await asyncio.to_thread(self._reranker.rerank, query, fused, limit=limit)
