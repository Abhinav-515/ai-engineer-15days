"""Persistent local vector storage powered by Qdrant."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class SearchResult:
    """One ranked vector-search result."""

    id: str
    text: str
    score: float
    metadata: dict


class QdrantVectorStore:
    """Store and search embedded text chunks in a persistent Qdrant collection."""

    def __init__(
        self,
        path: str | Path = "output/day5/qdrant",
        *,
        collection_name: str = "documents",
        vector_size: int = 768,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = QdrantClient(path=str(path))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self._client.get_collections().collections
        existing = {collection.name for collection in collections}

        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        info = self._client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        stored_size = getattr(vectors, "size", None)

        if stored_size != self.vector_size:
            raise ValueError(
                f"Collection uses {stored_size} dimensions; requested {self.vector_size}"
            )

    def close(self) -> None:
        """Close the local Qdrant database."""
        self._client.close()

    def __enter__(self) -> QdrantVectorStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def upsert(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadata: Sequence[dict] | None = None,
    ) -> int:
        """Insert or update text chunks and their embedding vectors."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        if metadata is not None and len(metadata) != len(texts):
            raise ValueError("metadata and texts must have the same length")

        if not texts:
            return 0

        points = []

        for index, (text, embedding) in enumerate(zip(texts, embeddings, strict=True)):
            if not isinstance(text, str) or not text.strip():
                raise ValueError("texts must contain only non-empty strings")

            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Expected {self.vector_size} embedding dimensions, received {len(embedding)}"
                )

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in embedding
            ):
                raise ValueError("Embeddings must contain finite numeric values")

            point_id = str(uuid5(NAMESPACE_URL, f"{self.collection_name}:{text}"))
            payload = {
                "text": text,
                "metadata": metadata[index] if metadata is not None else {},
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[float(value) for value in embedding],
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Return the most similar stored chunks, ranked by cosine similarity."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        if len(embedding) != self.vector_size:
            raise ValueError(
                f"Expected {self.vector_size} query dimensions, received {len(embedding)}"
            )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in embedding
        ):
            raise ValueError("Query embedding must contain finite numeric values")

        if score_threshold is not None and not -1 <= score_threshold <= 1:
            raise ValueError("score_threshold must be between -1 and 1")

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=[float(value) for value in embedding],
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchResult(
                    id=str(point.id),
                    text=payload.get("text", ""),
                    score=float(point.score),
                    metadata=payload.get("metadata", {}),
                )
            )

        return results

    def scroll_all(self, *, batch_size: int = 256) -> list[SearchResult]:
        """Return every stored chunk (score is 0.0), ordered by point id."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        results: list[SearchResult] = []
        offset = None

        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                results.append(
                    SearchResult(
                        id=str(point.id),
                        text=payload.get("text", ""),
                        score=0.0,
                        metadata=payload.get("metadata", {}),
                    )
                )
            if offset is None:
                break

        return sorted(results, key=lambda result: result.id)
