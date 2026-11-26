"""Combine ranked result lists from multiple retrievers."""

from __future__ import annotations

from collections.abc import Sequence

from llmkit.vector_store import SearchResult


def _validate_lists(result_lists: Sequence[Sequence[SearchResult]]) -> None:
    if not result_lists:
        raise ValueError("result_lists must not be empty")


def _finalize(
    scores: dict[str, float],
    documents: dict[str, SearchResult],
    limit: int | None,
) -> list[SearchResult]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ranked = ranked[:limit]
    return [
        SearchResult(
            id=doc_id,
            text=documents[doc_id].text,
            score=score,
            metadata=documents[doc_id].metadata,
        )
        for doc_id, score in ranked
    ]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchResult]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[SearchResult]:
    """Fuse ranked lists using Reciprocal Rank Fusion (rank-based, scale-free)."""
    _validate_lists(result_lists)
    if k <= 0:
        raise ValueError("k must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    scores: dict[str, float] = {}
    documents: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank)
            documents.setdefault(result.id, result)

    return _finalize(scores, documents, limit)


def _min_max(results: Sequence[SearchResult]) -> dict[str, float]:
    if not results:
        return {}
    values = [result.score for result in results]
    low, high = min(values), max(values)
    if high == low:
        return {result.id: 1.0 for result in results}
    return {result.id: (result.score - low) / (high - low) for result in results}


def weighted_score_fusion(
    dense: Sequence[SearchResult],
    lexical: Sequence[SearchResult],
    *,
    alpha: float = 0.5,
    limit: int | None = None,
) -> list[SearchResult]:
    """Fuse dense and lexical results by min-max normalising and weighting scores.

    The final score is alpha * dense + (1 - alpha) * lexical.
    A document missing from one list contributes 0 from that side.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    dense_scores = _min_max(dense)
    lexical_scores = _min_max(lexical)

    documents: dict[str, SearchResult] = {}
    for result in [*dense, *lexical]:
        documents.setdefault(result.id, result)

    scores = {
        doc_id: alpha * dense_scores.get(doc_id, 0.0)
        + (1 - alpha) * lexical_scores.get(doc_id, 0.0)
        for doc_id in documents
    }

    return _finalize(scores, documents, limit)
