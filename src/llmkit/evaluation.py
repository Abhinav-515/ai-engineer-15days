"""Retrieval evaluation metrics: Precision@k, Recall@k and MRR."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def _validate_relevance(relevant_ids: Collection[str]) -> None:
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """Calculate the fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_relevance(relevant_ids)

    top_k = retrieved_ids[:k]
    return sum(doc_id in relevant_ids for doc_id in top_k) / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """Calculate the fraction of relevant documents found in the top k."""
    if k <= 0:
        raise ValueError("k must be positive")
    _validate_relevance(relevant_ids)

    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    _validate_relevance(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]],
    relevance: Sequence[Collection[str]],
) -> float:
    """Calculate mean reciprocal rank across multiple queries."""
    if not rankings:
        raise ValueError("rankings must not be empty")
    if len(rankings) != len(relevance):
        raise ValueError("rankings and relevance must have the same length")

    scores = [
        reciprocal_rank(ranking, relevant_ids)
        for ranking, relevant_ids in zip(rankings, relevance, strict=True)
    ]
    return sum(scores) / len(scores)
