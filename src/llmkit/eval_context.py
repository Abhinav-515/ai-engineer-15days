"""Context Recall and Context Precision based on evidence snippets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk, reduced to what context metrics need."""

    id: str
    text: str


@dataclass(frozen=True)
class ContextScores:
    """Context metrics for one answerable case.

    ``precision`` is ``None`` when nothing was retrieved, because 0/0 is undefined.
    """

    recall: float
    precision: float | None
    found_snippets: tuple[str, ...]
    missing_snippets: tuple[str, ...]
    relevant_chunk_ids: tuple[str, ...]
    retrieved_count: int


def normalize_for_match(text: str) -> str:
    """Collapse whitespace and casefold so matching ignores layout and case."""
    return " ".join(text.split()).casefold()


def _distinct_snippets(evidence: Sequence[str]) -> dict[str, str]:
    """Map normalized snippet -> display form, dropping duplicates."""
    if not evidence:
        raise ValueError("evidence must not be empty")

    distinct: dict[str, str] = {}
    for snippet in evidence:
        key = normalize_for_match(snippet)
        if not key:
            raise ValueError("evidence snippets must not be blank")
        distinct.setdefault(key, " ".join(snippet.split()))
    return distinct


def score_context(chunks: Sequence[RetrievedChunk], evidence: Sequence[str]) -> ContextScores:
    """Score retrieved chunks against the evidence snippets of an answerable case."""
    distinct = _distinct_snippets(evidence)
    normalized = [(chunk, normalize_for_match(chunk.text)) for chunk in chunks]

    found = [
        display for key, display in distinct.items() if any(key in text for _, text in normalized)
    ]
    missing = [display for display in distinct.values() if display not in found]
    relevant_ids = [chunk.id for chunk, text in normalized if any(key in text for key in distinct)]

    precision = len(relevant_ids) / len(chunks) if chunks else None
    return ContextScores(
        recall=len(found) / len(distinct),
        precision=precision,
        found_snippets=tuple(found),
        missing_snippets=tuple(missing),
        relevant_chunk_ids=tuple(relevant_ids),
        retrieved_count=len(chunks),
    )


def context_recall(chunks: Sequence[RetrievedChunk], evidence: Sequence[str]) -> float:
    """Share of distinct evidence snippets found in at least one retrieved chunk."""
    return score_context(chunks, evidence).recall


def context_precision(chunks: Sequence[RetrievedChunk], evidence: Sequence[str]) -> float | None:
    """Share of retrieved chunks containing at least one snippet (None if no chunks)."""
    return score_context(chunks, evidence).precision
