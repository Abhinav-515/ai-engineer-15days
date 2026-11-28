"""Okapi BM25 lexical search over a fixed corpus of chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from llmkit.vector_store import SearchResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 index over SearchResult documents."""

    def __init__(
        self,
        documents: Sequence[SearchResult],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._documents = list(documents)
        self._term_freqs = [Counter(tokenize(doc.text)) for doc in self._documents]
        self._lengths = [sum(freqs.values()) for freqs in self._term_freqs]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

        doc_freq: Counter[str] = Counter()
        for freqs in self._term_freqs:
            doc_freq.update(freqs.keys())

        total = len(self._documents)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._documents)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        """Return up to `limit` documents with a positive BM25 score, best first."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        query_terms = set(tokenize(query))
        if not query_terms or not self._documents:
            return []

        scored: list[tuple[float, int]] = []
        for index, freqs in enumerate(self._term_freqs):
            score = self._score(query_terms, freqs, self._lengths[index])
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], self._documents[item[1]].id))

        return [
            SearchResult(
                id=self._documents[index].id,
                text=self._documents[index].text,
                score=score,
                metadata=self._documents[index].metadata,
            )
            for score, index in scored[:limit]
        ]

    def _score(self, query_terms: set[str], freqs: Counter[str], length: int) -> float:
        score = 0.0
        norm = 1 - self.b + self.b * (length / self._avg_length) if self._avg_length else 1.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            score += self._idf[term] * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)
        return score
