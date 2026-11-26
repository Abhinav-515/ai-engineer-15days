"""Compare dense, BM25, hybrid and optional reranked retrieval."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from llmkit.embeddings import OllamaEmbedder
from llmkit.evaluation import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from llmkit.hybrid import HybridRetriever
from llmkit.reranker import CrossEncoderReranker
from llmkit.vector_store import QdrantVectorStore

K = 3

# Relevance labels refer to (source filename, chunk_index).
GOLDEN_QUERIES: list[dict[str, Any]] = [
    {
        "query": "What products and quantities are listed on invoice INV-2026-101?",
        "relevant": {("invoices.csv", 0)},
    },
    {
        "query": "What is the total amount and tax for invoice NS-2026-204?",
        "relevant": {("messy_invoice.txt", 1)},
    },
    {
        "query": "When is invoice NS-2026-204 payment due?",
        "relevant": {("messy_invoice.txt", 0)},
    },
    {
        "query": "What happened with the payment gateway?",
        "relevant": {("application.log", 0)},
    },
    {
        "query": "Why was the database connection slow?",
        "relevant": {("application.log", 0)},
    },
]


def document_key(document: Any) -> tuple[str, int]:
    """Build a stable relevance key from stored document metadata."""
    source = Path(str(document.metadata.get("source", ""))).name
    chunk_index = int(document.metadata.get("chunk_index", -1))
    return source, chunk_index


async def evaluate_mode(
    retriever: HybridRetriever,
    mode: str,
    documents: list[Any],
) -> None:
    """Evaluate one retrieval strategy against the golden queries."""
    ids_by_key = {document_key(doc): doc.id for doc in documents}

    rankings: list[list[str]] = []
    relevance_sets: list[set[str]] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    precision_at_1_scores: list[float] = []
    recall_at_1_scores: list[float] = []

    print(f"\n{'=' * 72}")
    print(f"RETRIEVAL MODE: {mode.upper()}")
    print(f"{'=' * 72}")

    for case in GOLDEN_QUERIES:
        relevant_ids = {ids_by_key[key] for key in case["relevant"] if key in ids_by_key}

        if not relevant_ids:
            raise ValueError(f"No stored documents match relevance labels: {case['relevant']}")

        results = await retriever.search(
            case["query"],
            mode=mode,  # type: ignore[arg-type]
            limit=K,
        )
        ranking = [result.id for result in results]

        rankings.append(ranking)
        relevance_sets.append(relevant_ids)

        precision = precision_at_k(ranking, relevant_ids, k=K)
        recall = recall_at_k(ranking, relevant_ids, k=K)
        p1 = precision_at_k(ranking, relevant_ids, k=1)
        r1 = recall_at_k(ranking, relevant_ids, k=1)
        rr = reciprocal_rank(ranking, relevant_ids)

        precision_scores.append(precision)
        recall_scores.append(recall)
        precision_at_1_scores.append(p1)
        recall_at_1_scores.append(r1)

        print(f"\nQuery: {case['query']}")
        print(
            f"Precision@1: {p1:.3f} | Recall@1: {r1:.3f} | "
            f"Precision@{K}: {precision:.3f} | "
            f"Recall@{K}: {recall:.3f} | RR: {rr:.3f}"
        )

        for rank, result in enumerate(results, start=1):
            marker = "RELEVANT" if result.id in relevant_ids else "other"
            print(
                f"  {rank}. [{marker}] "
                f"{Path(str(result.metadata.get('source', ''))).name} "
                f"chunk={result.metadata.get('chunk_index')} "
                f"score={result.score:.4f}"
            )

    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_p1 = sum(precision_at_1_scores) / len(precision_at_1_scores)
    mean_r1 = sum(recall_at_1_scores) / len(recall_at_1_scores)
    mrr = mean_reciprocal_rank(rankings, relevance_sets)

    print(f"\n--- {mode.upper()} SUMMARY ---")
    print(f"Mean Precision@1:   {mean_p1:.3f}")
    print(f"Mean Recall@1:      {mean_r1:.3f}")
    print(f"Mean Precision@{K}: {mean_precision:.3f}")
    print(f"Mean Recall@{K}:    {mean_recall:.3f}")
    print(f"MRR:                {mrr:.3f}")


async def main(include_rerank: bool) -> None:
    db_dir = Path("output/day5/qdrant")

    with QdrantVectorStore(
        db_dir,
        collection_name="day5_recursive",
        vector_size=768,
    ) as store:
        documents = store.scroll_all()

        if not documents:
            raise RuntimeError("The evaluation collection contains no documents.")

        print(f"Evaluation corpus: {len(documents)} chunks")
        print(f"Golden queries: {len(GOLDEN_QUERIES)}")
        print(f"Metrics: Precision@1, Recall@1, Precision@{K}, Recall@{K}, MRR")

        reranker = CrossEncoderReranker() if include_rerank else None

        async with OllamaEmbedder() as embedder:
            retriever = HybridRetriever(
                store,
                embedder,
                reranker=reranker,
                candidate_pool=20,
                fusion="rrf",
            )

            modes = ["dense", "bm25", "hybrid"]
            if include_rerank:
                modes.append("hybrid+rerank")

            for mode in modes:
                await evaluate_mode(retriever, mode, documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Include the real cross-encoder reranking strategy.",
    )
    args = parser.parse_args()

    asyncio.run(main(args.rerank))
