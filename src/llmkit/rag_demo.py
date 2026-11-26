"""Day 7 demo: grounded RAG with hybrid retrieval and source citations."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from llmkit.client import make_client
from llmkit.embeddings import OllamaEmbedder
from llmkit.hybrid import HybridRetriever
from llmkit.rag import RAGService
from llmkit.vector_store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "output" / "day7" / "qdrant"
DEFAULT_COLLECTION = "day7_recursive"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default=(
            "For invoice NS-2026-204, what vendor, line-item amounts, "
            "total, and payment status are listed?"
        ),
        help="Question to answer using the indexed documents.",
    )
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval strategy to use.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    print("\n=== Day 7: Grounded RAG Demo ===")
    print(f"Database:   {args.db_dir}")
    print(f"Collection: {args.collection}")
    print(f"Retrieval:  {args.mode}")
    print(f"Question:   {args.query}")

    with QdrantVectorStore(
        args.db_dir,
        collection_name=args.collection,
        vector_size=768,
    ) as store:
        documents = store.scroll_all()

        if not documents:
            raise RuntimeError(
                f"Collection '{args.collection}' is empty. Run the Day 5 ingestion demo first."
            )

        print(f"\nIndexed chunks: {len(documents)}")

        async with OllamaEmbedder() as embedder:
            retriever = HybridRetriever(store, embedder)

            async with make_client("ollama") as client:
                service = RAGService(
                    retriever,
                    client,
                    retrieval_mode=args.mode,
                    top_k=args.top_k,
                )

                result = await service.ask(args.query)

    print("\n" + "=" * 68)
    print("GROUNDED ANSWER")
    print("=" * 68)
    print(result.answer)

    print("\n" + "=" * 68)
    print("SOURCE CITATIONS")
    print("=" * 68)

    if result.citations:
        for citation in result.citations:
            print(
                f"{citation.label} {Path(citation.document).name} "
                f"| chunk={citation.chunk_index} "
                f"| kind={citation.kind} "
                f"| strategy={citation.strategy}"
            )
    else:
        print("No source citations (answer was refused or abstained).")

    print("\n" + "=" * 68)
    print("EXECUTION SUMMARY")
    print("=" * 68)
    print(f"Retrieved chunks: {result.retrieved_count}")
    print(f"Abstained:        {result.abstained}")
    print(f"Provider:         {result.provider}")
    print(f"Generation time:  {result.latency_s:.2f}s")
    print(f"Estimated cost:   ${result.cost_usd:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
