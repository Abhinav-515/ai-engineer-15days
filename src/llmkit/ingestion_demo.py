"""Day 5 demo: document chunking, Ollama embeddings and Qdrant retrieval."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from llmkit.embeddings import OllamaEmbedder
from llmkit.ingestion import ChunkStrategy, ingest_file
from llmkit.vector_store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples" / "day4"

SAMPLE_FILES = [
    SAMPLES / "invoices.csv",
    SAMPLES / "messy_invoice.txt",
    SAMPLES / "application.log",
]


async def run_strategy(
    strategy: ChunkStrategy,
    embedder: OllamaEmbedder,
    db_dir: Path,
    query: str,
    collection_prefix: str,
) -> None:
    collection = f"{collection_prefix}_{strategy}"

    print(f"\n{'=' * 68}")
    print(f"CHUNKING STRATEGY: {strategy.upper()}")
    print("=" * 68)

    with QdrantVectorStore(
        db_dir,
        collection_name=collection,
        vector_size=768,
    ) as store:
        for path in SAMPLE_FILES:
            if not path.is_file():
                print(f"[SKIP] Missing sample: {path.name}")
                continue

            try:
                result = await ingest_file(
                    path,
                    embedder,
                    store,
                    strategy=strategy,
                )
            except Exception as error:
                print(f"[FAILED] {path.name}: {type(error).__name__}: {error}")
                continue

            print(
                f"[OK] {path.name} | chunks={result.chunks_created} | stored={result.chunks_stored}"
            )

        query_vector = (await embedder.embed([query]))[0]
        results = store.search(query_vector, limit=5)

        print(f"\nQuery: {query}")
        print(f"Top results: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. Score: {result.score:.4f} | "
                f"Source: {result.metadata.get('source', 'unknown')} | "
                f"Chunk: {result.metadata.get('chunk_index', 'n/a')}"
            )
            print(f"   {result.text[:300].replace(chr(10), ' ')}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 5: compare chunking strategies with Ollama and Qdrant"
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["fixed", "recursive", "semantic"],
        default=["recursive"],
    )
    parser.add_argument(
        "--query",
        default="What are the invoice amounts, vendors, and payment details?",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=ROOT / "output" / "day5" / "qdrant",
    )
    parser.add_argument(
        "--collection-prefix",
        default="day5",
        help="Prefix for Qdrant collection names (default: day5).",
    )
    args = parser.parse_args()

    args.db_dir.mkdir(parents=True, exist_ok=True)

    async with OllamaEmbedder() as embedder:
        for strategy in args.strategies:
            await run_strategy(
                strategy,
                embedder,
                args.db_dir,
                args.query,
                args.collection_prefix,
            )


if __name__ == "__main__":
    asyncio.run(main())
