"""Document ingestion: load, chunk, embed and store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llmkit.chunking import (
    async_semantic_chunks,
    fixed_size_chunks,
    recursive_chunks,
)
from llmkit.embeddings import OllamaEmbedder
from llmkit.loaders import RawDocument, load_document
from llmkit.vector_store import QdrantVectorStore

ChunkStrategy = Literal["fixed", "recursive", "semantic"]


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one document ingestion operation."""

    source: str
    kind: str
    strategy: str
    chunks_created: int
    chunks_stored: int


async def ingest_document(
    document: RawDocument,
    embedder: OllamaEmbedder,
    store: QdrantVectorStore,
    *,
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
    similarity_threshold: float = 0.65,
    max_chunk_size: int = 1000,
) -> IngestionResult:
    """Chunk, embed and store an already-loaded document."""
    if strategy == "fixed":
        chunks = fixed_size_chunks(
            document.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif strategy == "recursive":
        chunks = recursive_chunks(
            document.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    elif strategy == "semantic":
        chunks = await async_semantic_chunks(
            document.text,
            embedder,
            similarity_threshold=similarity_threshold,
            max_chunk_size=max_chunk_size,
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    if not chunks:
        return IngestionResult(
            source=document.source,
            kind=document.kind,
            strategy=strategy,
            chunks_created=0,
            chunks_stored=0,
        )

    vectors = await embedder.embed(chunks)
    metadata = [
        {
            "source": document.source,
            "kind": document.kind,
            "chunk_index": index,
            "strategy": strategy,
        }
        for index in range(len(chunks))
    ]

    stored = store.upsert(chunks, vectors, metadata)

    return IngestionResult(
        source=document.source,
        kind=document.kind,
        strategy=strategy,
        chunks_created=len(chunks),
        chunks_stored=stored,
    )


async def ingest_file(
    path: str | Path,
    embedder: OllamaEmbedder,
    store: QdrantVectorStore,
    **options,
) -> IngestionResult:
    """Load a supported file and send it through the ingestion workflow."""
    document = await load_document(path)
    return await ingest_document(document, embedder, store, **options)
