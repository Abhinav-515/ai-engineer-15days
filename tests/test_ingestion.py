import pytest

from llmkit.ingestion import ingest_document, ingest_file
from llmkit.loaders import RawDocument
from llmkit.vector_store import QdrantVectorStore


class FakeEmbedder:
    async def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_document(text="Finance invoice processing. Finance invoice validation."):
    return RawDocument(
        source="invoice.txt",
        kind="text",
        text=text,
    )


def make_store(tmp_path):
    return QdrantVectorStore(
        tmp_path / "qdrant",
        collection_name="ingestion_test",
        vector_size=3,
    )


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
async def test_ingest_document_supports_all_strategies(tmp_path, strategy):
    with make_store(tmp_path) as store:
        result = await ingest_document(
            make_document(),
            FakeEmbedder(),
            store,
            strategy=strategy,
            chunk_size=30,
            overlap=5,
            max_chunk_size=100,
            similarity_threshold=0.5,
        )
        results = store.search([1.0, 0.0, 0.0], limit=10)

    assert result.chunks_created > 0
    assert result.chunks_stored == result.chunks_created
    assert result.strategy == strategy
    assert len(results) == result.chunks_stored
    assert results[0].metadata["source"] == "invoice.txt"
    assert results[0].metadata["strategy"] == strategy


async def test_ingest_empty_document_stores_nothing(tmp_path):
    with make_store(tmp_path) as store:
        result = await ingest_document(
            make_document("   "),
            FakeEmbedder(),
            store,
        )

    assert result.chunks_created == 0
    assert result.chunks_stored == 0


async def test_ingest_rejects_unknown_strategy(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="Unsupported"):
            await ingest_document(
                make_document(),
                FakeEmbedder(),
                store,
                strategy="unknown",
            )


async def test_ingest_file_loads_and_indexes_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Invoice processing is important. Validation catches errors.")

    with make_store(tmp_path) as store:
        result = await ingest_file(path, FakeEmbedder(), store, strategy="recursive")
        results = store.search([1.0, 0.0, 0.0])

    assert result.kind == "text"
    assert result.chunks_stored > 0
    assert results[0].metadata["source"] == str(path)
