import pytest

from llmkit.vector_store import QdrantVectorStore


def make_store(tmp_path, *, vector_size=3):
    return QdrantVectorStore(
        tmp_path / "qdrant",
        collection_name="test_documents",
        vector_size=vector_size,
    )


def test_upsert_and_search_returns_ranked_results(tmp_path):
    with make_store(tmp_path) as store:
        count = store.upsert(
            ["finance invoice", "sports football"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        results = store.search([0.9, 0.1, 0.0], limit=2)

    assert count == 2
    assert len(results) == 2
    assert results[0].text == "finance invoice"
    assert results[0].score > results[1].score


def test_upsert_preserves_metadata(tmp_path):
    with make_store(tmp_path) as store:
        store.upsert(
            ["invoice document"],
            [[1.0, 0.0, 0.0]],
            metadata=[{"source": "invoice.pdf", "page": 1}],
        )
        results = store.search([1.0, 0.0, 0.0])

    assert results[0].metadata == {"source": "invoice.pdf", "page": 1}


def test_repeated_upsert_updates_existing_point(tmp_path):
    with make_store(tmp_path) as store:
        store.upsert(["same text"], [[1.0, 0.0, 0.0]])
        store.upsert(["same text"], [[0.0, 1.0, 0.0]])

        results = store.search([0.0, 1.0, 0.0])

    assert len(results) == 1
    assert results[0].text == "same text"


def test_store_persists_after_reopening(tmp_path):
    path = tmp_path / "persistent"

    with QdrantVectorStore(path, collection_name="persistent_test", vector_size=3) as store:
        store.upsert(["persisted text"], [[1.0, 0.0, 0.0]])

    with QdrantVectorStore(path, collection_name="persistent_test", vector_size=3) as store:
        results = store.search([1.0, 0.0, 0.0])

    assert len(results) == 1
    assert results[0].text == "persisted text"


def test_upsert_rejects_mismatched_lengths(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="same length"):
            store.upsert(["one", "two"], [[1.0, 0.0, 0.0]])


def test_upsert_rejects_wrong_dimensions(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="dimensions"):
            store.upsert(["text"], [[1.0, 0.0]])


def test_upsert_rejects_blank_text(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="non-empty"):
            store.upsert([" "], [[1.0, 0.0, 0.0]])


def test_search_rejects_wrong_dimensions(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="dimensions"):
            store.search([1.0, 0.0])


def test_search_rejects_invalid_limit(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="limit"):
            store.search([1.0, 0.0, 0.0], limit=0)


def test_search_rejects_invalid_score_threshold(tmp_path):
    with make_store(tmp_path) as store:
        with pytest.raises(ValueError, match="score_threshold"):
            store.search([1.0, 0.0, 0.0], score_threshold=2)


def test_store_rejects_existing_collection_with_wrong_dimensions(tmp_path):
    path = tmp_path / "wrong_dimensions"

    with QdrantVectorStore(path, collection_name="dimension_test", vector_size=3):
        pass

    with pytest.raises(ValueError, match="dimensions"):
        QdrantVectorStore(path, collection_name="dimension_test", vector_size=4)
