import pytest

from llmkit.hybrid import HybridRetriever
from llmkit.vector_store import QdrantVectorStore

PAYMENTS = "general payment overview and billing policy"
INVOICE = "invoice INV-2026-101 from Northstar is unpaid"
WEATHER = "weather report sunny with light winds"

QUERY = "INV-2026-101"
VECTORS = {
    PAYMENTS: [1.0, 0.0, 0.0],
    INVOICE: [0.0, 1.0, 0.0],
    WEATHER: [0.0, 0.0, 1.0],
}


class FakeEmbedder:
    """Maps known query strings to fixed vectors; the query is dense-closest to PAYMENTS."""

    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.2, 0.0] for _ in texts]


class FakeReranker:
    def __init__(self, scores_by_text):
        self.scores = scores_by_text
        self.calls = []

    def rerank(self, query, documents, *, limit=None, min_score=None):
        self.calls.append((query, [d.text for d in documents], limit))
        ranked = sorted(documents, key=lambda d: -self.scores[d.text])
        return ranked[:limit] if limit else ranked


@pytest.fixture
def store(tmp_path):
    with QdrantVectorStore(
        tmp_path / "qdrant", collection_name="hybrid_test", vector_size=3
    ) as vector_store:
        vector_store.upsert(list(VECTORS), list(VECTORS.values()))
        yield vector_store


async def test_dense_mode_returns_nearest_vector(store):
    retriever = HybridRetriever(store, FakeEmbedder())

    results = await retriever.search(QUERY, mode="dense", limit=3)

    assert results[0].text == PAYMENTS


async def test_bm25_mode_finds_exact_identifier_without_embedding(store):
    embedder = FakeEmbedder()
    retriever = HybridRetriever(store, embedder)

    results = await retriever.search(QUERY, mode="bm25", limit=3)

    assert [r.text for r in results] == [INVOICE]
    assert embedder.calls == []


async def test_hybrid_recovers_chunk_dense_search_ranks_too_low(store):
    retriever = HybridRetriever(store, FakeEmbedder())

    results = await retriever.search(QUERY, mode="hybrid", limit=3)

    assert results[0].text == INVOICE
    assert PAYMENTS in {r.text for r in results}


async def test_hybrid_respects_limit(store):
    retriever = HybridRetriever(store, FakeEmbedder())

    results = await retriever.search(QUERY, mode="hybrid", limit=1)

    assert len(results) == 1


async def test_weighted_fusion_with_alpha_zero_follows_lexical_ranking(store):
    retriever = HybridRetriever(store, FakeEmbedder(), fusion="weighted", alpha=0.0)

    results = await retriever.search(QUERY, mode="hybrid", limit=3)

    assert results[0].text == INVOICE


async def test_hybrid_rerank_uses_reranker_order_on_fused_candidates(store):
    reranker = FakeReranker({PAYMENTS: 0.1, INVOICE: 0.5, WEATHER: 0.9})
    retriever = HybridRetriever(store, FakeEmbedder(), reranker=reranker)

    results = await retriever.search(QUERY, mode="hybrid+rerank", limit=1)

    assert [r.text for r in results] == [WEATHER]
    query, candidate_texts, limit = reranker.calls[0]
    assert query == QUERY
    assert set(candidate_texts) == set(VECTORS)
    assert limit == 1


async def test_rerank_mode_requires_a_reranker(store):
    retriever = HybridRetriever(store, FakeEmbedder())

    with pytest.raises(ValueError, match="reranker"):
        await retriever.search(QUERY, mode="hybrid+rerank")


async def test_bm25_index_is_cached_until_refresh(store):
    retriever = HybridRetriever(store, FakeEmbedder())
    assert await retriever.search("zebra", mode="bm25") == []

    store.upsert(["brand new zebra document"], [[0.5, 0.5, 0.0]])

    assert await retriever.search("zebra", mode="bm25") == []
    assert retriever.refresh() == 4
    results = await retriever.search("zebra", mode="bm25")
    assert [r.text for r in results] == ["brand new zebra document"]


async def test_search_rejects_invalid_arguments(store):
    retriever = HybridRetriever(store, FakeEmbedder())

    with pytest.raises(ValueError, match="query"):
        await retriever.search("   ")
    with pytest.raises(ValueError, match="limit"):
        await retriever.search(QUERY, limit=0)
    with pytest.raises(ValueError, match="mode"):
        await retriever.search(QUERY, mode="nonsense")


def test_constructor_rejects_invalid_arguments(store):
    with pytest.raises(ValueError, match="candidate_pool"):
        HybridRetriever(store, FakeEmbedder(), candidate_pool=0)
    with pytest.raises(ValueError, match="fusion"):
        HybridRetriever(store, FakeEmbedder(), fusion="magic")
