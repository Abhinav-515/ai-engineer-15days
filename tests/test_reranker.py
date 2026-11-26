import sys

import pytest

from llmkit.reranker import CrossEncoderReranker, RerankerError
from llmkit.vector_store import SearchResult


class FakeModel:
    """Stands in for sentence_transformers.CrossEncoder."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return self.scores


def doc(doc_id, text=None, **metadata):
    return SearchResult(id=doc_id, text=text or f"text {doc_id}", score=0.0, metadata=metadata)


def test_rerank_orders_documents_by_model_score():
    reranker = CrossEncoderReranker(model=FakeModel([0.1, 0.9, 0.5]))

    results = reranker.rerank("query", [doc("a"), doc("b"), doc("c")])

    assert [r.id for r in results] == ["b", "c", "a"]
    assert [r.score for r in results] == [0.9, 0.5, 0.1]


def test_rerank_sends_query_and_chunk_text_pairs_to_model():
    model = FakeModel([0.2, 0.4])
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank("find invoice", [doc("a", "first chunk"), doc("b", "second chunk")])

    assert model.calls == [[("find invoice", "first chunk"), ("find invoice", "second chunk")]]


def test_rerank_preserves_text_and_metadata():
    reranker = CrossEncoderReranker(model=FakeModel([1.0]))

    results = reranker.rerank("q", [doc("a", "alpha", source="a.txt")])

    assert results[0].text == "alpha"
    assert results[0].metadata == {"source": "a.txt"}


def test_rerank_respects_limit():
    reranker = CrossEncoderReranker(model=FakeModel([0.3, 0.9, 0.6]))

    results = reranker.rerank("q", [doc("a"), doc("b"), doc("c")], limit=2)

    assert [r.id for r in results] == ["b", "c"]


def test_rerank_min_score_filters_low_scoring_chunks():
    reranker = CrossEncoderReranker(model=FakeModel([-3.0, 2.0, 0.5]))

    results = reranker.rerank("q", [doc("a"), doc("b"), doc("c")], min_score=0.5)

    assert [r.id for r in results] == ["b", "c"]


def test_rerank_breaks_ties_by_id():
    reranker = CrossEncoderReranker(model=FakeModel([1.0, 1.0]))

    results = reranker.rerank("q", [doc("b"), doc("a")])

    assert [r.id for r in results] == ["a", "b"]


def test_rerank_empty_documents_does_not_load_model():
    reranker = CrossEncoderReranker()

    assert reranker.rerank("q", []) == []
    assert reranker._model is None


def test_construction_is_lazy():
    assert CrossEncoderReranker()._model is None


def test_rerank_rejects_wrong_score_count():
    reranker = CrossEncoderReranker(model=FakeModel([0.5]))

    with pytest.raises(RerankerError, match="unexpected number"):
        reranker.rerank("q", [doc("a"), doc("b")])


def test_rerank_rejects_non_finite_scores():
    reranker = CrossEncoderReranker(model=FakeModel([float("nan")]))

    with pytest.raises(RerankerError, match="non-finite"):
        reranker.rerank("q", [doc("a")])


def test_rerank_wraps_model_failures():
    class BrokenModel:
        def predict(self, pairs):
            raise RuntimeError("boom")

    reranker = CrossEncoderReranker(model=BrokenModel())

    with pytest.raises(RerankerError, match="boom"):
        reranker.rerank("q", [doc("a")])


def test_missing_dependency_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    reranker = CrossEncoderReranker()

    with pytest.raises(RerankerError, match="sentence-transformers"):
        reranker.rerank("q", [doc("a")])


def test_invalid_arguments_are_rejected():
    reranker = CrossEncoderReranker(model=FakeModel([1.0]))

    with pytest.raises(ValueError, match="model_name"):
        CrossEncoderReranker(" ")
    with pytest.raises(ValueError, match="query"):
        reranker.rerank("  ", [doc("a")])
    with pytest.raises(ValueError, match="limit"):
        reranker.rerank("q", [doc("a")], limit=0)
