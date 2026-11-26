import pytest

from llmkit.fusion import reciprocal_rank_fusion, weighted_score_fusion
from llmkit.vector_store import SearchResult


def r(doc_id, score=0.0, text=None, **metadata):
    return SearchResult(id=doc_id, text=text or f"text {doc_id}", score=score, metadata=metadata)


def test_rrf_scores_match_hand_computed_values():
    fused = reciprocal_rank_fusion([[r("a"), r("b")], [r("b"), r("c")]], k=60)

    by_id = {result.id: result.score for result in fused}
    assert by_id["a"] == pytest.approx(1 / 61)
    assert by_id["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"] == pytest.approx(1 / 62)


def test_rrf_ranks_document_found_by_both_lists_first():
    fused = reciprocal_rank_fusion([[r("a"), r("b")], [r("b"), r("c")]])

    assert [result.id for result in fused] == ["b", "a", "c"]


def test_rrf_deduplicates_and_preserves_text_and_metadata():
    fused = reciprocal_rank_fusion(
        [[r("a", text="alpha", source="a.txt")], [r("a", text="alpha", source="a.txt")]]
    )

    assert len(fused) == 1
    assert fused[0].text == "alpha"
    assert fused[0].metadata == {"source": "a.txt"}


def test_rrf_respects_limit():
    fused = reciprocal_rank_fusion([[r("a"), r("b"), r("c")]], limit=2)

    assert [result.id for result in fused] == ["a", "b"]


def test_rrf_breaks_ties_by_id():
    fused = reciprocal_rank_fusion([[r("b")], [r("a")]])

    assert [result.id for result in fused] == ["a", "b"]


def test_rrf_with_one_empty_list_returns_the_other():
    fused = reciprocal_rank_fusion([[], [r("a"), r("b")]])

    assert [result.id for result in fused] == ["a", "b"]


def test_rrf_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="result_lists"):
        reciprocal_rank_fusion([])
    with pytest.raises(ValueError, match="k must"):
        reciprocal_rank_fusion([[r("a")]], k=0)
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion([[r("a")]], limit=0)


def test_weighted_fusion_normalises_and_weights_scores():
    dense = [r("a", 0.9), r("b", 0.5)]  # normalised: a=1.0, b=0.0
    lexical = [r("b", 10.0), r("c", 5.0)]  # normalised: b=1.0, c=0.0

    fused = weighted_score_fusion(dense, lexical, alpha=0.5)

    by_id = {result.id: result.score for result in fused}
    assert by_id["a"] == pytest.approx(0.5)
    assert by_id["b"] == pytest.approx(0.5)
    assert by_id["c"] == pytest.approx(0.0)


def test_weighted_fusion_alpha_one_equals_dense_order():
    dense = [r("a", 0.9), r("b", 0.5)]
    lexical = [r("b", 10.0), r("a", 1.0)]

    fused = weighted_score_fusion(dense, lexical, alpha=1.0)

    assert [result.id for result in fused] == ["a", "b"]


def test_weighted_fusion_alpha_zero_equals_lexical_order():
    dense = [r("a", 0.9), r("b", 0.5)]
    lexical = [r("b", 10.0), r("a", 1.0)]

    fused = weighted_score_fusion(dense, lexical, alpha=0.0)

    assert [result.id for result in fused] == ["b", "a"]


def test_weighted_fusion_handles_identical_scores_and_empty_side():
    fused = weighted_score_fusion([r("a", 0.7), r("b", 0.7)], [], alpha=0.5)

    assert [result.score for result in fused] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert [result.id for result in fused] == ["a", "b"]


def test_weighted_fusion_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="alpha"):
        weighted_score_fusion([], [], alpha=1.5)
    with pytest.raises(ValueError, match="limit"):
        weighted_score_fusion([], [], limit=0)
