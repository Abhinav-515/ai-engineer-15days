import pytest

from llmkit.evaluation import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(
        ["a", "b", "c", "d"],
        {"a", "c"},
        k=3,
    ) == pytest.approx(2 / 3)


def test_precision_at_k_uses_k_as_denominator_when_fewer_results():
    assert precision_at_k(["a"], {"a", "b"}, k=3) == pytest.approx(1 / 3)


def test_recall_at_k():
    assert recall_at_k(
        ["a", "b", "c", "d"],
        {"a", "c", "e"},
        k=3,
    ) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_no_relevant_result():
    assert reciprocal_rank(["a", "b"], {"c"}) == 0.0


def test_mean_reciprocal_rank():
    rankings = [
        ["a", "b", "c"],
        ["x", "y", "z"],
        ["p", "q", "r"],
    ]
    relevance = [{"b"}, {"x"}, {"r"}]

    assert mean_reciprocal_rank(rankings, relevance) == pytest.approx((1 / 2 + 1 + 1 / 3) / 3)


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        precision_at_k(["a"], {"a"}, k=k)

    with pytest.raises(ValueError, match="k"):
        recall_at_k(["a"], {"a"}, k=k)


def test_metrics_reject_empty_relevance():
    with pytest.raises(ValueError, match="relevant_ids"):
        precision_at_k(["a"], set())

    with pytest.raises(ValueError, match="relevant_ids"):
        recall_at_k(["a"], set())

    with pytest.raises(ValueError, match="relevant_ids"):
        reciprocal_rank(["a"], set())


def test_mrr_rejects_mismatched_query_counts():
    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank([["a"]], [])


def test_mrr_rejects_empty_rankings():
    with pytest.raises(ValueError, match="must not be empty"):
        mean_reciprocal_rank([], [])
