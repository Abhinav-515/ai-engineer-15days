"""Tests for evidence-based Context Recall and Context Precision."""

import pytest

from llmkit.eval_context import (
    RetrievedChunk,
    context_precision,
    context_recall,
    score_context,
)

C1 = RetrievedChunk("a", "Invoice INV-2026-101 lists Widget x2")
C2 = RetrievedChunk("b", "Payment gateway timeout at 10:42")
C3 = RetrievedChunk("c", "Total amount: USD 15.00")


def test_recall_counts_snippets_across_different_chunks():
    assert context_recall([C1, C2, C3], ["INV-2026-101", "Total amount"]) == 1.0


def test_recall_is_partial_when_a_snippet_is_missing():
    assert context_recall([C1, C2, C3], ["INV-2026-101", "Gizmo"]) == 0.5


def test_matching_ignores_case_and_whitespace():
    chunk = RetrievedChunk("x", "Total\n  AMOUNT:   USD 15.00")

    assert context_recall([chunk], ["total amount"]) == 1.0


def test_no_chunks_gives_zero_recall_and_undefined_precision():
    assert context_recall([], ["Widget"]) == 0.0
    assert context_precision([], ["Widget"]) is None


def test_precision_is_share_of_chunks_containing_evidence():
    assert context_precision([C1, C2, C3], ["INV-2026-101"]) == pytest.approx(1 / 3)
    assert context_precision([C1, C2, C3], ["INV-2026-101", "Total amount"]) == pytest.approx(2 / 3)


def test_precision_is_zero_when_no_chunk_is_relevant():
    assert context_precision([C2], ["Widget"]) == 0.0


def test_duplicate_evidence_is_counted_once():
    scores = score_context([C1], ["Widget", " widget "])

    assert scores.recall == 1.0
    assert scores.found_snippets == ("Widget",)


def test_empty_evidence_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        context_recall([C1], [])


def test_blank_evidence_snippet_is_rejected():
    with pytest.raises(ValueError, match="must not be blank"):
        context_recall([C1], ["Widget", "  "])


def test_score_context_reports_details():
    scores = score_context([C1, C2, C3], ["INV-2026-101", "Gizmo"])

    assert scores.recall == 0.5
    assert scores.precision == pytest.approx(1 / 3)
    assert scores.found_snippets == ("INV-2026-101",)
    assert scores.missing_snippets == ("Gizmo",)
    assert scores.relevant_chunk_ids == ("a",)
    assert scores.retrieved_count == 3
