import math

import pytest

from llmkit.lexical import BM25Index, tokenize
from llmkit.vector_store import SearchResult


def doc(doc_id, text, **metadata):
    return SearchResult(id=doc_id, text=text, score=0.0, metadata=metadata)


def test_tokenize_lowercases_and_splits_identifiers():
    assert tokenize("Invoice INV-2026-101, USB-C!") == ["invoice", "inv", "2026", "101", "usb", "c"]


def test_tokenize_empty_text():
    assert tokenize("   ...  ") == []


def test_exact_identifier_ranks_matching_document_first():
    index = BM25Index(
        [
            doc("a", "payment gateway timeout error"),
            doc("b", "invoice INV-2026-101 from Northstar"),
            doc("c", "database connection is slow"),
        ]
    )

    results = index.search("INV-2026-101")

    assert results[0].id == "b"
    assert results[0].score > 0


def test_documents_without_query_terms_are_excluded():
    index = BM25Index([doc("a", "alpha beta"), doc("b", "gamma delta")])

    results = index.search("alpha")

    assert [r.id for r in results] == ["a"]


def test_unknown_terms_return_empty_list():
    index = BM25Index([doc("a", "alpha beta")])

    assert index.search("zzz") == []


def test_empty_query_and_empty_index_return_empty_list():
    assert BM25Index([doc("a", "alpha")]).search("   ") == []
    assert BM25Index([]).search("alpha") == []


def test_rarer_terms_get_higher_idf_weight():
    index = BM25Index(
        [
            doc("a", "common rare"),
            doc("b", "common filler"),
            doc("c", "common other"),
        ]
    )

    rare = index.search("rare")[0].score
    common = index.search("common")[0].score

    assert rare > common


def test_score_matches_hand_computed_bm25():
    # N=2, doc "a" = "apple banana" (len 2), doc "b" = "banana" (len 1), avg length 1.5
    index = BM25Index([doc("a", "apple banana"), doc("b", "banana")], k1=1.5, b=0.75)

    results = index.search("apple")

    idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    norm = 1 - 0.75 + 0.75 * (2 / 1.5)
    expected = idf * (1 * 2.5) / (1 + 1.5 * norm)
    assert results[0].id == "a"
    assert results[0].score == pytest.approx(expected)


def test_shorter_document_scores_higher_for_same_term_frequency():
    index = BM25Index([doc("long", "target " + "filler " * 20), doc("short", "target filler")])

    results = index.search("target")

    assert results[0].id == "short"


def test_limit_and_metadata_are_preserved():
    index = BM25Index(
        [doc("a", "alpha one", source="a.txt"), doc("b", "alpha two", source="b.txt")]
    )

    results = index.search("alpha", limit=1)

    assert len(results) == 1
    assert results[0].metadata["source"] in {"a.txt", "b.txt"}


def test_ties_are_broken_deterministically_by_id():
    index = BM25Index([doc("b", "alpha"), doc("a", "alpha")])

    assert [r.id for r in index.search("alpha")] == ["a", "b"]


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="k1"):
        BM25Index([], k1=-1)
    with pytest.raises(ValueError, match="b must"):
        BM25Index([], b=1.5)
    with pytest.raises(ValueError, match="limit"):
        BM25Index([doc("a", "alpha")]).search("alpha", limit=0)
