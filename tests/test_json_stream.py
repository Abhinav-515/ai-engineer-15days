"""Tests for incremental extraction of the JSON answer field."""

from __future__ import annotations

import json

import pytest

from llmkit.json_stream import AnswerFieldExtractor

ANSWERS = [
    "The total is USD 110.00 [S1].",
    'He said "hi"\nsecond line \\ path [S1]',
    "Caf\u00e9 \u20b9500 \U0001f600 done [S2]",
    'The word answer appears here: "answer": "x" [S1]',
]


def _payloads(answer: str) -> list[str]:
    body = {"answer": answer, "citations": ["[S1]"], "abstain": False}
    reordered = {"citations": ["[S1]"], "abstain": False, "answer": answer}
    return [
        json.dumps(body),
        json.dumps(body, ensure_ascii=False),
        json.dumps(body, indent=2),
        json.dumps(reordered),
    ]


def _run(payload: str, size: int) -> tuple[str, AnswerFieldExtractor]:
    extractor = AnswerFieldExtractor()
    parts = [extractor.feed(payload[i : i + size]) for i in range(0, len(payload), size)]
    return "".join(parts), extractor


@pytest.mark.parametrize("answer", ANSWERS)
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 1000])
def test_extracts_answer_for_any_chunking(answer: str, size: int) -> None:
    for payload in _payloads(answer):
        text, extractor = _run(payload, size)

        assert text == answer
        assert extractor.done


def test_emits_text_incrementally() -> None:
    extractor = AnswerFieldExtractor()

    assert extractor.feed('{"answer": "Hel') == "Hel"
    assert extractor.feed("lo wor") == "lo wor"
    assert extractor.feed('ld", "citations": []}') == "ld"
    assert extractor.done


def test_nothing_emitted_before_answer_key() -> None:
    extractor = AnswerFieldExtractor()

    assert extractor.feed('{"citations": ["[S1]"], ') == ""
    assert extractor.feed('"abstain": false, "ans') == ""
    assert not extractor.done


def test_output_after_closing_quote_is_ignored() -> None:
    extractor = AnswerFieldExtractor()

    assert extractor.feed('{"answer": "done"}') == "done"
    assert extractor.feed('"answer": "second"') == ""


def test_escape_split_across_chunks() -> None:
    extractor = AnswerFieldExtractor()
    pieces = ['{"answer": "a\\', "nb \\u00", 'e9 end"}']

    assert "".join(extractor.feed(p) for p in pieces) == "a\nb \u00e9 end"


def test_lone_surrogate_becomes_replacement_char() -> None:
    extractor = AnswerFieldExtractor()

    text = extractor.feed('{"answer": "bad \\ud83d end"}')

    assert text == "bad \ufffd end"
    text.encode("utf-8")


def test_lone_low_surrogate_becomes_replacement_char() -> None:
    extractor = AnswerFieldExtractor()

    assert extractor.feed('{"answer": "x\\udc00y"}') == "x\ufffdy"


def test_no_answer_key_yields_nothing() -> None:
    extractor = AnswerFieldExtractor()

    assert extractor.feed('{"other": "value"}') == ""
    assert not extractor.done
