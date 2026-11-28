"""Tests for evaluation case validation and JSONL loading."""

import json

import pytest
from pydantic import ValidationError

from llmkit.eval_cases import EvalCase, EvalCaseError, load_eval_cases


def case_dict(**overrides):
    base = {
        "id": "inv-101-items",
        "question": "What products are listed on invoice INV-2026-101?",
        "answerable": True,
        "category": "factual",
        "evidence": ["INV-2026-101", "Widget"],
    }
    base.update(overrides)
    return base


def write_jsonl(path, rows):
    lines = [json.dumps(row) if isinstance(row, dict) else row for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_valid_case_normalizes_whitespace():
    case = EvalCase.model_validate(
        case_dict(
            id="  inv-101 ",
            question="What   is\nthe total?",
            evidence=["  Total   amount ", "tax"],
        )
    )

    assert case.id == "inv-101"
    assert case.question == "What is the total?"
    assert case.evidence == ("Total amount", "tax")


def test_unanswerable_case_without_evidence_is_valid():
    case = EvalCase.model_validate(
        case_dict(answerable=False, category="out_of_scope", evidence=[])
    )

    assert case.answerable is False
    assert case.evidence == ()


def test_answerable_case_requires_evidence():
    with pytest.raises(ValidationError, match="at least one evidence snippet"):
        EvalCase.model_validate(case_dict(evidence=[]))


def test_unanswerable_case_rejects_evidence():
    with pytest.raises(ValidationError, match="must not have evidence"):
        EvalCase.model_validate(case_dict(answerable=False, category="out_of_scope"))


def test_blank_evidence_snippet_is_rejected():
    with pytest.raises(ValidationError, match="must not be blank"):
        EvalCase.model_validate(case_dict(evidence=["Widget", "   "]))


def test_duplicate_evidence_is_rejected_case_insensitively():
    with pytest.raises(ValidationError, match="duplicates"):
        EvalCase.model_validate(case_dict(evidence=["Widget", " widget "]))


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError, match="must not be blank"):
        EvalCase.model_validate(case_dict(question="   "))


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(case_dict(category="factul"))


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        EvalCase.model_validate(case_dict(expected_answer="x"))


def test_load_eval_cases_skips_blank_lines(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        case_dict(),
        "",
        case_dict(
            id="no-answer",
            question="What is the CEO's favourite colour?",
            answerable=False,
            category="out_of_scope",
            evidence=[],
        ),
    ]
    write_jsonl(path, rows)

    cases = load_eval_cases(path)

    assert [case.id for case in cases] == ["inv-101-items", "no-answer"]


def test_load_reports_duplicate_ids_with_line_numbers(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [case_dict(), case_dict()])

    with pytest.raises(EvalCaseError, match=r":2: duplicate id 'inv-101-items'.*line 1"):
        load_eval_cases(path)


def test_load_reports_invalid_json_with_line_number(tmp_path):
    path = tmp_path / "cases.jsonl"
    write_jsonl(path, [case_dict(), "{not json"])

    with pytest.raises(EvalCaseError, match=r":2: invalid case"):
        load_eval_cases(path)


def test_load_rejects_empty_file(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(EvalCaseError, match="no evaluation cases"):
        load_eval_cases(path)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(EvalCaseError, match="cannot read"):
        load_eval_cases(tmp_path / "missing.jsonl")
