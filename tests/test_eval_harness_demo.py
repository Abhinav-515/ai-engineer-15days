"""Tests for the Day 11 evaluation CLI helpers and argument validation (no live models)."""

import sys

import pytest

from llmkit.config import Settings
from llmkit.eval_cases import EvalCase
from llmkit.eval_context import RetrievedChunk, score_context
from llmkit.eval_harness import CaseResult, MetricSummary, build_report
from llmkit.eval_harness_demo import (
    find_unmatched_evidence,
    format_case_line,
    format_metric,
    format_rate,
    format_report,
    main,
    resolve_judge,
)
from llmkit.eval_judge import FaithfulnessResult, RelevanceResult


def make_settings():
    return Settings(_env_file=None, ollama_model="gen-model", openai_model="cloud-model")


def answerable(case_id="a", evidence=("Acme",)):
    return EvalCase(
        id=case_id,
        question="Q?",
        answerable=True,
        category="factual",
        evidence=evidence,
    )


def test_find_unmatched_evidence_reports_missing_snippets():
    cases = [answerable("a", ("Acme", "Gizmo"))]

    assert find_unmatched_evidence(cases, ["Vendor: Acme"]) == [("a", "Gizmo")]


def test_find_unmatched_evidence_ignores_unanswerable_cases():
    case = EvalCase(id="u", question="Q?", answerable=False, category="out_of_scope")

    assert find_unmatched_evidence([case], ["anything"]) == []


def test_find_unmatched_evidence_matches_case_and_whitespace():
    cases = [answerable("a", ("total amount",))]

    assert find_unmatched_evidence(cases, ["TOTAL\n   AMOUNT: 5"]) == []


def test_find_unmatched_evidence_requires_snippet_inside_a_single_chunk():
    cases = [answerable("a", ("alpha beta",))]

    assert find_unmatched_evidence(cases, ["alpha", "beta"]) == [("a", "alpha beta")]


def test_format_metric_with_mean():
    assert format_metric(MetricSummary(0.75, 4, 1)) == "0.750 (n=4, excluded=1)"


def test_format_metric_without_mean():
    assert format_metric(MetricSummary(None, 0, 2)) == "n/a (n=0, excluded=2)"


def test_format_rate():
    assert format_rate(0.5) == "50%"
    assert format_rate(None) == "n/a"


def test_case_line_for_error():
    result = CaseResult("c1", "factual", True, None, error="generation_error: RAGError: bad")

    assert format_case_line(1, 8, result) == (
        "[1/8] c1 (factual): ERROR generation_error: RAGError: bad"
    )


def test_case_line_for_refusal():
    scores = score_context([RetrievedChunk("x", "Acme")], ["Acme"])
    result = CaseResult("c1", "factual", True, True, context_scores=scores)

    assert format_case_line(2, 8, result) == "[2/8] c1 (factual): refused | recall=1.00"


def test_case_line_for_answer_with_judge_scores():
    scores = score_context([RetrievedChunk("x", "Acme")], ["Acme"])
    result = CaseResult(
        "c1",
        "factual",
        True,
        False,
        context_scores=scores,
        faithfulness=FaithfulnessResult(1.0, 1, 1, ()),
        relevance=RelevanceResult(0.5, "partial"),
    )

    line = format_case_line(3, 8, result)

    assert "answered" in line
    assert "faithfulness=1.00" in line
    assert "relevance=0.50" in line


def test_format_report_contains_metrics_and_single_run_caveat():
    report = build_report([CaseResult("c1", "out_of_scope", False, True)])

    text = format_report(report, generator_label="ollama:gen", judge_label="ollama:judge")

    assert "Generator: ollama:gen" in text
    assert "Judge:     ollama:judge" in text
    assert "Context Recall:    n/a (n=0, excluded=0)" in text
    assert "Correct refusal rate (unanswerable cases): 100%" in text
    assert "single run" in text
    assert "not reliability measurements" in text


def test_format_report_without_judge_says_so():
    report = build_report([CaseResult("c1", "out_of_scope", False, True)])

    text = format_report(report, generator_label="ollama:gen", judge_label=None)

    assert "none (--no-judge)" in text


def test_resolve_judge_defaults_to_generator_model_and_flags_it():
    config = resolve_judge("ollama", None, make_settings())

    assert config.label == "ollama:gen-model"
    assert config.same_as_generator is True
    assert config.settings.ollama_model == "gen-model"


def test_resolve_judge_override_changes_only_the_judge_settings():
    settings = make_settings()

    config = resolve_judge("ollama", "other-model", settings)

    assert config.label == "ollama:other-model"
    assert config.same_as_generator is False
    assert config.settings.ollama_model == "other-model"
    assert settings.ollama_model == "gen-model"


def test_resolve_judge_cloud_uses_cloud_model():
    config = resolve_judge("cloud", None, make_settings())

    assert config.label == "cloud:cloud-model"
    assert config.same_as_generator is False


async def test_main_rejects_non_positive_top_k(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eval_harness_demo", "--top-k", "0"])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--top-k must be positive" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [["--judge-model", "x"], ["--judge-provider", "cloud"]],
)
async def test_main_rejects_no_judge_with_judge_options(monkeypatch, capsys, extra):
    monkeypatch.setattr(sys, "argv", ["eval_harness_demo", "--no-judge", *extra])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "--no-judge cannot be combined" in capsys.readouterr().err


async def test_main_rejects_missing_dataset(monkeypatch, capsys, tmp_path):
    missing = tmp_path / "missing.jsonl"
    monkeypatch.setattr(sys, "argv", ["eval_harness_demo", "--dataset", str(missing)])

    with pytest.raises(SystemExit) as exc_info:
        await main()

    assert exc_info.value.code == 2
    assert "cannot read" in capsys.readouterr().err
