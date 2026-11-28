"""Tests for the evaluation runner and aggregation (mocked RAG service and judge)."""

import json

import pytest

from llmkit.eval_cases import EvalCase
from llmkit.eval_harness import (
    evaluate_case,
    run_evaluation,
    summarize,
    write_report,
)
from llmkit.eval_judge import FaithfulnessResult, RelevanceResult
from llmkit.rag import RAGError, RAGResponse
from llmkit.vector_store import SearchResult


class FakeRag:
    def __init__(self, responses):
        self.responses = responses
        self.questions = []

    async def ask(self, question):
        self.questions.append(question)
        item = self.responses[question]
        if isinstance(item, Exception):
            raise item
        return item


class FakeJudge:
    def __init__(self, faithfulness=None, relevance=None):
        self.faithfulness_result = faithfulness or FaithfulnessResult(1.0, 1, 1, (), cost_usd=0.002)
        self.relevance_result = relevance or RelevanceResult(1.0, "full", "ok", cost_usd=0.001)
        self.faithfulness_calls = []
        self.relevance_calls = []

    async def faithfulness(self, answer, context):
        self.faithfulness_calls.append((answer, context))
        return self.faithfulness_result

    async def answer_relevance(self, question, answer):
        self.relevance_calls.append((question, answer))
        return self.relevance_result


def chunk(chunk_id, text):
    return SearchResult(id=chunk_id, text=text, score=0.5, metadata={})


def response(
    answer="The vendor is Acme [S1].",
    *,
    abstained=False,
    retrieved=(),
    context="CTX",
    latency_s=0.5,
    cost_usd=0.01,
):
    return RAGResponse(
        answer=answer,
        citations=[],
        abstained=abstained,
        retrieved_count=len(retrieved),
        latency_s=latency_s,
        cost_usd=cost_usd,
        context=context,
        retrieved=tuple(retrieved),
    )


def answerable(case_id="a", question="Q-a?", evidence=("Acme",)):
    return EvalCase(
        id=case_id,
        question=question,
        answerable=True,
        category="factual",
        evidence=evidence,
    )


def unanswerable(case_id="u", question="Q-u?"):
    return EvalCase(
        id=case_id,
        question=question,
        answerable=False,
        category="out_of_scope",
    )


def test_summarize_excludes_none_values():
    summary = summarize([1.0, None, 0.0])

    assert (summary.mean, summary.n, summary.excluded) == (0.5, 2, 1)


def test_summarize_all_none_has_no_mean():
    summary = summarize([None, None])

    assert (summary.mean, summary.n, summary.excluded) == (None, 0, 2)


async def test_answered_case_is_fully_scored():
    case = answerable()
    rag = FakeRag(
        {case.question: response(retrieved=[chunk("c1", "Vendor: Acme"), chunk("c2", "other")])}
    )
    judge = FakeJudge()

    result = await evaluate_case(case, rag, judge)

    assert result.error is None
    assert result.abstained is False
    assert result.context_scores.recall == 1.0
    assert result.context_scores.precision == 0.5
    assert result.retrieved_ids == ("c1", "c2")
    assert judge.faithfulness_calls == [("The vendor is Acme [S1].", "CTX")]
    assert judge.relevance_calls == [(case.question, "The vendor is Acme [S1].")]
    assert result.faithfulness.score == 1.0
    assert result.judge_cost_usd == pytest.approx(0.003)
    assert result.cost_usd == 0.01
    assert result.latency_s == 0.5


async def test_abstained_answerable_case_is_a_false_refusal_and_skips_judge():
    case = answerable()
    rag = FakeRag(
        {
            case.question: response(
                "No info.", abstained=True, retrieved=[chunk("c1", "Vendor: Acme")], context=""
            )
        }
    )
    judge = FakeJudge()

    report = await run_evaluation([case], rag, judge)

    result = report.cases[0]
    assert result.abstained is True
    assert result.faithfulness is None
    assert result.relevance is None
    assert judge.faithfulness_calls == []
    assert judge.relevance_calls == []
    assert result.context_scores.recall == 1.0
    assert report.false_refusal_rate == 1.0
    assert report.faithfulness.n == 0


async def test_unanswerable_case_abstained_is_a_correct_refusal():
    case = unanswerable()
    rag = FakeRag({case.question: response("No info.", abstained=True, context="")})

    report = await run_evaluation([case], rag, FakeJudge())

    assert report.cases[0].context_scores is None
    assert report.correct_refusal_rate == 1.0
    assert report.false_refusal_rate is None
    assert report.context_recall.n == 0


async def test_unanswerable_case_answered_is_judged_and_not_a_correct_refusal():
    case = unanswerable()
    rag = FakeRag({case.question: response(retrieved=[chunk("c1", "x")])})

    report = await run_evaluation([case], rag, FakeJudge())

    assert report.cases[0].context_scores is None
    assert report.cases[0].faithfulness is not None
    assert report.correct_refusal_rate == 0.0
    assert report.context_recall.n == 0


async def test_generation_error_is_recorded_and_excluded_everywhere():
    case = answerable()
    rag = FakeRag({case.question: RAGError("bad json")})

    report = await run_evaluation([case], rag, FakeJudge())

    result = report.cases[0]
    assert result.error.startswith("generation_error: RAGError")
    assert result.abstained is None
    assert report.generation_errors == 1
    assert report.context_recall.n == 0
    assert report.context_recall.excluded == 0
    assert report.false_refusal_rate is None
    assert report.mean_latency_s is None


async def test_judge_failure_is_excluded_not_zero():
    case = answerable()
    rag = FakeRag({case.question: response(retrieved=[chunk("c1", "Vendor: Acme")])})
    failed = FaithfulnessResult(None, 0, 1, (), error="sentence 1: invalid_output")

    report = await run_evaluation([case], rag, FakeJudge(faithfulness=failed))

    assert report.faithfulness.mean is None
    assert report.faithfulness.n == 0
    assert report.faithfulness.excluded == 1
    assert report.answer_relevance.n == 1


async def test_without_judge_only_deterministic_metrics_run():
    case = answerable()
    rag = FakeRag({case.question: response(retrieved=[chunk("c1", "Vendor: Acme")])})

    report = await run_evaluation([case], rag)

    assert report.cases[0].faithfulness is None
    assert report.faithfulness.n == 0
    assert report.faithfulness.excluded == 0
    assert report.context_recall.mean == 1.0
    assert report.judge_cost_usd == 0.0


async def test_nothing_retrieved_gives_zero_recall_and_excluded_precision():
    case = answerable()
    rag = FakeRag({case.question: response("No info.", abstained=True, context="")})

    report = await run_evaluation([case], rag, FakeJudge())

    assert report.context_recall.mean == 0.0
    assert report.context_recall.n == 1
    assert report.context_precision.n == 0
    assert report.context_precision.excluded == 1


async def test_report_aggregates_metrics_and_rates():
    a1, a2 = answerable("a1", "Q-a1?"), answerable("a2", "Q-a2?")
    u1, u2 = unanswerable("u1", "Q-u1?"), unanswerable("u2", "Q-u2?")
    rag = FakeRag(
        {
            a1.question: response(retrieved=[chunk("c1", "Vendor: Acme")]),
            a2.question: response(
                "No info.", abstained=True, retrieved=[chunk("c2", "other")], context=""
            ),
            u1.question: response("No info.", abstained=True, context=""),
            u2.question: response("It is blue [S1].", retrieved=[chunk("c3", "blue")]),
        }
    )

    report = await run_evaluation([a1, a2, u1, u2], rag, FakeJudge())

    assert (report.context_recall.mean, report.context_recall.n) == (0.5, 2)
    assert (report.context_precision.mean, report.context_precision.n) == (0.5, 2)
    assert (report.faithfulness.mean, report.faithfulness.n) == (1.0, 2)
    assert (report.answer_relevance.mean, report.answer_relevance.n) == (1.0, 2)
    assert report.false_refusal_rate == 0.5
    assert report.correct_refusal_rate == 0.5
    assert report.generation_errors == 0
    assert report.total_cost_usd == pytest.approx(0.04)
    assert report.judge_cost_usd == pytest.approx(0.006)
    assert report.mean_latency_s == pytest.approx(0.5)


async def test_empty_cases_are_rejected():
    with pytest.raises(ValueError, match="cases must not be empty"):
        await run_evaluation([], FakeRag({}))


async def test_progress_callback_receives_each_result():
    cases = [answerable("a1", "Q1?"), answerable("a2", "Q2?")]
    rag = FakeRag({case.question: response(retrieved=[chunk("c", "Acme")]) for case in cases})
    seen = []

    await run_evaluation(cases, rag, progress=lambda result: seen.append(result.id))

    assert seen == ["a1", "a2"]


async def test_write_report_creates_parent_directories_and_valid_json(tmp_path):
    case = answerable()
    rag = FakeRag({case.question: response(retrieved=[chunk("c1", "Vendor: Acme")])})
    report = await run_evaluation([case], rag, FakeJudge())

    path = write_report(report, tmp_path / "nested" / "report.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["context_recall"]["n"] == 1
    assert data["cases"][0]["id"] == "a"
    assert data["cases"][0]["context_scores"]["recall"] == 1.0
