"""Evaluation runner: scores RAG responses case by case and aggregates the metrics.

Generation errors and judge failures are recorded and excluded from the metrics they
affect. They are never converted into zero scores.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from llmkit.client import LLMError
from llmkit.eval_cases import EvalCase
from llmkit.eval_context import ContextScores, RetrievedChunk, score_context
from llmkit.eval_judge import FaithfulnessResult, LLMJudge, RelevanceResult
from llmkit.rag import RAGError, RAGResponse


class Answerer(Protocol):
    """Anything that answers a question with a RAGResponse, such as RAGService."""

    async def ask(self, question: str) -> RAGResponse: ...


@dataclass(frozen=True)
class CaseResult:
    """Everything recorded for one evaluation case."""

    id: str
    category: str
    answerable: bool
    abstained: bool | None
    error: str | None = None
    answer: str = ""
    retrieved_ids: tuple[str, ...] = ()
    context_scores: ContextScores | None = None
    faithfulness: FaithfulnessResult | None = None
    relevance: RelevanceResult | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    context: str = ""


@dataclass(frozen=True)
class MetricSummary:
    """Mean over scored cases, how many were scored, and how many were excluded."""

    mean: float | None
    n: int
    excluded: int


@dataclass(frozen=True)
class EvalReport:
    cases: tuple[CaseResult, ...]
    context_recall: MetricSummary
    context_precision: MetricSummary
    faithfulness: MetricSummary
    answer_relevance: MetricSummary
    false_refusal_rate: float | None
    correct_refusal_rate: float | None
    generation_errors: int
    total_cost_usd: float
    judge_cost_usd: float
    mean_latency_s: float | None


def summarize(values: Sequence[float | None]) -> MetricSummary:
    """Average the scored values; None values are counted as excluded, not as zero."""
    scored = [value for value in values if value is not None]
    mean = sum(scored) / len(scored) if scored else None
    return MetricSummary(mean=mean, n=len(scored), excluded=len(values) - len(scored))


async def evaluate_case(
    case: EvalCase,
    rag: Answerer,
    judge: LLMJudge | None = None,
) -> CaseResult:
    """Run one case through the RAG service and score the response."""
    try:
        response = await rag.ask(case.question)
    except (RAGError, LLMError, httpx.HTTPError) as exc:
        detail = " ".join(str(exc).split())[:200]
        return CaseResult(
            id=case.id,
            category=case.category,
            answerable=case.answerable,
            abstained=None,
            error=f"generation_error: {type(exc).__name__}: {detail}",
        )

    chunks = [RetrievedChunk(item.id, item.text) for item in response.retrieved]
    context_scores = score_context(chunks, case.evidence) if case.answerable else None

    faithfulness: FaithfulnessResult | None = None
    relevance: RelevanceResult | None = None
    judge_cost = 0.0
    if judge is not None and not response.abstained:
        faithfulness = await judge.faithfulness(response.answer, response.context)
        relevance = await judge.answer_relevance(case.question, response.answer)
        judge_cost = faithfulness.cost_usd + relevance.cost_usd

    return CaseResult(
        id=case.id,
        category=case.category,
        answerable=case.answerable,
        abstained=response.abstained,
        answer=response.answer,
        retrieved_ids=tuple(chunk.id for chunk in chunks),
        context_scores=context_scores,
        faithfulness=faithfulness,
        relevance=relevance,
        latency_s=response.latency_s,
        cost_usd=response.cost_usd,
        judge_cost_usd=judge_cost,
        context=response.context,
    )


def build_report(results: Sequence[CaseResult]) -> EvalReport:
    """Aggregate per-case results into an EvalReport."""
    ok = [result for result in results if result.error is None]
    contexts = [r.context_scores for r in ok if r.context_scores is not None]
    answerable_ok = [r for r in ok if r.answerable]
    unanswerable_ok = [r for r in ok if not r.answerable]

    return EvalReport(
        cases=tuple(results),
        context_recall=summarize([c.recall for c in contexts]),
        context_precision=summarize([c.precision for c in contexts]),
        faithfulness=summarize([r.faithfulness.score for r in ok if r.faithfulness is not None]),
        answer_relevance=summarize([r.relevance.score for r in ok if r.relevance is not None]),
        false_refusal_rate=(
            sum(bool(r.abstained) for r in answerable_ok) / len(answerable_ok)
            if answerable_ok
            else None
        ),
        correct_refusal_rate=(
            sum(bool(r.abstained) for r in unanswerable_ok) / len(unanswerable_ok)
            if unanswerable_ok
            else None
        ),
        generation_errors=len(results) - len(ok),
        total_cost_usd=sum(r.cost_usd for r in ok),
        judge_cost_usd=sum(r.judge_cost_usd for r in ok),
        mean_latency_s=sum(r.latency_s for r in ok) / len(ok) if ok else None,
    )


async def run_evaluation(
    cases: Sequence[EvalCase],
    rag: Answerer,
    judge: LLMJudge | None = None,
    *,
    progress: Callable[[CaseResult], None] | None = None,
) -> EvalReport:
    """Evaluate every case sequentially and return the aggregated report."""
    if not cases:
        raise ValueError("cases must not be empty")

    results: list[CaseResult] = []
    for case in cases:
        result = await evaluate_case(case, rag, judge)
        results.append(result)
        if progress is not None:
            progress(result)
    return build_report(results)


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return asdict(report)


def write_report(report: EvalReport, path: str | Path) -> Path:
    """Write the report as JSON, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_to_dict(report), indent=2) + "\n", encoding="utf-8")
    return output
