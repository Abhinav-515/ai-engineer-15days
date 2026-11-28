"""Day 11 demo: run the RAG evaluation harness against an indexed collection.

Every printed result is a single run over a small dataset with a small judge model.
It is an observation, not a measurement of reliability.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path

from llmkit.client import make_client
from llmkit.config import Settings, get_settings
from llmkit.embeddings import OllamaEmbedder
from llmkit.eval_cases import EvalCase, EvalCaseError, load_eval_cases
from llmkit.eval_context import normalize_for_match
from llmkit.eval_harness import (
    CaseResult,
    EvalReport,
    MetricSummary,
    run_evaluation,
    write_report,
)
from llmkit.eval_judge import LLMJudge
from llmkit.hybrid import HybridRetriever
from llmkit.rag import RAGService
from llmkit.vector_store import QdrantVectorStore

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "samples" / "day11" / "eval_cases.jsonl"
DEFAULT_DB = ROOT / "output" / "day7" / "qdrant"
DEFAULT_COLLECTION = "day7_recursive"
DEFAULT_OUTPUT = ROOT / "output" / "day11" / "report.json"


@dataclass(frozen=True)
class JudgeConfig:
    settings: Settings
    label: str
    same_as_generator: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--judge-provider", choices=["ollama", "cloud"], default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Run deterministic metrics only (no Faithfulness or Answer Relevance).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="Continue even if some evidence snippets are not in any indexed chunk.",
    )
    return parser


def resolve_judge(provider: str, model: str | None, settings: Settings) -> JudgeConfig:
    """Build the judge's settings. The generator is always the local Ollama model."""
    if provider == "ollama":
        chosen = model or settings.ollama_model
        return JudgeConfig(
            settings.model_copy(update={"ollama_model": chosen}),
            f"ollama:{chosen}",
            same_as_generator=chosen == settings.ollama_model,
        )
    chosen = model or settings.openai_model
    return JudgeConfig(
        settings.model_copy(update={"openai_model": chosen}),
        f"cloud:{chosen}",
        same_as_generator=False,
    )


def find_unmatched_evidence(
    cases: Sequence[EvalCase], chunk_texts: Sequence[str]
) -> list[tuple[str, str]]:
    """Return (case id, snippet) pairs whose snippet is not inside any single chunk."""
    normalized = [normalize_for_match(text) for text in chunk_texts]
    return [
        (case.id, snippet)
        for case in cases
        if case.answerable
        for snippet in case.evidence
        if not any(normalize_for_match(snippet) in text for text in normalized)
    ]


def format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def format_metric(summary: MetricSummary) -> str:
    mean = "n/a" if summary.mean is None else f"{summary.mean:.3f}"
    return f"{mean} (n={summary.n}, excluded={summary.excluded})"


def format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def format_case_line(index: int, total: int, result: CaseResult) -> str:
    prefix = f"[{index}/{total}] {result.id} ({result.category})"
    if result.error is not None:
        return f"{prefix}: ERROR {result.error}"
    recall = "-" if result.context_scores is None else format_score(result.context_scores.recall)
    if result.abstained:
        return f"{prefix}: refused | recall={recall}"
    faith = "-" if result.faithfulness is None else format_score(result.faithfulness.score)
    relevance = "-" if result.relevance is None else format_score(result.relevance.score)
    return f"{prefix}: answered | recall={recall} faithfulness={faith} relevance={relevance}"


def format_report(report: EvalReport, *, generator_label: str, judge_label: str | None) -> str:
    lines = [
        "=" * 68,
        "EVALUATION REPORT",
        "=" * 68,
        f"Generator: {generator_label}",
        f"Judge:     {judge_label or 'none (--no-judge)'}",
        f"Cases:     {len(report.cases)} (generation errors: {report.generation_errors})",
        "",
        f"Context Recall:    {format_metric(report.context_recall)}",
        f"Context Precision: {format_metric(report.context_precision)}",
        f"Faithfulness:      {format_metric(report.faithfulness)}",
        f"Answer Relevance:  {format_metric(report.answer_relevance)}",
        "",
        f"False refusal rate (answerable cases):    {format_rate(report.false_refusal_rate)}",
        f"Correct refusal rate (unanswerable cases): {format_rate(report.correct_refusal_rate)}",
        "",
        "Mean latency: "
        + ("n/a" if report.mean_latency_s is None else f"{report.mean_latency_s:.2f}s"),
        f"Generation cost: ${report.total_cost_usd:.6f} | Judge cost: ${report.judge_cost_usd:.6f}",
        "",
        "NOTE: single run over a small dataset with a small judge model. These numbers are",
        "observations, not reliability measurements. Judge scores are noisy.",
    ]
    return "\n".join(lines)


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.no_judge and (args.judge_provider or args.judge_model):
        parser.error("--no-judge cannot be combined with --judge-provider or --judge-model")

    try:
        cases = load_eval_cases(args.dataset)
    except EvalCaseError as exc:
        parser.error(str(exc))

    settings = get_settings()
    generator_label = f"ollama:{settings.ollama_model}"
    judge_config = (
        None
        if args.no_judge
        else resolve_judge(args.judge_provider or "ollama", args.judge_model, settings)
    )

    print("\n=== Day 11: RAG Evaluation Harness ===")
    print(f"Dataset:    {args.dataset} ({len(cases)} cases)")
    print(f"Collection: {args.collection} | Retrieval: {args.mode} | top-k: {args.top_k}")
    print(f"Generator:  {generator_label}")
    print(f"Judge:      {judge_config.label if judge_config else 'none (--no-judge)'}")
    if judge_config is not None and judge_config.same_as_generator:
        print(
            "WARNING: the judge is the same model as the generator. A model grading its own "
            "answers tends to score them too generously."
        )

    with QdrantVectorStore(
        args.db_dir,
        collection_name=args.collection,
        vector_size=768,
    ) as store:
        documents = store.scroll_all()
        if not documents:
            raise RuntimeError(
                f"Collection '{args.collection}' is empty. Run the ingestion demo first."
            )

        unmatched = find_unmatched_evidence(cases, [document.text for document in documents])
        if unmatched:
            print(
                f"\n{len(unmatched)} evidence snippet(s) are not inside any single indexed chunk:"
            )
            for case_id, snippet in unmatched:
                print(f"  - {case_id}: {snippet!r}")
            if not args.allow_missing_evidence:
                print("Aborting: those cases would score 0 recall. Use --allow-missing-evidence.")
                raise SystemExit(1)

        print(f"\nIndexed chunks: {len(documents)}\n")

        async with AsyncExitStack() as stack:
            embedder = await stack.enter_async_context(OllamaEmbedder())
            generator = await stack.enter_async_context(make_client("ollama", settings))
            judge = None
            if judge_config is not None:
                provider = judge_config.label.split(":", 1)[0]
                judge_client = await stack.enter_async_context(
                    make_client(provider, judge_config.settings)
                )
                judge = LLMJudge(judge_client)

            service = RAGService(
                HybridRetriever(store, embedder),
                generator,
                retrieval_mode=args.mode,
                top_k=args.top_k,
            )

            counter = {"done": 0}

            def progress(result: CaseResult) -> None:
                counter["done"] += 1
                print(format_case_line(counter["done"], len(cases), result), flush=True)

            report = await run_evaluation(cases, service, judge, progress=progress)

    print()
    print(
        format_report(
            report,
            generator_label=generator_label,
            judge_label=judge_config.label if judge_config else None,
        )
    )
    path = write_report(report, args.output)
    print(f"\nPer-case JSON report: {path}")


if __name__ == "__main__":
    asyncio.run(main())
