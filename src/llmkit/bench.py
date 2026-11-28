"""Benchmark harness: run a task set against each provider, aggregate, write a report."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llmkit.bench_tasks import TASKS, Task
from llmkit.client import BaseLLMClient, LLMError, make_client

PROVIDER_ERRORS = (LLMError, httpx.HTTPError)
MIN_TPS_TOKENS = 80  # shorter answers give a decode window too small to measure
RETRY_DELAYS_S = (3, 5, 10, 20)  # waits between retries on HTTP 429


@dataclass
class RunRecord:
    provider: str
    model: str
    task_id: str
    category: str
    run: int
    ok: bool
    error: str | None = None
    text: str = ""
    passed: bool = False
    judge_score: int | None = None
    ttft_s: float | None = None
    ttft_any_s: float | None = None
    total_s: float | None = None
    tokens_per_s: float | None = None
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


# ---------------------------------------------------------------- statistics


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. With few samples, p95 equals the maximum."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def summarize(records: list[RunRecord]) -> dict[str, Any]:
    ok = [r for r in records if r.ok]

    def col(attr: str) -> list[float]:
        return [v for r in ok if (v := getattr(r, attr)) is not None]

    ttft, total = col("ttft_s"), col("total_s")
    tps = [
        r.tokens_per_s
        for r in ok
        if r.tokens_per_s is not None and r.completion_tokens >= MIN_TPS_TOKENS
    ]
    judged = [r.judge_score for r in ok if r.judge_score is not None]
    completion = sum(r.completion_tokens for r in ok)
    reasoning = sum(r.reasoning_tokens for r in ok)
    return {
        "runs": len(records),
        "errors": len(records) - len(ok),
        "pass_rate": sum(r.passed for r in records) / len(records) if records else 0.0,
        "judge_mean": _mean(judged),
        "ttft_mean": _mean(ttft),
        "ttft_p95": percentile(ttft, 0.95) if ttft else None,
        "total_mean": _mean(total),
        "tps_mean": _mean(tps),
        "reasoning_share": reasoning / completion if completion else 0.0,
        "cost_per_1k": 1000 * (_mean([r.cost_usd for r in ok]) or 0.0),
    }


# -------------------------------------------------------------------- running


async def generate_with_retry(client: BaseLLMClient, messages: Any, **kwargs: Any) -> Any:
    """Retry only on HTTP 429 (rate limit). Timing is measured inside each attempt."""
    for delay in (*RETRY_DELAYS_S, None):
        try:
            return await client.generate(messages, **kwargs)
        except LLMError as e:
            if "HTTP 429" not in str(e) or delay is None:
                raise
            await asyncio.sleep(delay)


async def judge_answer(judge_client: BaseLLMClient, task: Task, answer: str) -> int | None:
    """Ask an LLM to score an answer 1-5. Returns None if the judge fails or is unparseable."""
    prompt = (
        f"Task given to a model:\n{task.prompt}\n\nThe model's answer:\n{answer}\n\n"
        "Score the answer from 1 (wrong or unusable) to 5 (correct, clear, follows every "
        'instruction). Reply with only JSON: {"score": <int>, "reason": "<short>"}'
    )
    try:
        r = await generate_with_retry(
            judge_client, [{"role": "user", "content": prompt}], max_tokens=800
        )
    except PROVIDER_ERRORS:
        return None
    match = re.search(r"\{.*?\}", r.text, re.DOTALL)
    if not match:
        return None
    try:
        score = int(json.loads(match.group())["score"])
    except (ValueError, KeyError, TypeError):
        return None
    return score if 1 <= score <= 5 else None


async def run_once(
    client: BaseLLMClient, task: Task, run_idx: int, judge_client: BaseLLMClient | None
) -> RunRecord:
    base = {
        "provider": client.provider,
        "model": client.model,
        "task_id": task.id,
        "category": task.category,
        "run": run_idx,
    }
    try:
        r = await generate_with_retry(
            client, [{"role": "user", "content": task.prompt}], max_tokens=task.max_tokens
        )
    except PROVIDER_ERRORS as e:
        return RunRecord(**base, ok=False, error=str(e)[:300])

    try:
        passed = bool(task.check(r.text))
    except Exception:
        passed = False
    score = await judge_answer(judge_client, task, r.text) if judge_client else None
    return RunRecord(
        **base,
        ok=True,
        text=r.text,
        passed=passed,
        judge_score=score,
        ttft_s=r.ttft_s,
        ttft_any_s=r.ttft_any_s,
        total_s=r.total_s,
        tokens_per_s=r.tokens_per_s,
        completion_tokens=r.usage.completion_tokens,
        reasoning_tokens=r.usage.reasoning_tokens,
        cost_usd=r.cost_usd,
    )


async def bench_provider(
    provider: str, tasks: list[Task], runs: int, judge_client: BaseLLMClient | None
) -> list[RunRecord]:
    records: list[RunRecord] = []
    async with make_client(provider) as client:
        # Warm-up: model load / connection setup must not be counted.
        await client.generate([{"role": "user", "content": "Hi"}], max_tokens=16)
        # Sequential on purpose: concurrent requests would distort latency numbers.
        for task in tasks:
            for i in range(runs):
                rec = await run_once(client, task, i, judge_client if i == 0 else None)
                records.append(rec)
                status = f"pass={rec.passed}" if rec.ok else f"ERROR {rec.error}"
                total = f"{rec.total_s:.2f}s" if rec.total_s is not None else "-"
                print(f"  [{provider}] {task.id:<8} run {i}: {status} total={total}")
    return records


# --------------------------------------------------------------------- report


def _f(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


_ROWS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("Runs (errors)", lambda s: f"{s['runs']} ({s['errors']})"),
    ("Auto-check pass rate", lambda s: f"{s['pass_rate']:.0%}"),
    ("LLM-judge score, mean (1-5)", lambda s: _f(s["judge_mean"], 2)),
    ("TTFT visible, mean (s)", lambda s: _f(s["ttft_mean"], 3)),
    ("TTFT visible, p95 (s)", lambda s: _f(s["ttft_p95"], 3)),
    ("Total latency, mean (s)", lambda s: _f(s["total_mean"], 2)),
    (
        f"Decode tokens/sec, mean (answers >= {MIN_TPS_TOKENS} tokens)",
        lambda s: _f(s["tps_mean"], 1),
    ),
    ("Reasoning tokens (% of output)", lambda s: f"{s['reasoning_share']:.0%}"),
    ("Cost per 1,000 requests (USD)", lambda s: f"${s['cost_per_1k']:.4f}"),
]


def build_report(records: list[RunRecord], meta: dict[str, Any]) -> str:
    providers = list(dict.fromkeys(r.provider for r in records))
    models = {p: next(r.model for r in records if r.provider == p) for p in providers}
    stats = {p: summarize([r for r in records if r.provider == p]) for p in providers}
    cols = " | ".join(f"{p} ({models[p]})" for p in providers)
    divider = "|---|" + "---|" * len(providers)

    out = [
        "# Local vs Cloud LLM Benchmark",
        "",
        f"- Date: {meta['date']}",
        f"- Local hardware: {meta['hardware']}",
        f"- Runs per task: {meta['runs']} (after one warm-up call per provider)",
        f"- Tasks: {len({r.task_id for r in records})} across several categories",
        "",
        "## Results",
        "",
        f"| Metric | {cols} |",
        divider,
    ]
    for label, fn in _ROWS:
        out.append(f"| {label} | " + " | ".join(fn(stats[p]) for p in providers) + " |")

    out += ["", "## Per-task auto-check results (passed / runs)", ""]
    out += [f"| Task | {' | '.join(providers)} |", divider]
    for tid in dict.fromkeys(r.task_id for r in records):
        cells = []
        for p in providers:
            rs = [r for r in records if r.provider == p and r.task_id == tid]
            cells.append(f"{sum(r.passed for r in rs)}/{len(rs)}")
        out.append(f"| {tid} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## Methodology",
        "",
        "- Requests run sequentially so providers do not compete for resources.",
        "- TTFT is measured to the first visible answer token (reasoning tokens excluded).",
        "- Decode tokens/sec = (completion_tokens - 1) / (total time - time to first token),",
        f"  averaged only over answers with >= {MIN_TPS_TOKENS} tokens "
        "(shorter ones are too brief to time).",
        "- HTTP 429 rate-limit responses are retried with backoff; other errors count as failures.",
        "- Cost = token usage x configured per-million prices; local inference is $0.",
        "- Quality: automatic pass/fail checks on every run (primary), plus an LLM judge",
        "  scoring the first run of each task (secondary).",
        "",
        "## Caveats",
        "",
        "- Few runs per task: p95 is effectively the maximum and not statistically strong.",
        "- Models differ in size (3B local vs 20B cloud), so this compares deployments,",
        "  not architectures.",
        "- The cloud model acts as judge, which can favor its own style.",
        "- Cloud latency includes network distance from the test location.",
        "- Reasoning tokens are billed as output tokens.",
        "",
        "## Findings",
        "",
        "_To be written after reviewing the numbers._",
        "",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------------ CLI


def save_outputs(records: list[RunRecord], args: argparse.Namespace) -> tuple[Path, Path, str]:
    """Write raw JSON and the Markdown report. Sync on purpose: plain file I/O."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    raw_dir = Path("benchmarks/results")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"run-{stamp}.json"
    raw_path.write_text(json.dumps([asdict(r) for r in records], indent=2))

    meta = {"date": stamp[:8], "runs": args.runs, "hardware": args.hardware}
    report = build_report(records, meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    return raw_path, out_path, report


async def amain(args: argparse.Namespace) -> None:
    judge_client = None if args.no_judge else make_client("cloud")
    records: list[RunRecord] = []
    try:
        for provider in args.providers:
            print(f"\nBenchmarking {provider} ...")
            try:
                records += await bench_provider(provider, TASKS, args.runs, judge_client)
            except PROVIDER_ERRORS as e:
                print(f"[{provider}] skipped: {e}")
    finally:
        if judge_client:
            await judge_client.aclose()

    if not records:
        print("No results collected.")
        return

    raw_path, out_path, report = save_outputs(records, args)
    print(f"\nRaw data: {raw_path}\nReport:   {out_path}\n")
    print(report.split("## Per-task")[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Local vs cloud LLM benchmark")
    ap.add_argument("--runs", type=int, default=3, help="runs per task per provider")
    ap.add_argument("--providers", nargs="+", default=["ollama", "cloud"])
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judge scoring")
    ap.add_argument("--out", default="docs/benchmark_report.md")
    ap.add_argument("--hardware", default="MacBook Air M1, 8 GB")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
