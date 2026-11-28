from types import SimpleNamespace

import pytest

from llmkit.bench import RunRecord, build_report, judge_answer, percentile, summarize
from llmkit.bench_tasks import TASKS

CHECKS = {t.id: t.check for t in TASKS}


def _rec(provider="p", ok=True, passed=True, judge=None):
    return RunRecord(
        provider=provider,
        model="m",
        task_id="t",
        category="c",
        run=0,
        ok=ok,
        passed=passed,
        judge_score=judge,
        ttft_s=0.5,
        total_s=1.0,
        tokens_per_s=10.0,
        completion_tokens=10,
        cost_usd=0.001,
    )


def test_percentile_nearest_rank():
    assert percentile(list(range(1, 11)), 0.95) == 10
    assert percentile([5], 0.95) == 5
    assert percentile([3, 1, 2], 0.5) == 2


def test_task_checks_accept_good_and_reject_bad():
    assert CHECKS["math"]("391")
    assert not CHECKS["math"]("392")
    assert CHECKS["json"]('```json\n{"name": "Ann", "age": 30}\n```')
    assert not CHECKS["json"]('{"name": "Ann", "age": "30"}')
    assert CHECKS["list"]("apple\nbanana\ncherry")
    assert not CHECKS["list"]("1. apple\n2. banana\n3. cherry")
    assert CHECKS["capital"]("Canberra")
    assert not CHECKS["capital"]("Sydney")
    assert CHECKS["ball"]("5 cents")
    assert not CHECKS["ball"]("10 cents")


def test_summarize_counts_errors_as_failures():
    recs = [_rec(passed=True, judge=4), _rec(passed=False, judge=2), _rec(ok=False, passed=False)]
    s = summarize(recs)
    assert s["runs"] == 3
    assert s["errors"] == 1
    assert s["pass_rate"] == pytest.approx(1 / 3)
    assert s["judge_mean"] == pytest.approx(3.0)
    assert s["ttft_mean"] == pytest.approx(0.5)
    assert s["cost_per_1k"] == pytest.approx(1.0)


class FakeJudge:
    def __init__(self, text):
        self.text = text

    async def generate(self, messages, **kwargs):
        return SimpleNamespace(text=self.text)


async def test_judge_parses_json_and_rejects_garbage():
    task = TASKS[0]
    assert await judge_answer(FakeJudge('{"score": 4, "reason": "ok"}'), task, "x") == 4
    assert await judge_answer(FakeJudge("no json here"), task, "x") is None
    assert await judge_answer(FakeJudge('{"score": 9}'), task, "x") is None


def test_report_contains_both_providers():
    recs = [_rec(provider="ollama"), _rec(provider="cloud")]
    md = build_report(recs, {"date": "20261125", "runs": 1, "hardware": "test"})
    assert "ollama" in md and "cloud" in md
    assert "TTFT" in md and "Caveats" in md
