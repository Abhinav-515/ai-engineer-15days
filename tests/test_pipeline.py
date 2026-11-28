import asyncio
import json
from types import SimpleNamespace

import pytest

from llmkit.pipeline import BatchReport, process_document, run_batch

GOOD = {
    "invoice_number": "A-1",
    "vendor_name": "Acme",
    "invoice_date": "2026-03-14",
    "due_date": None,
    "currency": "USD",
    "status": "paid",
    "line_items": [{"description": "Widget", "quantity": 2, "unit_price": 5.0, "amount": 10.0}],
    "subtotal": 10.0,
    "tax": 1.0,
    "total": 11.0,
}
BAD = {**GOOD, "total": 99.0}


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response, total_s=0.1, cost_usd=0.5)


class SlowClient:
    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return SimpleNamespace(text=json.dumps(GOOD), total_s=0.01, cost_usd=0.0)


async def no_sleep(delay):
    return None


def write_doc(tmp_path, name="inv.txt", text="Invoice A-1 total 11.00"):
    path = tmp_path / name
    path.write_text(text)
    return path


async def test_successful_document(tmp_path):
    outcome = await process_document(
        write_doc(tmp_path), FakeClient(json.dumps(GOOD)), FakeClient(), sleep=no_sleep
    )
    assert outcome.status == "ok"
    assert outcome.invoice.invoice_number == "A-1"
    assert outcome.used_fallback is False
    assert outcome.cost_usd == 0.5
    assert outcome.error is None


async def test_validation_failure_uses_fallback(tmp_path):
    primary = FakeClient(json.dumps(BAD))
    fallback = FakeClient(json.dumps(GOOD))
    outcome = await process_document(write_doc(tmp_path), primary, fallback, sleep=no_sleep)
    assert outcome.status == "ok"
    assert outcome.used_fallback is True
    assert [a.role for a in outcome.attempts] == ["primary", "fallback"]


async def test_both_providers_failing_is_recorded(tmp_path):
    outcome = await process_document(
        write_doc(tmp_path),
        FakeClient(json.dumps(BAD)),
        FakeClient(json.dumps(BAD)),
        sleep=no_sleep,
    )
    assert outcome.status == "failed"
    assert outcome.error_stage == "extract"
    assert len(outcome.attempts) == 2
    assert outcome.invoice is None


async def test_unexpected_error_is_isolated(tmp_path):
    outcome = await process_document(
        write_doc(tmp_path), FakeClient(KeyError("boom")), FakeClient(), sleep=no_sleep
    )
    assert outcome.status == "failed"
    assert outcome.error_stage == "unexpected"
    assert "KeyError" in outcome.error


async def test_bad_files_do_not_stop_the_batch(tmp_path):
    good = write_doc(tmp_path)
    empty = write_doc(tmp_path, "empty.log", "  \n")
    unknown = write_doc(tmp_path, "data.xyz", "x")
    missing = tmp_path / "missing.pdf"
    report = await run_batch(
        [empty, good, unknown, missing],
        FakeClient(json.dumps(GOOD)),
        FakeClient(),
        sleep=no_sleep,
    )
    assert isinstance(report, BatchReport)
    assert [o.status for o in report.outcomes] == ["failed", "ok", "failed", "failed"]
    assert [o.error_stage for o in report.outcomes] == ["load", None, "load", "load"]
    assert len(report.invoices) == 1
    assert report.summary()["ok"] == 1
    assert report.summary()["failed"] == 3


async def test_concurrency_is_bounded_and_order_preserved(tmp_path):
    paths = [write_doc(tmp_path, f"doc{i}.txt") for i in range(6)]
    client = SlowClient()
    report = await run_batch(paths, client, FakeClient(), concurrency=2, sleep=no_sleep)
    assert client.max_in_flight == 2
    assert [o.source for o in report.outcomes] == [str(p) for p in paths]


async def test_summary_counts_fallback_and_cost(tmp_path):
    first = write_doc(tmp_path, "a.txt")
    second = write_doc(tmp_path, "b.txt")
    primary = FakeClient(json.dumps(GOOD), json.dumps(BAD))
    fallback = FakeClient(json.dumps(GOOD))
    report = await run_batch([first, second], primary, fallback, concurrency=1, sleep=no_sleep)
    summary = report.summary()
    assert summary["ok"] == 2
    assert summary["used_fallback"] == 1
    assert summary["total_cost_usd"] == 1.0


async def test_invalid_concurrency_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        await run_batch([], FakeClient(), FakeClient(), concurrency=0)
