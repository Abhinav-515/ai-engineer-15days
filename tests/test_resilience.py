import json
from types import SimpleNamespace

import httpx
import pytest

from llmkit.client import LLMError
from llmkit.extract import ExtractionError
from llmkit.resilience import FallbackError, compute_delay, extract_with_fallback
from llmkit.schemas import Invoice

GOOD = {
    "invoice_number": "A-1",
    "vendor_name": "Acme",
    "invoice_date": "2026-03-14",
    "due_date": None,
    "currency": "USD",
    "status": "paid",
    "line_items": [
        {
            "description": "Widget",
            "quantity": 2,
            "unit_price": 5.0,
            "amount": 10.0,
        }
    ],
    "subtotal": 10.0,
    "tax": 1.0,
    "total": 11.0,
}


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate(
        self,
        messages,
        *,
        max_tokens=1024,
        temperature=0.0,
        json_schema=None,
    ):
        self.calls += 1
        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return SimpleNamespace(
            text=response,
            total_s=0.1,
            cost_usd=0.0,
        )


@pytest.fixture
def valid_json():
    return json.dumps(GOOD)


async def test_primary_success_does_not_call_fallback(valid_json):
    primary = FakeClient(valid_json)
    fallback = FakeClient()

    result = await extract_with_fallback(primary, fallback, "doc", Invoice)

    assert result.data.invoice_number == "A-1"
    assert primary.calls == 1
    assert fallback.calls == 0


async def test_retries_http_503_then_succeeds(valid_json):
    primary = FakeClient(
        LLMError("HTTP 503 from x: unavailable"),
        valid_json,
    )
    fallback = FakeClient()
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    result = await extract_with_fallback(
        primary,
        fallback,
        "doc",
        Invoice,
        max_attempts=3,
        base_delay_s=0.25,
        sleep=fake_sleep,
    )

    assert result.data.total == 11.0
    assert primary.calls == 2
    assert fallback.calls == 0
    assert delays == [0.25]


async def test_exponential_backoff_then_fallback(valid_json):
    primary = FakeClient(
        LLMError("HTTP 503 from x: unavailable"),
        LLMError("HTTP 429 from x: rate limited"),
    )
    fallback = FakeClient(valid_json)
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    result = await extract_with_fallback(
        primary,
        fallback,
        "doc",
        Invoice,
        max_attempts=2,
        base_delay_s=0.5,
        sleep=fake_sleep,
    )

    assert result.data.invoice_number == "A-1"
    assert primary.calls == 2
    assert fallback.calls == 1
    assert delays == [0.5]


async def test_validation_failure_uses_fallback(valid_json):
    invalid = {**GOOD, "total": 99.0}
    primary = FakeClient(json.dumps(invalid))
    fallback = FakeClient(valid_json)
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    result = await extract_with_fallback(
        primary,
        fallback,
        "doc",
        Invoice,
        sleep=fake_sleep,
    )

    assert result.data.total == 11.0
    assert primary.calls == 1
    assert fallback.calls == 1
    assert delays == []


async def test_permanent_http_error_skips_retries(valid_json):
    primary = FakeClient(
        LLMError("HTTP 401 from x: unauthorized"),
    )
    fallback = FakeClient(valid_json)
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    result = await extract_with_fallback(
        primary,
        fallback,
        "doc",
        Invoice,
        max_attempts=3,
        sleep=fake_sleep,
    )

    assert result.data.invoice_number == "A-1"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert delays == []


async def test_timeout_is_retried(valid_json):
    primary = FakeClient(
        httpx.ReadTimeout("timed out"),
        valid_json,
    )
    fallback = FakeClient()
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    result = await extract_with_fallback(
        primary,
        fallback,
        "doc",
        Invoice,
        sleep=fake_sleep,
    )

    assert result.data.invoice_number == "A-1"
    assert primary.calls == 2
    assert fallback.calls == 0
    assert delays == [0.5]


async def test_both_providers_failing_preserves_both_errors():
    primary = FakeClient(json.dumps({**GOOD, "total": 99.0}))
    fallback = FakeClient(
        LLMError("HTTP 503 from cloud: unavailable"),
    )

    with pytest.raises(FallbackError) as exc_info:
        await extract_with_fallback(primary, fallback, "doc", Invoice)

    assert isinstance(exc_info.value.primary_error, ExtractionError)
    assert isinstance(exc_info.value.fallback_error, LLMError)
    assert "Primary provider failed" in str(exc_info.value)


async def test_invalid_attempt_count_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        await extract_with_fallback(
            FakeClient(),
            FakeClient(),
            "doc",
            Invoice,
            max_attempts=0,
        )


def test_compute_delay_doubles_each_attempt():
    delays = [compute_delay(n, base_delay_s=0.5) for n in range(3)]
    assert delays == [0.5, 1.0, 2.0]


def test_compute_delay_is_capped():
    assert compute_delay(3, base_delay_s=10.0, max_delay_s=30.0) == 30.0


def test_compute_delay_honors_retry_after_and_caps_it():
    assert compute_delay(0, base_delay_s=0.5, retry_after_s=5.0) == 5.0
    assert compute_delay(0, base_delay_s=0.5, retry_after_s=999.0, max_delay_s=30.0) == 30.0


def test_compute_delay_jitter_range():
    kwargs = {"base_delay_s": 1.0, "jitter_ratio": 0.25}
    assert compute_delay(0, rng=lambda: 1.0, **kwargs) == 1.25
    assert compute_delay(0, rng=lambda: 0.0, **kwargs) == 0.75
    assert compute_delay(0, rng=lambda: 0.5, **kwargs) == 1.0


def test_jitter_never_exceeds_cap():
    delay = compute_delay(
        0, base_delay_s=30.0, max_delay_s=30.0, jitter_ratio=0.25, rng=lambda: 1.0
    )
    assert delay == 30.0


async def test_retry_after_header_sets_the_wait(valid_json):
    primary = FakeClient(
        LLMError("HTTP 429 from x: slow down", status_code=429, retry_after_s=4.0),
        valid_json,
    )
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    await extract_with_fallback(primary, FakeClient(), "doc", Invoice, sleep=fake_sleep)

    assert delays == [4.0]


async def test_fallback_retries_are_opt_in(valid_json):
    primary = FakeClient(json.dumps({**GOOD, "total": 99.0}))
    fallback = FakeClient(LLMError("HTTP 503 from x: unavailable"), valid_json)
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    result = await extract_with_fallback(
        primary,
        fallback,
        "doc",
        Invoice,
        fallback_max_attempts=2,
        sleep=fake_sleep,
    )

    assert result.data.invoice_number == "A-1"
    assert fallback.calls == 2
    assert delays == [0.5]


async def test_attempt_log_on_success(valid_json):
    primary = FakeClient(LLMError("HTTP 503 from x: unavailable"), valid_json)

    async def fake_sleep(delay):
        pass

    result = await extract_with_fallback(
        primary,
        FakeClient(),
        "doc",
        Invoice,
        base_delay_s=0.25,
        sleep=fake_sleep,
    )

    summary = [(a.role, a.attempt, a.outcome) for a in result.attempts]
    assert summary == [("primary", 1, "error"), ("primary", 2, "success")]
    assert result.attempts[0].status_code == 503
    assert result.attempts[0].delay_s == 0.25
    assert result.attempts[1].delay_s is None


async def test_attempt_log_on_total_failure():
    primary = FakeClient(json.dumps({**GOOD, "total": 99.0}))
    fallback = FakeClient(LLMError("HTTP 503 from cloud: unavailable"))

    with pytest.raises(FallbackError) as exc_info:
        await extract_with_fallback(primary, fallback, "doc", Invoice)

    attempts = exc_info.value.attempts
    assert [(a.role, a.error_type) for a in attempts] == [
        ("primary", "ExtractionError"),
        ("fallback", "LLMError"),
    ]
    assert attempts[1].status_code == 503


@pytest.mark.parametrize(
    "kwargs",
    [{"jitter_ratio": 1.5}, {"fallback_max_attempts": 0}, {"max_delay_s": -1}],
)
async def test_invalid_retry_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        await extract_with_fallback(FakeClient(), FakeClient(), "doc", Invoice, **kwargs)
