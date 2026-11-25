import json
from types import SimpleNamespace

import httpx
import pytest

from llmkit.client import LLMError
from llmkit.extract import ExtractionError
from llmkit.resilience import FallbackError, extract_with_fallback
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
