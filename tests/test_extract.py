import json
from types import SimpleNamespace

import pytest

from llmkit.client import LLMError
from llmkit.extract import ExtractionError, clean_json_text, extract
from llmkit.schemas import Invoice

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


class FakeClient:
    """Stands in for an LLM client; records the arguments it was called with."""

    def __init__(self, *texts_or_errors):
        self.queue = list(texts_or_errors)
        self.schemas = []

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.schemas.append(json_schema)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, total_s=0.1, cost_usd=0.0)


async def test_valid_json_becomes_invoice():
    res = await extract(FakeClient(json.dumps(GOOD)), "doc", Invoice)
    assert res.data.total == 11.0
    assert res.data.due_date is None
    assert res.data.currency.value == "USD"
    assert res.used_native_schema


async def test_fenced_json_with_chatter_is_cleaned():
    text = "Sure! Here it is:\n```json\n" + json.dumps(GOOD) + "\n```\nHope that helps."
    res = await extract(FakeClient(text), "doc", Invoice)
    assert res.data.invoice_number == "A-1"


async def test_total_mismatch_is_rejected_by_business_rule():
    bad = {**GOOD, "total": 99.0}
    with pytest.raises(ExtractionError, match="tax"):
        await extract(FakeClient(json.dumps(bad)), "doc", Invoice)


async def test_invalid_enum_is_rejected():
    bad = {**GOOD, "currency": "BTC"}
    with pytest.raises(ExtractionError, match="currency"):
        await extract(FakeClient(json.dumps(bad)), "doc", Invoice)


async def test_missing_required_field_is_rejected():
    bad = {k: v for k, v in GOOD.items() if k != "vendor_name"}
    with pytest.raises(ExtractionError, match="vendor_name"):
        await extract(FakeClient(json.dumps(bad)), "doc", Invoice)


async def test_falls_back_without_schema_on_http_400():
    client = FakeClient(LLMError("HTTP 400 from x: bad schema"), json.dumps(GOOD))
    res = await extract(client, "doc", Invoice)
    assert not res.used_native_schema
    assert client.schemas[0] is not None and client.schemas[1] is None


async def test_other_provider_errors_propagate():
    with pytest.raises(LLMError):
        await extract(FakeClient(LLMError("HTTP 401 from x: nope")), "doc", Invoice)


def test_schema_is_strict_friendly():
    schema = Invoice.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])  # nothing optional
    assert schema["properties"]["currency"]["$ref"].endswith("Currency")


def test_clean_json_text():
    assert clean_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'
