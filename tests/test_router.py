"""Tests for structured query routing and safe fallback behavior."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from llmkit.client import BaseLLMClient, JsonSchema, LLMError, Messages, StreamEvent
from llmkit.router import QueryRouter


class FakeClient(BaseLLMClient):
    """Returns a canned response and records the request."""

    provider = "fake"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        super().__init__("fake-model", base_url="http://fake.invalid")
        self.text = text
        self.error = error
        self.calls: list[Messages] = []
        self.schemas: list[JsonSchema | None] = []

    async def stream(
        self,
        messages: Messages,
        *,
        max_tokens: int,
        temperature: float,
        json_schema: JsonSchema | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(messages)
        self.schemas.append(json_schema)
        if self.error is not None:
            raise self.error
        yield StreamEvent(content=self.text)


def reply(route: str, confidence: float = 0.95, reason: str = "Relevant source") -> str:
    return json.dumps({"route": route, "confidence": confidence, "reason": reason})


@pytest.mark.parametrize("route", ["compute", "retrieve", "both"])
async def test_valid_routes_are_returned(route: str) -> None:
    client = FakeClient(reply(route))
    router = QueryRouter(client)

    result = await router.route("What is the total for invoice INV-1?")

    assert result.route == route
    assert result.confidence == 0.95
    assert result.reason == "Relevant source"
    assert result.fallback_reason is None
    assert len(client.calls) == 1
    assert client.schemas[0] is not None
    assert "Route this question" in client.calls[0][-1]["content"]


async def test_router_sends_structured_output_schema() -> None:
    client = FakeClient(reply("compute"))
    router = QueryRouter(client)

    await router.route("How much is unpaid?")

    schema = client.schemas[0]
    assert schema is not None
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"route", "confidence", "reason"}


async def test_invalid_json_falls_back_to_retrieval() -> None:
    router = QueryRouter(FakeClient("not json"))

    result = await router.route("What is unpaid?")

    assert result.route == "retrieve"
    assert result.confidence == 0.0
    assert result.fallback_reason == "invalid_output"


@pytest.mark.parametrize(
    "payload",
    [
        {"route": "delete", "confidence": 1.0, "reason": "bad route"},
        {"route": "compute", "confidence": 1.5, "reason": "bad confidence"},
        {"route": "retrieve", "confidence": 0.5},
        {
            "route": "both",
            "confidence": 0.5,
            "reason": "valid",
            "sql": "DROP TABLE invoices",
        },
    ],
)
async def test_invalid_structured_output_falls_back(payload: dict) -> None:
    router = QueryRouter(FakeClient(json.dumps(payload)))

    result = await router.route("Question")

    assert result.route == "retrieve"
    assert result.fallback_reason == "invalid_output"


async def test_llm_error_falls_back_to_retrieval() -> None:
    error = LLMError("HTTP 500", status_code=500)
    router = QueryRouter(FakeClient(error=error))

    result = await router.route("Question")

    assert result.route == "retrieve"
    assert result.fallback_reason == "llm_error: LLMError"


@pytest.mark.parametrize(
    "question",
    [
        "What happened to the payment gateway on March 14, 2026?",
        "When was the retry completed?",
        "What does the application.log say about the incident?",
    ],
)
async def test_operational_event_questions_are_forced_to_retrieval(
    question: str,
) -> None:
    # Even if the model incorrectly chooses compute, document-event questions
    # must use the source-document retrieval path.
    router = QueryRouter(FakeClient(reply("compute")))

    result = await router.route(question)

    assert result.route == "retrieve"
    assert "Operational events" in result.reason


async def test_invoice_aggregate_question_can_still_use_compute() -> None:
    router = QueryRouter(FakeClient(reply("compute")))

    result = await router.route(
        "What is the total amount of unpaid invoices in the structured dataset?"
    )

    assert result.route == "compute"


async def test_invoice_date_question_is_not_forced_to_retrieval() -> None:
    router = QueryRouter(FakeClient(reply("compute")))

    result = await router.route("When was invoice INV-2026-204 paid?")

    assert result.route == "compute"
    assert result.fallback_reason is None


async def test_whitespace_in_reason_is_normalized() -> None:
    router = QueryRouter(FakeClient(reply("compute", reason="  exact\n totals  ")))

    result = await router.route("Question")

    assert result.reason == "exact totals"


async def test_empty_question_is_rejected() -> None:
    router = QueryRouter(FakeClient(reply("retrieve")))

    with pytest.raises(ValueError, match="question"):
        await router.route("   ")

    assert router._client.calls == []


def test_non_positive_max_tokens_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        QueryRouter(FakeClient(), max_tokens=0)
