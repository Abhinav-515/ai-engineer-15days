"""Tests for streaming grounded RAG using fake retriever and client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from llmkit.client import LLMError, StreamEvent, Usage
from llmkit.rag import REFUSAL_MESSAGE, RAGService, RAGStreamEvent
from llmkit.vector_store import SearchResult


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple] = []

    async def search(self, query: str, *, mode: str, limit: int) -> list[SearchResult]:
        self.calls.append((query, mode, limit))
        return self.results


class FakeStreamClient:
    provider = "fake"

    def __init__(
        self,
        chunks: list[str],
        *,
        error: Exception | None = None,
        usage: Usage | None = None,
        price_in_per_m: float = 0.0,
        price_out_per_m: float = 0.0,
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.usage = usage
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self.finished = False
        self.calls = 0

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_schema: dict | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        for chunk in self.chunks:
            yield StreamEvent(content=chunk)
        if self.error is not None:
            raise self.error
        if self.usage is not None:
            yield StreamEvent(usage=self.usage)
        self.finished = True


def make_result() -> SearchResult:
    return SearchResult(
        id="chunk-1",
        text="The service costs $50.",
        score=0.9,
        metadata={"source": "pricing.pdf", "kind": "pdf", "chunk_index": 0, "strategy": "x"},
    )


def payload(
    answer: str = "The service costs $50 [S1].",
    citations: list[str] | None = None,
    abstain: bool = False,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": ["[S1]"] if citations is None else citations,
            "abstain": abstain,
        }
    )


def split(text: str, size: int = 6) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def make_service(
    client: FakeStreamClient, results: list[SearchResult] | None = None
) -> tuple[RAGService, FakeRetriever]:
    retriever = FakeRetriever([make_result()] if results is None else results)
    return RAGService(retriever, client), retriever  # type: ignore[arg-type]


async def collect(service: RAGService, question: str = "How much?") -> list[RAGStreamEvent]:
    return [event async for event in service.stream_ask(question)]


async def test_streams_tokens_then_validated_final() -> None:
    service, _ = make_service(FakeStreamClient(split(payload())))

    events = await collect(service)

    tokens = [e for e in events if e.type == "token"]
    assert len(tokens) > 1
    assert "".join(e.text for e in tokens) == "The service costs $50 [S1]."
    assert [e.type for e in events[:-1]] == ["token"] * len(tokens)
    final = events[-1]
    assert final.type == "final"
    assert final.response is not None
    assert final.response.answer == "The service costs $50 [S1]."
    assert [c.label for c in final.response.citations] == ["[S1]"]
    assert not final.response.abstained


async def test_tokens_are_emitted_before_generation_finishes() -> None:
    client = FakeStreamClient(split(payload()))
    service, _ = make_service(client)
    finished_at_first_token: bool | None = None

    async for event in service.stream_ask("How much?"):
        if event.type == "token" and finished_at_first_token is None:
            finished_at_first_token = client.finished

    assert finished_at_first_token is False
    assert client.finished


async def test_no_results_yields_refusal_without_calling_llm() -> None:
    client = FakeStreamClient(split(payload()))
    service, _ = make_service(client, results=[])

    events = await collect(service)

    assert len(events) == 1
    assert events[0].type == "final"
    assert events[0].response is not None
    assert events[0].response.answer == REFUSAL_MESSAGE
    assert events[0].response.abstained
    assert client.calls == 0


async def test_abstention_returns_final_without_citations() -> None:
    answer = "The documents do not contain the answer."
    service, _ = make_service(FakeStreamClient(split(payload(answer, [], True))))

    events = await collect(service)

    final = events[-1]
    assert final.type == "final"
    assert final.response is not None
    assert final.response.abstained
    assert final.response.citations == []


async def test_invalid_citation_yields_error_and_no_final() -> None:
    service, _ = make_service(FakeStreamClient(split(payload("Bad [S9].", ["[S9]"]))))

    events = await collect(service)

    assert events[-1].type == "error"
    assert events[-1].error is not None
    assert "unknown source" in events[-1].error
    assert all(e.type != "final" for e in events)


async def test_invalid_json_yields_error_event() -> None:
    service, _ = make_service(FakeStreamClient(["not json"]))

    events = await collect(service)

    assert [e.type for e in events] == ["error"]
    assert events[0].error is not None
    assert "invalid structured RAG response" in events[0].error


async def test_llm_error_mid_stream_yields_error_event() -> None:
    chunks = split(payload())[:3]
    client = FakeStreamClient(chunks, error=LLMError("HTTP 500", status_code=500))
    service, _ = make_service(client)

    events = await collect(service)

    assert events[-1].type == "error"
    assert events[-1].error == "generation failed: LLMError"
    assert any(e.type == "token" for e in events)
    assert all(e.type != "final" for e in events)


async def test_cost_is_estimated_from_usage() -> None:
    client = FakeStreamClient(
        split(payload()),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        price_in_per_m=1_000_000.0,
        price_out_per_m=2_000_000.0,
    )
    service, _ = make_service(client)

    final = (await collect(service))[-1]

    assert final.response is not None
    assert final.response.cost_usd == pytest.approx(20.0)
    assert final.response.provider == "fake"


async def test_retriever_receives_question_mode_and_limit() -> None:
    service, retriever = make_service(FakeStreamClient(split(payload())))

    await collect(service, "How much?")

    assert retriever.calls == [("How much?", "hybrid", 5)]


async def test_empty_question_is_rejected() -> None:
    service, _ = make_service(FakeStreamClient([]))

    with pytest.raises(ValueError, match="question"):
        await collect(service, "   ")


async def test_missing_invoice_identifier_abstains_without_streaming():
    result = SearchResult(
        id="invoice-1",
        text="Invoice No: NS-2026-204\nPayment Status: UNPAID",
        score=0.9,
        metadata={
            "source": "invoice.txt",
            "kind": "text",
            "chunk_index": 0,
            "strategy": "recursive",
        },
    )
    client = FakeStreamClient(split(payload()))
    service, _ = make_service(client, results=[result])

    events = await collect(
        service,
        "What is the payment status of invoice NS-2026-999?",
    )

    assert len(events) == 1
    assert events[0].type == "final"
    assert events[0].response is not None
    assert events[0].response.abstained
    assert events[0].response.citations == []
    assert client.calls == 0
