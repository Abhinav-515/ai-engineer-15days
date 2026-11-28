"""Tests for grounded RAG generation and citation validation."""

import json
from types import SimpleNamespace

import pytest

from llmkit.rag import REFUSAL_MESSAGE, RAGError, RAGService
from llmkit.vector_store import SearchResult


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def search(self, query, *, mode, limit):
        self.calls.append((query, mode, limit))
        return self.results


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate(
        self,
        messages,
        *,
        max_tokens=1024,
        temperature=0.0,
        json_schema=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_schema": json_schema,
            }
        )
        return SimpleNamespace(
            text=self.response,
            total_s=0.25,
            cost_usd=0.001,
        )


def make_result(
    text="The service costs $50.",
    *,
    source="pricing.pdf",
    chunk_index=0,
):
    return SearchResult(
        id="chunk-1",
        text=text,
        score=0.9,
        metadata={
            "source": source,
            "kind": "pdf",
            "chunk_index": chunk_index,
            "strategy": "recursive",
        },
    )


def make_service(response, results=None):
    retriever = FakeRetriever([make_result()] if results is None else results)
    client = FakeClient(response)
    service = RAGService(retriever, client)
    return service, retriever, client


def response_json(
    answer="The service costs $50 [S1].",
    citations=None,
    abstain=False,
):
    return json.dumps(
        {
            "answer": answer,
            "citations": ["[S1]"] if citations is None else citations,
            "abstain": abstain,
        }
    )


async def test_valid_answer_returns_citation_metadata():
    service, retriever, client = make_service(response_json())

    result = await service.ask("How much does the service cost?")

    assert result.answer == "The service costs $50 [S1]."
    assert not result.abstained
    assert result.retrieved_count == 1
    assert len(result.citations) == 1
    assert result.citations[0].label == "[S1]"
    assert result.citations[0].document == "pricing.pdf"
    assert result.citations[0].chunk_index == 0
    assert result.citations[0].chunk_id == "chunk-1"
    assert len(client.calls) == 1
    assert retriever.calls[0][1] == "hybrid"


async def test_empty_retrieval_refuses_without_calling_llm():
    service, retriever, client = make_service(
        response_json(),
        results=[],
    )

    result = await service.ask("What does the document say?")

    assert result.answer == REFUSAL_MESSAGE
    assert result.abstained
    assert result.retrieved_count == 0
    assert result.citations == []
    assert retriever.calls
    assert client.calls == []


async def test_unknown_inline_citation_is_rejected():
    service, _, _ = make_service(
        response_json(
            answer="The price is $50 [S2].",
            citations=["[S2]"],
        )
    )

    with pytest.raises(RAGError, match="unknown source citation"):
        await service.ask("What is the price?")


async def test_inline_and_declared_citations_must_match():
    service, _, _ = make_service(
        response_json(
            answer="The price is $50 [S1].",
            citations=[],
        )
    )

    with pytest.raises(RAGError, match="do not match"):
        await service.ask("What is the price?")


async def test_non_abstaining_answer_requires_inline_citation():
    service, _, _ = make_service(
        response_json(
            answer="The price is $50.",
            citations=[],
        )
    )

    with pytest.raises(RAGError, match="must include inline citations"):
        await service.ask("What is the price?")


async def test_model_abstention_returns_no_citations():
    service, _, _ = make_service(
        response_json(
            answer="The documents do not contain this information.",
            citations=[],
            abstain=True,
        )
    )

    result = await service.ask("What is the weather tomorrow?")

    assert result.abstained
    assert result.citations == []
    assert "do not contain" in result.answer


async def test_abstention_with_citation_is_rejected():
    service, _, _ = make_service(
        response_json(
            answer="I cannot answer [S1].",
            citations=["[S1]"],
            abstain=True,
        )
    )

    with pytest.raises(RAGError, match="abstaining responses"):
        await service.ask("What is the price?")


async def test_invalid_json_is_rejected():
    service, _, _ = make_service("This is not JSON.")

    with pytest.raises(RAGError, match="invalid structured RAG response"):
        await service.ask("What is the price?")


async def test_empty_question_is_rejected_before_retrieval():
    service, retriever, client = make_service(response_json())

    with pytest.raises(ValueError, match="question must not be empty"):
        await service.ask("   ")

    assert retriever.calls == []
    assert client.calls == []


async def test_context_respects_configured_character_limit():
    service, _, client = make_service(response_json())
    service.max_context_chars = 80

    await service.ask("What is the price?")

    user_message = client.calls[0]["messages"][1]["content"]
    context = user_message.split("Retrieved context:\n", 1)[1]

    assert len(context) <= 80


async def test_context_limit_includes_separators_between_chunks():
    second = SearchResult(
        id="chunk-2",
        text="The service is available on weekdays.",
        score=0.8,
        metadata={
            "source": "schedule.pdf",
            "kind": "pdf",
            "chunk_index": 2,
            "strategy": "recursive",
        },
    )
    service, _, client = make_service(
        response_json(),
        results=[make_result(), second],
    )
    service.max_context_chars = 150

    await service.ask("What is the service cost?")

    user_message = client.calls[0]["messages"][1]["content"]
    context = user_message.split("Retrieved context:\n", 1)[1]

    assert len(context) <= service.max_context_chars
    assert "[S2]" in context


async def test_repeated_inline_citation_is_allowed():
    service, _, _ = make_service(
        response_json(
            answer="The service costs $50 [S1]. The same source confirms it [S1].",
            citations=["[S1]"],
        )
    )

    result = await service.ask("What is the service cost?")

    assert not result.abstained
    assert len(result.citations) == 1


async def test_duplicate_declared_citations_are_rejected():
    service, _, _ = make_service(
        response_json(
            answer="The service costs $50 [S1].",
            citations=["[S1]", "[S1]"],
        )
    )

    with pytest.raises(RAGError, match="must not contain duplicates"):
        await service.ask("What is the service cost?")


async def test_multiple_sources_receive_distinct_labels():
    second = SearchResult(
        id="chunk-2",
        text="The service is available on weekdays.",
        score=0.8,
        metadata={
            "source": "schedule.pdf",
            "kind": "pdf",
            "chunk_index": 2,
            "strategy": "recursive",
        },
    )
    service, _, _ = make_service(
        response_json(
            answer="It costs $50 [S1] and is available on weekdays [S2].",
            citations=["[S1]", "[S2]"],
        ),
        results=[make_result(), second],
    )

    result = await service.ask("Tell me the price and schedule.")

    assert [citation.label for citation in result.citations] == ["[S1]", "[S2]"]
    assert [citation.document for citation in result.citations] == [
        "pricing.pdf",
        "schedule.pdf",
    ]


async def test_uncited_retrieved_sources_are_not_returned():
    second = SearchResult(
        id="chunk-2",
        text="Unrelated information.",
        score=0.7,
        metadata={"source": "other.pdf", "chunk_index": 1},
    )
    service, _, _ = make_service(
        response_json(
            answer="The service costs $50 [S1].",
            citations=["[S1]"],
        ),
        results=[make_result(), second],
    )

    result = await service.ask("What is the service cost?")

    assert len(result.citations) == 1
    assert result.citations[0].document == "pricing.pdf"


async def test_missing_invoice_identifier_abstains_without_calling_llm():
    result = make_result("Invoice No: NS-2026-204\nPayment Status: UNPAID")
    service, _, client = make_service(response_json(), results=[result])

    response = await service.ask("What is the payment status of invoice NS-2026-999?")

    assert response.abstained
    assert response.answer == REFUSAL_MESSAGE
    assert response.citations == []
    assert client.calls == []


async def test_exact_invoice_identifier_allows_generation():
    result = make_result("Invoice No: NS-2026-204\nPayment Status: UNPAID")
    service, _, client = make_service(
        response_json("The invoice is unpaid [S1]."),
        results=[result],
    )

    response = await service.ask("What is the payment status of invoice NS-2026-204?")

    assert not response.abstained
    assert len(client.calls) == 1


async def test_question_without_record_identifier_keeps_existing_behavior():
    service, _, client = make_service(response_json())

    response = await service.ask("How much does the service cost?")

    assert not response.abstained
    assert len(client.calls) == 1


async def test_hyphenated_phrase_is_not_treated_as_record_identifier():
    service, _, client = make_service(response_json())

    response = await service.ask("What is the follow-up process?")

    assert not response.abstained
    assert len(client.calls) == 1


async def test_partial_invoice_identifier_does_not_match_longer_identifier():
    result = make_result("Invoice No: NS-2026-204\nPayment Status: UNPAID")
    service, _, client = make_service(response_json(), results=[result])

    response = await service.ask("What is the payment status of invoice NS-2026-20?")

    assert response.abstained
    assert response.answer == REFUSAL_MESSAGE
    assert response.citations == []
    assert client.calls == []
