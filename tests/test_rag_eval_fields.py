"""Tests that RAGResponse exposes evaluation data without changing the SSE payload."""

import json
from types import SimpleNamespace

from llmkit.rag import RAGService, RAGStreamEvent
from llmkit.sse import rag_event_to_sse
from llmkit.vector_store import SearchResult

ANSWER = json.dumps(
    {"answer": "The service costs $50 [S1].", "citations": ["[S1]"], "abstain": False}
)
ABSTAIN = json.dumps({"answer": "The documents do not say.", "citations": [], "abstain": True})


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    async def search(self, query, *, mode, limit):
        return self.results


class FakeClient:
    def __init__(self, text):
        self.text = text

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        return SimpleNamespace(text=self.text, total_s=0.1, cost_usd=0.0)

    async def stream(self, messages, *, max_tokens, temperature, json_schema=None):
        yield SimpleNamespace(content=self.text, usage=None)


def make_results():
    return [
        SearchResult(
            id="chunk-1",
            text="The service costs $50.",
            score=0.9,
            metadata={"source": "pricing.pdf", "chunk_index": 0},
        ),
        SearchResult(
            id="chunk-2",
            text="Support is available weekdays.",
            score=0.5,
            metadata={"source": "pricing.pdf", "chunk_index": 1},
        ),
    ]


def make_service(text, results, **kwargs):
    return RAGService(FakeRetriever(results), FakeClient(text), **kwargs)


async def test_ask_exposes_context_and_all_retrieved_chunks():
    results = make_results()

    response = await make_service(ANSWER, results).ask("What does it cost?")

    assert response.abstained is False
    assert "The service costs $50." in response.context
    assert "[S1]" in response.context
    assert response.retrieved == tuple(results)


async def test_ask_abstention_keeps_retrieved_chunks():
    results = make_results()

    response = await make_service(ABSTAIN, results).ask("Unknown?")

    assert response.abstained is True
    assert response.retrieved == tuple(results)
    assert "The service costs $50." in response.context


async def test_ask_refusal_without_citable_context_keeps_retrieved_chunks():
    results = make_results()

    response = await make_service(ANSWER, results, max_context_chars=5).ask("Q?")

    assert response.abstained is True
    assert response.context == ""
    assert response.retrieved == tuple(results)


async def test_ask_with_no_results_has_empty_evaluation_fields():
    response = await make_service(ANSWER, []).ask("Q?")

    assert response.abstained is True
    assert response.context == ""
    assert response.retrieved == ()


async def test_stream_final_exposes_context_and_retrieved():
    results = make_results()

    events = [event async for event in make_service(ANSWER, results).stream_ask("Q?")]

    final = events[-1].response
    assert events[-1].type == "final"
    assert "The service costs $50." in final.context
    assert final.retrieved == tuple(results)


async def test_stream_refusal_keeps_retrieved_chunks():
    results = make_results()

    events = [
        event async for event in make_service(ANSWER, results, max_context_chars=5).stream_ask("Q?")
    ]

    assert events[-1].response.abstained is True
    assert events[-1].response.retrieved == tuple(results)


async def test_sse_final_payload_omits_evaluation_fields():
    response = await make_service(ANSWER, make_results()).ask("Q?")

    payload = rag_event_to_sse(RAGStreamEvent("final", response=response))

    assert '"retrieved_count"' in payload
    assert '"context"' not in payload
    assert '"retrieved"' not in payload
    assert "chunk-2" not in payload
