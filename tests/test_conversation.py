"""Tests for multi-turn conversational RAG using fake retriever and clients."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llmkit.client import LLMError, StreamEvent
from llmkit.conversation import ConversationalRAG
from llmkit.memory import ConversationMemory
from llmkit.rag import RAGError, RAGService
from llmkit.rewriter import QueryRewriter
from llmkit.sse import sse_stream
from llmkit.vector_store import SearchResult


def make_result() -> SearchResult:
    return SearchResult(
        id="chunk-1",
        text="Invoice NS-2026-204 is unpaid. The total is USD 110.00.",
        score=0.9,
        metadata={"source": "invoice.txt", "kind": "text", "chunk_index": 0, "strategy": "x"},
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(self, query: str, *, mode: str, limit: int) -> list[SearchResult]:
        self.calls.append(query)
        return [make_result()]


class FakeLLM:
    provider = "fake"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls.append(messages)
        return SimpleNamespace(text=self.text, total_s=0.1, cost_usd=0.0)

    async def stream(self, messages, *, max_tokens, temperature, json_schema=None):
        self.calls.append(messages)
        for i in range(0, len(self.text), 8):
            yield StreamEvent(content=self.text[i : i + 8])
        if self.error is not None:
            raise self.error


def answer_json(
    answer: str = "The invoice is unpaid [S1].",
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


def rewrite_json(question: str) -> str:
    return json.dumps({"standalone_question": question})


def build(
    *,
    max_turns: int = 5,
    gen_text: str | None = None,
    gen_error: Exception | None = None,
) -> tuple[ConversationalRAG, FakeRetriever, FakeLLM, FakeLLM]:
    retriever = FakeRetriever()
    gen_client = FakeLLM(answer_json() if gen_text is None else gen_text, gen_error)
    rewrite_client = FakeLLM()
    rag = RAGService(retriever, gen_client)  # type: ignore[arg-type]
    rewriter = QueryRewriter(rewrite_client)  # type: ignore[arg-type]
    conversation = ConversationalRAG(rag, rewriter, ConversationMemory(max_turns=max_turns))
    return conversation, retriever, gen_client, rewrite_client


FIRST = "What is the payment status of invoice NS-2026-204?"
FOLLOW_UP = "What is its total amount?"
STANDALONE = "What is the total amount of invoice NS-2026-204?"


async def test_first_question_skips_rewrite_and_is_remembered() -> None:
    conversation, retriever, _, rewrite_client = build()

    result = await conversation.ask(FIRST)

    assert rewrite_client.calls == []
    assert not result.rewrite.rewritten
    assert retriever.calls == [FIRST]
    assert result.response.answer == "The invoice is unpaid [S1]."
    assert len(conversation.memory) == 1


async def test_follow_up_retrieval_receives_standalone_question() -> None:
    conversation, retriever, gen_client, rewrite_client = build()
    await conversation.ask(FIRST)
    rewrite_client.text = rewrite_json(STANDALONE)

    result = await conversation.ask(FOLLOW_UP)

    assert result.rewrite.rewritten
    assert retriever.calls == [FIRST, STANDALONE]
    generation_prompt = gen_client.calls[-1][-1]["content"]
    assert STANDALONE in generation_prompt
    assert FOLLOW_UP not in generation_prompt


async def test_memory_stores_standalone_question_without_citation_markers() -> None:
    conversation, _, _, rewrite_client = build()
    await conversation.ask(FIRST)
    rewrite_client.text = rewrite_json(STANDALONE)
    await conversation.ask(FOLLOW_UP)

    turns = conversation.memory.turns

    assert turns[1].question == STANDALONE
    assert turns[0].answer == "The invoice is unpaid."
    assert all("[S" not in turn.answer for turn in turns)


async def test_independent_question_is_not_rewritten() -> None:
    conversation, retriever, _, rewrite_client = build()
    await conversation.ask(FIRST)
    independent = "Who is the vendor of invoice AB-77?"
    rewrite_client.text = rewrite_json(independent)

    result = await conversation.ask(independent)

    assert not result.rewrite.rewritten
    assert retriever.calls[-1] == independent


async def test_rewriter_failure_falls_back_to_original_question() -> None:
    conversation, retriever, _, rewrite_client = build()
    await conversation.ask(FIRST)
    rewrite_client.text = "not json"

    result = await conversation.ask(FOLLOW_UP)

    assert result.rewrite.fallback_reason == "invalid_output"
    assert retriever.calls[-1] == FOLLOW_UP


async def test_history_window_stays_bounded() -> None:
    conversation, _, _, rewrite_client = build(max_turns=2)
    rewrite_client.text = rewrite_json("standalone question")

    for i in range(5):
        await conversation.ask(f"question {i}")

    assert len(conversation.memory) == 2


async def test_failed_answer_does_not_update_memory() -> None:
    conversation, _, _, _ = build(gen_text=answer_json("Bad [S9].", ["[S9]"]))

    with pytest.raises(RAGError):
        await conversation.ask(FIRST)

    assert conversation.memory.is_empty()


async def test_abstention_is_remembered() -> None:
    text = answer_json("The documents do not contain the answer.", [], True)
    conversation, _, _, _ = build(gen_text=text)

    result = await conversation.ask(FIRST)

    assert result.response.abstained
    assert len(conversation.memory) == 1


async def test_sessions_do_not_share_memory() -> None:
    first, _, _, _ = build()
    second, _, _, _ = build()

    await first.ask(FIRST)

    assert second.memory.is_empty()


async def test_empty_question_rejected_without_side_effects() -> None:
    conversation, retriever, _, _ = build()

    with pytest.raises(ValueError, match="question"):
        await conversation.ask("   ")
    with pytest.raises(ValueError, match="question"):
        _ = [event async for event in conversation.stream_ask("   ")]

    assert retriever.calls == []
    assert conversation.memory.is_empty()


async def test_stream_emits_rewrite_tokens_final_and_updates_memory() -> None:
    conversation, retriever, _, rewrite_client = build()
    await conversation.ask(FIRST)
    rewrite_client.text = rewrite_json(STANDALONE)

    events = [event async for event in conversation.stream_ask(FOLLOW_UP)]

    assert events[0].type == "rewrite"
    assert events[0].text == STANDALONE
    assert events[-1].type == "final"
    assert all(e.type == "token" for e in events[1:-1])
    assert "".join(e.text for e in events[1:-1]) == "The invoice is unpaid [S1]."
    assert retriever.calls[-1] == STANDALONE
    assert len(conversation.memory) == 2


async def test_stream_error_does_not_update_memory() -> None:
    conversation, _, _, _ = build(gen_error=LLMError("HTTP 500", status_code=500))

    events = [event async for event in conversation.stream_ask(FIRST)]

    assert events[-1].type == "error"
    assert conversation.memory.is_empty()


async def test_sse_stream_includes_rewrite_frame_and_done() -> None:
    conversation, _, _, _ = build()

    frames = [frame async for frame in sse_stream(conversation.stream_ask(FIRST))]

    names = [frame.split("\n")[0][len("event: ") :] for frame in frames]
    assert names[0] == "rewrite"
    assert names[-2:] == ["final", "done"]
    assert frames[0] == f"event: rewrite\ndata: {json.dumps({'query': FIRST})}\n\n"
