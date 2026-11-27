"""Tests for follow-up query rewriting using a fake LLM client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from llmkit.client import BaseLLMClient, JsonSchema, LLMError, Messages, StreamEvent
from llmkit.memory import ConversationMemory
from llmkit.rewriter import QueryRewriter


class FakeClient(BaseLLMClient):
    """Returns a canned response and records every call."""

    provider = "fake"

    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        super().__init__("fake-model", base_url="http://fake.invalid")
        self.text = text
        self.error = error
        self.calls: list[Messages] = []

    async def stream(
        self,
        messages: Messages,
        *,
        max_tokens: int,
        temperature: float,
        json_schema: JsonSchema | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        yield StreamEvent(content=self.text)


def _reply(question: str) -> str:
    return json.dumps({"standalone_question": question})


@pytest.fixture
def memory() -> ConversationMemory:
    mem = ConversationMemory()
    mem.add_turn("What is the payment status of invoice NS-2026-204?", "It is unpaid.")
    return mem


async def test_empty_history_skips_llm_call() -> None:
    client = FakeClient(_reply("should not be used"))
    rewriter = QueryRewriter(client)

    result = await rewriter.rewrite("What is its total amount?", ConversationMemory())

    assert result.query == "What is its total amount?"
    assert not result.rewritten
    assert result.fallback_reason is None
    assert client.calls == []


async def test_follow_up_is_rewritten_using_history(memory: ConversationMemory) -> None:
    client = FakeClient(_reply("What is the total amount of invoice NS-2026-204?"))
    rewriter = QueryRewriter(client)

    result = await rewriter.rewrite("What is its total amount?", memory)

    assert result.rewritten
    assert result.query == "What is the total amount of invoice NS-2026-204?"
    assert result.original == "What is its total amount?"
    assert len(client.calls) == 1
    user_prompt = client.calls[0][-1]["content"]
    assert "NS-2026-204" in user_prompt
    assert "What is its total amount?" in user_prompt


async def test_independent_question_is_not_rewritten(memory: ConversationMemory) -> None:
    question = "Who is the vendor of invoice AB-77?"
    rewriter = QueryRewriter(FakeClient(_reply(question)))

    result = await rewriter.rewrite(question, memory)

    assert result.query == question
    assert not result.rewritten
    assert result.fallback_reason is None


async def test_ambiguous_reference_stays_unchanged(memory: ConversationMemory) -> None:
    question = "Who approved them?"
    rewriter = QueryRewriter(FakeClient(_reply(question)))

    result = await rewriter.rewrite(question, memory)

    assert result.query == question
    assert not result.rewritten


async def test_invalid_json_falls_back(memory: ConversationMemory) -> None:
    rewriter = QueryRewriter(FakeClient("not json at all"))

    result = await rewriter.rewrite("What is its total amount?", memory)

    assert result.query == "What is its total amount?"
    assert not result.rewritten
    assert result.fallback_reason == "invalid_output"


async def test_empty_output_falls_back(memory: ConversationMemory) -> None:
    rewriter = QueryRewriter(FakeClient(_reply("   ")))

    result = await rewriter.rewrite("What is its total amount?", memory)

    assert result.query == "What is its total amount?"
    assert result.fallback_reason == "empty_output"


async def test_overlong_output_falls_back(memory: ConversationMemory) -> None:
    rewriter = QueryRewriter(FakeClient(_reply("word " * 200)))

    result = await rewriter.rewrite("What is its total amount?", memory)

    assert result.query == "What is its total amount?"
    assert result.fallback_reason == "output_too_long"


async def test_llm_error_falls_back(memory: ConversationMemory) -> None:
    rewriter = QueryRewriter(FakeClient(error=LLMError("HTTP 500", status_code=500)))

    result = await rewriter.rewrite("What is its total amount?", memory)

    assert result.query == "What is its total amount?"
    assert not result.rewritten
    assert result.fallback_reason == "llm_error: LLMError"


async def test_rewrite_does_not_mutate_memory(memory: ConversationMemory) -> None:
    before = memory.turns
    rewriter = QueryRewriter(FakeClient(_reply("What is the total of NS-2026-204?")))

    await rewriter.rewrite("What is its total amount?", memory)

    assert memory.turns == before


async def test_whitespace_in_rewrite_is_normalized(memory: ConversationMemory) -> None:
    rewriter = QueryRewriter(FakeClient(_reply("  What is\nthe total of NS-2026-204?  ")))

    result = await rewriter.rewrite("What is its total amount?", memory)

    assert result.query == "What is the total of NS-2026-204?"


async def test_empty_question_is_rejected(memory: ConversationMemory) -> None:
    rewriter = QueryRewriter(FakeClient(_reply("x")))

    with pytest.raises(ValueError, match="question"):
        await rewriter.rewrite("   ", memory)


def test_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        QueryRewriter(FakeClient(), max_tokens=0)
