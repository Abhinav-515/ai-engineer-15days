"""Tests for LLM-judged Faithfulness and Answer Relevance (mocked client)."""

import json
from types import SimpleNamespace

import httpx
import pytest

from llmkit.eval_judge import LLMJudge, split_sentences


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, messages, *, max_tokens=1024, temperature=0.0, json_schema=None):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_schema": json_schema,
            }
        )
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, cost_usd=0.001)


def sentence_verdict(supported):
    return json.dumps({"reason": "because", "supported": supported})


def relevance_verdict(rating):
    return json.dumps({"reason": "because", "rating": rating})


ANSWER = "The vendor is Acme [S1]. The total is USD 15.00 [S2]."


def test_split_sentences_strips_citations():
    assert split_sentences(ANSWER) == ["The vendor is Acme.", "The total is USD 15.00."]


def test_split_sentences_keeps_decimals_and_splits_newlines():
    assert split_sentences("Line one.\nTotal is 85.80 USD") == [
        "Line one.",
        "Total is 85.80 USD",
    ]


def test_split_sentences_drops_citation_only_text():
    assert split_sentences("  [S1] \n [S2]  ") == []


async def test_faithfulness_all_supported():
    judge = LLMJudge(FakeClient([sentence_verdict(True), sentence_verdict(True)]))

    result = await judge.faithfulness(ANSWER, "context text")

    assert result.score == 1.0
    assert (result.supported, result.total) == (2, 2)
    assert result.error is None
    assert result.cost_usd == pytest.approx(0.002)


async def test_faithfulness_mixed_verdicts():
    judge = LLMJudge(FakeClient([sentence_verdict(True), sentence_verdict(False)]))

    result = await judge.faithfulness(ANSWER, "context text")

    assert result.score == 0.5
    assert [item.supported for item in result.judgements] == [True, False]
    assert result.judgements[1].sentence == "The total is USD 15.00."


async def test_faithfulness_prompt_contains_context_claim_and_schema():
    client = FakeClient([sentence_verdict(True)])
    judge = LLMJudge(client, max_tokens=123)

    await judge.faithfulness("The vendor is Acme [S1].", "CTX-MARKER")

    call = client.calls[0]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 123
    assert call["json_schema"] is not None
    assert "CTX-MARKER" in call["messages"][1]["content"]
    assert "The vendor is Acme." in call["messages"][1]["content"]
    assert "[S1]" not in call["messages"][1]["content"]


async def test_faithfulness_invalid_output_scores_none():
    judge = LLMJudge(FakeClient(["not json"]))

    result = await judge.faithfulness("One claim.", "ctx")

    assert result.score is None
    assert result.error == "sentence 1: invalid_output"


async def test_faithfulness_http_error_scores_none():
    judge = LLMJudge(FakeClient([httpx.ConnectError("boom")]))

    result = await judge.faithfulness("One claim.", "ctx")

    assert result.score is None
    assert result.error == "sentence 1: llm_error: ConnectError"


async def test_faithfulness_failure_midway_drops_score_but_keeps_verdicts():
    judge = LLMJudge(FakeClient([sentence_verdict(True), "not json"]))

    result = await judge.faithfulness(ANSWER, "ctx")

    assert result.score is None
    assert result.error == "sentence 2: invalid_output"
    assert result.supported == 1
    assert result.total == 2
    assert len(result.judgements) == 1


async def test_faithfulness_without_sentences_scores_none():
    client = FakeClient([])
    judge = LLMJudge(client)

    result = await judge.faithfulness("[S1]", "ctx")

    assert result.score is None
    assert result.error == "no_sentences"
    assert client.calls == []


async def test_faithfulness_empty_answer_scores_none():
    client = FakeClient([])
    judge = LLMJudge(client)

    result = await judge.faithfulness("", "ctx")

    assert result.score is None
    assert result.error == "no_sentences"
    assert client.calls == []


@pytest.mark.parametrize(
    ("rating", "score"),
    [("full", 1.0), ("partial", 0.5), ("none", 0.0)],
)
async def test_answer_relevance_maps_ratings_to_scores(rating, score):
    judge = LLMJudge(FakeClient([relevance_verdict(rating)]))

    result = await judge.answer_relevance("What is the total?", "It is USD 15.00.")

    assert result.score == score
    assert result.rating == rating
    assert result.error is None


async def test_answer_relevance_invalid_rating_scores_none():
    judge = LLMJudge(FakeClient([relevance_verdict("great")]))

    result = await judge.answer_relevance("Q?", "A.")

    assert result.score is None
    assert result.error == "invalid_output"


async def test_answer_relevance_http_error_scores_none():
    judge = LLMJudge(FakeClient([httpx.ConnectError("boom")]))

    result = await judge.answer_relevance("Q?", "A.")

    assert result.score is None
    assert result.error == "llm_error: ConnectError"


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError, match="max_tokens"):
        LLMJudge(FakeClient([]), max_tokens=0)
