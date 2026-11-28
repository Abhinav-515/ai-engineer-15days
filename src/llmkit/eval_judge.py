"""LLM-judged Faithfulness and Answer Relevance.

Judge failures never become zeros: a failed or invalid judge call makes the metric
``None`` and records the error, so noisy infrastructure cannot masquerade as a bad answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from llmkit.client import BaseLLMClient, LLMError, Messages

_CITATION = re.compile(r"\s*\[S\d+\]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")

RELEVANCE_SCORES: dict[str, float] = {"full": 1.0, "partial": 0.5, "none": 0.0}

FAITHFULNESS_PROMPT = """You are a strict fact-checking judge.

You are given CONTEXT and one CLAIM taken from an answer.
Decide whether the CONTEXT explicitly states or directly entails the CLAIM.

Rules:
- Use only the CONTEXT. Ignore outside knowledge, even if the claim is true.
- Ignore source labels such as [S1]; judge the content.
- If the claim needs information that the CONTEXT does not contain, it is not supported.
- Write a one-sentence reason first, then the verdict.
- Return only one JSON object matching the supplied JSON Schema."""

RELEVANCE_PROMPT = """You are a strict judge of answer relevance.

You are given a QUESTION and an ANSWER. Judge only whether the ANSWER addresses the
QUESTION. Do not judge whether the facts are correct.

Ratings:
- full: the answer directly and completely addresses what was asked.
- partial: the answer addresses only part of the question, or adds mostly unrelated content.
- none: the answer does not address the question.

Write a one-sentence reason first, then the rating.
Return only one JSON object matching the supplied JSON Schema."""


class SentenceVerdict(BaseModel):
    """Judge output for one answer sentence."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    supported: bool


class RelevanceVerdict(BaseModel):
    """Judge output for answer relevance."""

    model_config = ConfigDict(extra="forbid")

    reason: str
    rating: Literal["full", "partial", "none"]


@dataclass(frozen=True)
class SentenceJudgement:
    sentence: str
    supported: bool
    reason: str


@dataclass(frozen=True)
class FaithfulnessResult:
    """Faithfulness for one answer. ``score`` is None if the metric could not be computed."""

    score: float | None
    supported: int
    total: int
    judgements: tuple[SentenceJudgement, ...]
    error: str | None = None
    cost_usd: float = 0.0


@dataclass(frozen=True)
class RelevanceResult:
    """Answer Relevance for one answer. ``score`` is None if the judge failed."""

    score: float | None
    rating: str | None
    reason: str = ""
    error: str | None = None
    cost_usd: float = 0.0


def split_sentences(answer: str) -> list[str]:
    """Split an answer into sentences with citation markers removed.

    This is a deliberately simple splitter: it breaks after ``.``, ``!`` or ``?`` followed
    by whitespace, and on newlines. Abbreviations such as "Inc. " will split incorrectly.
    """
    cleaned = _CITATION.sub("", answer)
    parts = (part.strip() for part in _SENTENCE_BREAK.split(cleaned))
    return [part for part in parts if any(char.isalnum() for char in part)]


T = TypeVar("T", bound=BaseModel)


class LLMJudge:
    """Scores answers with a language model. Use a dedicated client for the judge."""

    def __init__(self, client: BaseLLMClient, *, max_tokens: int = 300) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._client = client
        self.max_tokens = max_tokens

    async def _verdict(
        self, model: type[T], system: str, user: str
    ) -> tuple[T | None, str | None, float]:
        """Return (verdict, error, cost). Exactly one of verdict and error is set."""
        schema = model.model_json_schema()
        messages: Messages = [
            {"role": "system", "content": f"{system}\nJSON Schema:\n{json.dumps(schema)}"},
            {"role": "user", "content": user},
        ]
        try:
            result = await self._client.generate(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
                json_schema=schema,
            )
        except (LLMError, httpx.HTTPError) as exc:
            return None, f"llm_error: {type(exc).__name__}", 0.0

        cost = float(getattr(result, "cost_usd", 0.0) or 0.0)
        try:
            return model.model_validate_json(result.text), None, cost
        except ValidationError:
            return None, "invalid_output", cost

    async def faithfulness(self, answer: str, context: str) -> FaithfulnessResult:
        """Share of answer sentences supported by the context the model actually saw."""
        sentences = split_sentences(answer)
        if not sentences:
            return FaithfulnessResult(None, 0, 0, (), error="no_sentences")

        judgements: list[SentenceJudgement] = []
        total_cost = 0.0
        for number, sentence in enumerate(sentences, start=1):
            verdict, error, cost = await self._verdict(
                SentenceVerdict,
                FAITHFULNESS_PROMPT,
                f"CONTEXT:\n{context}\n\nCLAIM:\n{sentence}",
            )
            total_cost += cost
            if verdict is None:
                return FaithfulnessResult(
                    None,
                    sum(item.supported for item in judgements),
                    len(sentences),
                    tuple(judgements),
                    error=f"sentence {number}: {error}",
                    cost_usd=total_cost,
                )
            judgements.append(SentenceJudgement(sentence, verdict.supported, verdict.reason))

        supported = sum(item.supported for item in judgements)
        return FaithfulnessResult(
            supported / len(sentences),
            supported,
            len(sentences),
            tuple(judgements),
            cost_usd=total_cost,
        )

    async def answer_relevance(self, question: str, answer: str) -> RelevanceResult:
        """Rate how directly the answer addresses the question."""
        verdict, error, cost = await self._verdict(
            RelevanceVerdict,
            RELEVANCE_PROMPT,
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}",
        )
        if verdict is None:
            return RelevanceResult(None, None, error=error, cost_usd=cost)
        return RelevanceResult(
            RELEVANCE_SCORES[verdict.rating],
            verdict.rating,
            verdict.reason,
            cost_usd=cost,
        )
