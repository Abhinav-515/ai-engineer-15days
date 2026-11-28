"""Follow-up question rewriting: turn context-dependent questions into standalone ones."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmkit.client import BaseLLMClient, LLMError, Messages
from llmkit.memory import ConversationMemory

MAX_REWRITE_LENGTH_FACTOR = 4
MIN_REWRITE_CHARS = 200

REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions into standalone search questions.

You are given a conversation history and the user's latest question.

RULES:
- If the latest question refers to earlier turns (for example "it", "its",
  "that", "they", "the same one"), replace those references with the specific
  entities from the history so the question is understandable on its own.
- If the latest question is already standalone, return it EXACTLY unchanged.
- Do not answer the question. Do not add new facts or assumptions.
- If a reference cannot be resolved from the history, return the question
  EXACTLY unchanged. Never guess.
- Keep the question short and in the same language as the original.
- Return only one JSON object matching the supplied JSON Schema.

EXAMPLES (illustrative only):
History: User: What is the payment status of invoice AB-1? Assistant: Unpaid.
Latest: What is its total amount?
Output: What is the total amount of invoice AB-1?

History: User: What is the payment status of invoice AB-1? Assistant: Unpaid.
Latest: Who is the CEO of the company?
Output: Who is the CEO of the company?
"""


class RewrittenQuery(BaseModel):
    """Structured output expected from the rewriting model."""

    model_config = ConfigDict(extra="forbid")

    standalone_question: str = Field(
        description="The latest question rewritten so it can be understood without history."
    )


@dataclass(frozen=True)
class RewriteResult:
    """Outcome of rewriting; ``query`` is always safe to pass to retrieval."""

    original: str
    query: str
    rewritten: bool
    fallback_reason: str | None = None


class QueryRewriter:
    """Resolve follow-up references using conversation history.

    Rewriting is deliberately separate from retrieval: this class only produces a
    query string. Any failure falls back to the original question.
    """

    def __init__(self, client: BaseLLMClient, *, max_tokens: int = 200) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._client = client
        self.max_tokens = max_tokens

    async def rewrite(self, question: str, memory: ConversationMemory) -> RewriteResult:
        """Return a standalone version of ``question`` given ``memory``."""
        original = question.strip()
        if not original:
            raise ValueError("question must not be empty")

        if memory.is_empty():
            return RewriteResult(original=original, query=original, rewritten=False)

        schema = RewrittenQuery.model_json_schema()
        messages: Messages = [
            {
                "role": "system",
                "content": f"{REWRITE_SYSTEM_PROMPT}\nJSON Schema:\n{json.dumps(schema)}",
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{memory.format_for_prompt()}\n\n"
                    f"Latest question:\n{original}"
                ),
            },
        ]

        try:
            result = await self._client.generate(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.0,
                json_schema=schema,
            )
        except (LLMError, httpx.HTTPError) as exc:
            return self._fallback(original, f"llm_error: {type(exc).__name__}")

        try:
            candidate = RewrittenQuery.model_validate_json(result.text).standalone_question
        except ValidationError:
            return self._fallback(original, "invalid_output")

        candidate = " ".join(candidate.split())
        if not candidate:
            return self._fallback(original, "empty_output")

        max_chars = max(MIN_REWRITE_CHARS, len(original) * MAX_REWRITE_LENGTH_FACTOR)
        if len(candidate) > max_chars:
            return self._fallback(original, "output_too_long")

        return RewriteResult(
            original=original,
            query=candidate,
            rewritten=candidate != original,
        )

    @staticmethod
    def _fallback(original: str, reason: str) -> RewriteResult:
        return RewriteResult(
            original=original,
            query=original,
            rewritten=False,
            fallback_reason=reason,
        )
