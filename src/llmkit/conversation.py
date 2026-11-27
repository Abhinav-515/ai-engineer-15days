"""Multi-turn RAG: conversation memory + query rewriting + grounded retrieval/generation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass

from llmkit.memory import ConversationMemory
from llmkit.rag import RAGResponse, RAGService, RAGStreamEvent
from llmkit.rewriter import QueryRewriter, RewriteResult

_STALE_CITATION = re.compile(r"\s*\[S\d+\]")


@dataclass(frozen=True)
class ConversationResult:
    """A grounded response together with how the question was rewritten."""

    response: RAGResponse
    rewrite: RewriteResult


def _strip_citations(answer: str) -> str:
    """Remove ``[S#]`` markers; labels are only meaningful within a single answer."""
    return _STALE_CITATION.sub("", answer).strip()


class ConversationalRAG:
    """One conversation session. Create one instance per session; nothing is global."""

    def __init__(
        self,
        rag: RAGService,
        rewriter: QueryRewriter,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._rag = rag
        self._rewriter = rewriter
        self._memory = memory if memory is not None else ConversationMemory()
        self._lock = asyncio.Lock()  # serialize turns so history cannot interleave

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def _remember(self, standalone_query: str, response: RAGResponse) -> None:
        answer = _strip_citations(response.answer)
        if answer:
            self._memory.add_turn(standalone_query, answer)

    async def ask(self, question: str) -> ConversationResult:
        """Answer one turn; memory is updated only if answering succeeds."""
        if not question.strip():
            raise ValueError("question must not be empty")

        async with self._lock:
            rewrite = await self._rewriter.rewrite(question, self._memory)
            response = await self._rag.ask(rewrite.query)
            self._remember(rewrite.query, response)
            return ConversationResult(response=response, rewrite=rewrite)

    async def stream_ask(self, question: str) -> AsyncGenerator[RAGStreamEvent, None]:
        """Stream one turn: a ``rewrite`` event, then token events, then final or error."""
        if not question.strip():
            raise ValueError("question must not be empty")

        async with self._lock:
            rewrite = await self._rewriter.rewrite(question, self._memory)
            yield RAGStreamEvent("rewrite", text=rewrite.query)

            async with aclosing(self._rag.stream_ask(rewrite.query)) as events:  # type: ignore[type-var]
                async for event in events:
                    if event.type == "final" and event.response is not None:
                        self._remember(rewrite.query, event.response)
                    yield event
