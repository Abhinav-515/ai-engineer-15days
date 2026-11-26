"""Grounded RAG: hybrid retrieval, structured generation and citation validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmkit.client import BaseLLMClient, Messages
from llmkit.hybrid import HybridRetriever, RetrievalMode
from llmkit.vector_store import SearchResult

REFUSAL_MESSAGE = (
    "I don't have enough information in the available documents to answer that question."
)

CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


class RAGError(ValueError):
    """The generated RAG response is malformed or contains invalid citations."""


class GeneratedAnswer(BaseModel):
    """Structured response expected from the language model."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Answer grounded only in the supplied context.")
    citations: list[str] = Field(
        description="Source labels used in the answer, such as [S1] and [S2]."
    )
    abstain: bool = Field(description="True when the supplied context cannot answer the question.")


@dataclass(frozen=True)
class Citation:
    """A source chunk cited by a generated answer."""

    label: str
    document: str
    chunk_index: int | None
    kind: str | None
    strategy: str | None
    chunk_id: str


@dataclass(frozen=True)
class RAGResponse:
    """Final grounded answer and its validated source references."""

    answer: str
    citations: list[Citation]
    abstained: bool
    retrieved_count: int
    provider: str = "unknown"
    latency_s: float = 0.0
    cost_usd: float = 0.0


SYSTEM_PROMPT = """You are a careful retrieval-grounded assistant.

Answer the user's question using ONLY the supplied context.
Do not use outside knowledge or invent facts.

CITATION RULES:
- Every factual claim must include an inline citation such as [S1].
- The literal marker MUST appear inside the answer string, not only in
  the separate citations array.
- Cite the specific chunk that supports each claim. Do not assume that
  one chunk supports facts found in another chunk from the same document.
- When an answer uses facts from multiple chunks, write separate
  sentences for facts supported by different chunks.
- Put the citation marker at the end of EVERY factual sentence.
  Never put one citation only at the end of a paragraph containing
  facts from multiple chunks.
- Before answering, identify which retrieved chunk supports each
  requested fact. Cite that chunk directly beside the fact.
- Example format: "The vendor is Acme Supplies and the line-item
  amount is USD 90.00 [S1]. The invoice total is USD 110.00 and its
  payment status is unpaid [S2]."
- The example is a formatting illustration only. Use actual retrieved
  facts and labels, never copy example facts or labels.
- Cite only source labels that appear in the supplied context.
- The citations field must list every source label used inline.
- Do not cite a source that does not support the claim.
- If the context is insufficient, set abstain to true, use an empty
  citations list, and explain that the documents do not contain the answer.
- Return only one JSON object matching the supplied JSON Schema.

REQUIRED CITATION FORMAT EXAMPLE:
{{
  "answer": "The invoice is marked unpaid [S1]. The listed total is USD 242.00 [S2].",
  "citations": ["[S1]", "[S2]"],
  "abstain": false
}}

The example is illustrative only. Use only facts and source labels from
the actual retrieved context. Never copy example facts into your answer.

JSON Schema:
{schema}
"""


class RAGService:
    """Retrieve relevant chunks and generate answers with validated citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        client: BaseLLMClient,
        *,
        retrieval_mode: RetrievalMode = "hybrid",
        top_k: int = 5,
        max_context_chars: int = 12_000,
        max_tokens: int = 1_000,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        self._retriever = retriever
        self._client = client
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens

    @staticmethod
    def _prepare_context(
        results: list[SearchResult],
        max_chars: int,
    ) -> tuple[str, list[Citation]]:
        """Format retrieved chunks and assign stable labels for this answer."""
        sections: list[str] = []
        citations: list[Citation] = []
        used_chars = 0
        separator = "\n\n---\n\n"

        for result in results:
            metadata = result.metadata or {}
            source = str(metadata.get("source") or "unknown")
            chunk_index = metadata.get("chunk_index")
            kind = metadata.get("kind")
            strategy = metadata.get("strategy")
            label = f"[S{len(citations) + 1}]"

            content_prefix = f"{label}\nSource: {source}\nChunk index: {chunk_index}\nContent:\n"
            separator_chars = len(separator) if sections else 0
            remaining = max_chars - used_chars - separator_chars
            available = remaining - len(content_prefix)

            if available <= 0:
                break

            section = content_prefix + result.text[:available]
            sections.append(section)
            used_chars += separator_chars + len(section)

            citations.append(
                Citation(
                    label=label,
                    document=source,
                    chunk_index=chunk_index,
                    kind=str(kind) if kind is not None else None,
                    strategy=str(strategy) if strategy is not None else None,
                    chunk_id=result.id,
                )
            )

        return separator.join(sections), citations

    @staticmethod
    def _validate_answer(
        generated: GeneratedAnswer,
        available_labels: set[str],
    ) -> list[str]:
        """Require exact agreement between inline and declared source labels."""
        if generated.abstain:
            if generated.citations or CITATION_PATTERN.search(generated.answer):
                raise RAGError("abstaining responses must not contain citations")
            return []

        if not generated.answer.strip():
            raise RAGError("generated answer must not be empty")

        inline_labels = CITATION_PATTERN.findall(generated.answer)
        inline_labels = [f"[S{number}]" for number in inline_labels]
        declared_labels = generated.citations

        if not inline_labels:
            raise RAGError(
                "non-abstaining answer must include inline citations; "
                f"generated answer={generated.answer!r}; "
                f"declared citations={declared_labels!r}"
            )

        if any(label not in available_labels for label in inline_labels):
            raise RAGError("answer contains an unknown source citation")

        if any(label not in available_labels for label in declared_labels):
            raise RAGError("declared citations contain an unknown source")

        if set(inline_labels) != set(declared_labels):
            raise RAGError("inline citations do not match the declared citations")

        if len(declared_labels) != len(set(declared_labels)):
            raise RAGError("declared citations must not contain duplicates")

        return declared_labels

    async def ask(self, question: str) -> RAGResponse:
        """Answer a question using retrieved context and validated citations."""
        if not question.strip():
            raise ValueError("question must not be empty")

        results = await self._retriever.search(
            question,
            mode=self.retrieval_mode,
            limit=self.top_k,
        )

        if not results:
            return RAGResponse(
                answer=REFUSAL_MESSAGE,
                citations=[],
                abstained=True,
                retrieved_count=0,
            )

        context, citations = self._prepare_context(
            results,
            self.max_context_chars,
        )

        if not citations:
            return RAGResponse(
                answer=REFUSAL_MESSAGE,
                citations=[],
                abstained=True,
                retrieved_count=len(results),
            )

        schema = GeneratedAnswer.model_json_schema()
        messages: Messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(schema=json.dumps(schema)),
            },
            {
                "role": "user",
                "content": (f"Question:\n{question.strip()}\n\nRetrieved context:\n{context}"),
            },
        ]

        result = await self._client.generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
            json_schema=schema,
        )

        try:
            generated = GeneratedAnswer.model_validate_json(result.text)
        except ValidationError as exc:
            raise RAGError(f"invalid structured RAG response: {exc}") from exc

        labels = self._validate_answer(
            generated,
            {citation.label for citation in citations},
        )

        if generated.abstain:
            return RAGResponse(
                answer=generated.answer.strip() or REFUSAL_MESSAGE,
                citations=[],
                abstained=True,
                retrieved_count=len(results),
                provider=getattr(self._client, "provider", "unknown"),
                latency_s=result.total_s,
                cost_usd=result.cost_usd,
            )

        citation_by_label = {citation.label: citation for citation in citations}
        selected_citations = [citation_by_label[label] for label in labels]

        return RAGResponse(
            answer=generated.answer.strip(),
            citations=selected_citations,
            abstained=False,
            retrieved_count=len(results),
            provider=getattr(self._client, "provider", "unknown"),
            latency_s=result.total_s,
            cost_usd=result.cost_usd,
        )
