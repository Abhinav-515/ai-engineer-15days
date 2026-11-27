"""Orchestrate deterministic invoice computation and grounded RAG."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from llmkit.agent import AgentResult, ToolAgent
from llmkit.rag import Citation, RAGResponse, RAGService
from llmkit.router import QueryRouter, RouterResult


@dataclass(frozen=True)
class HybridResponse:
    """Combined answer and source results from the selected route."""

    route: Literal["compute", "retrieve", "both"]
    answer: str
    citations: list[Citation]
    routing: RouterResult
    compute_result: AgentResult | None = None
    rag_result: RAGResponse | None = None


class HybridAgent:
    """Route questions to invoice tools, RAG, or both."""

    def __init__(
        self,
        router: QueryRouter,
        tool_agent: ToolAgent,
        rag_service: RAGService,
    ) -> None:
        self._router = router
        self._tool_agent = tool_agent
        self._rag_service = rag_service

    async def ask(self, question: str) -> HybridResponse:
        """Route a question and return the selected service results."""
        if not question.strip():
            raise ValueError("question must not be empty")

        routing = await self._router.route(question)

        if routing.route == "compute":
            compute_result = await self._tool_agent.run(question)
            return HybridResponse(
                route="compute",
                answer=self._compute_answer(compute_result),
                citations=[],
                routing=routing,
                compute_result=compute_result,
            )

        if routing.route == "retrieve":
            rag_result = await self._rag_service.ask(self._document_task(question))
            return HybridResponse(
                route="retrieve",
                answer=rag_result.answer,
                citations=rag_result.citations,
                routing=routing,
                rag_result=rag_result,
            )

        compute_task = self._structured_data_task(question)
        document_task = self._document_task(question)
        compute_result, rag_result = await asyncio.gather(
            self._tool_agent.run(compute_task),
            self._rag_service.ask(document_task),
        )

        compute_summary = self._verified_status_summary(compute_result)
        answer = (
            "INVOICE DATA\n"
            f"{compute_summary or self._compute_answer(compute_result)}\n\n"
            "DOCUMENT CONTEXT\n"
            f"{rag_result.answer}"
        )

        return HybridResponse(
            route="both",
            answer=answer,
            citations=rag_result.citations,
            routing=routing,
            compute_result=compute_result,
            rag_result=rag_result,
        )

    @staticmethod
    def _structured_data_task(question: str) -> str:
        """Keep the compute agent focused on facts owned by the SQL dataset."""
        return (
            "Answer only the part of this request that can be answered from the "
            "structured invoice dataset. Do not look up invoice identifiers or "
            "facts that the user explicitly assigns to documents. For requests "
            "about combined totals by payment status, use the status_summary "
            "operation for the requested status. Do not invent document facts.\n\n"
            f"Original question:\n{question}"
        )

    @staticmethod
    def _document_task(question: str) -> str:
        """Keep retrieval focused on facts supported by source documents."""
        return (
            "Answer only the part of this request that requires information "
            "from the document collection. Include only facts that directly "
            "answer the question. Do not add unrelated facts merely because "
            "they appear in a retrieved chunk. Do not calculate or infer totals "
            "from the structured invoice dataset. Do not compare document "
            "figures with structured dataset figures. Cite every factual "
            "claim using the retrieved sources.\n\n"
            f"Original question:\n{question}"
        )

    @staticmethod
    def _verified_status_summary(result: AgentResult) -> str | None:
        """Render a status aggregate from successful SQL output, not model prose."""
        for step in result.steps:
            if (
                step.action is None
                or step.action.action != "call_tool"
                or step.action.tool != "invoice_query"
                or step.action.arguments.get("operation") != "status_summary"
                or step.outcome is None
                or not step.outcome.ok
                or step.outcome.truncated
            ):
                continue

            try:
                payload = json.loads(step.outcome.output)
                status = payload["status"]
                requested_status = step.action.arguments.get("status")
                invoice_count = payload["invoice_count"]
                totals = payload["totals_by_currency"]
                if (
                    not isinstance(status, str)
                    or status != requested_status
                    or not isinstance(invoice_count, int)
                    or isinstance(invoice_count, bool)
                    or invoice_count < 0
                    or not isinstance(totals, list)
                    or not totals
                ):
                    return None

                parts = []
                for item in totals:
                    currency = item["currency"]
                    count = item["invoice_count"]
                    amount = Decimal(item["total"])
                    if (
                        not isinstance(currency, str)
                        or not currency.strip()
                        or not isinstance(count, int)
                        or isinstance(count, bool)
                        or count < 0
                        or not amount.is_finite()
                    ):
                        return None
                    parts.append(f"{currency} {amount:,.2f} ({count} invoice(s))")

                return (
                    f"Structured dataset: {invoice_count} {status} invoice(s); "
                    f"totals by currency: {'; '.join(parts)}."
                )
            except (KeyError, TypeError, ValueError, InvalidOperation):
                return None

        return None

    @staticmethod
    def _compute_answer(result: AgentResult) -> str:
        """Describe computation only when its executed tools succeeded."""
        failed_tools = [
            step.outcome
            for step in result.steps
            if step.outcome is not None and not step.outcome.ok
        ]
        if failed_tools:
            return (
                "Invoice computation failed because a tool call was unsuccessful; "
                "no verified result is available."
            )

        if result.status == "answered" and result.answer:
            return result.answer

        return f"Invoice computation did not produce a final answer (status: {result.status})."
