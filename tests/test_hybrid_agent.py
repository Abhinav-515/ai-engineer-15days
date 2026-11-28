"""Tests for hybrid compute and retrieval orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llmkit.agent import AgentAction, AgentResult, AgentStep
from llmkit.hybrid_agent import HybridAgent
from llmkit.rag import RAGResponse
from llmkit.router import RouterResult
from llmkit.sandbox import ToolOutcome


class FakeRouter:
    def __init__(self, route: str) -> None:
        self.selected_route = route
        self.calls: list[str] = []

    async def route_question(self, question: str) -> RouterResult:
        self.calls.append(question)
        return RouterResult(
            route=self.selected_route,
            confidence=0.99,
            reason="Test route",
        )

    async def route(self, question: str) -> RouterResult:
        return await self.route_question(question)


class FakeToolAgent:
    def __init__(self, result: AgentResult | None = None) -> None:
        self.result = result or AgentResult("answered", "Invoice total is 132.00.")
        self.calls: list[str] = []

    async def run(self, task: str) -> AgentResult:
        self.calls.append(task)
        return self.result


class FakeRAGService:
    def __init__(self, result: RAGResponse | None = None) -> None:
        self.result = result or RAGResponse(
            answer="The policy document supports this.",
            citations=[],
            abstained=False,
            retrieved_count=1,
        )
        self.calls: list[str] = []

    async def ask(self, question: str) -> RAGResponse:
        self.calls.append(question)
        return self.result


def make_agent(route: str):
    router = FakeRouter(route)
    tool_agent = FakeToolAgent()
    rag_service = FakeRAGService()
    return HybridAgent(router, tool_agent, rag_service), router, tool_agent, rag_service


@pytest.mark.parametrize("route", ["compute", "retrieve", "both"])
async def test_router_decision_is_preserved(route: str) -> None:
    agent, _, _, _ = make_agent(route)

    result = await agent.ask("Test question")

    assert result.route == route
    assert result.routing.route == route


async def test_compute_route_calls_only_tool_agent() -> None:
    agent, router, tool_agent, rag_service = make_agent("compute")

    result = await agent.ask("What is the invoice total?")

    assert result.answer == "Invoice total is 132.00."
    assert result.compute_result is tool_agent.result
    assert result.rag_result is None
    assert result.citations == []
    assert tool_agent.calls == ["What is the invoice total?"]
    assert rag_service.calls == []
    assert router.calls == ["What is the invoice total?"]


async def test_retrieve_route_calls_only_rag() -> None:
    agent, _, tool_agent, rag_service = make_agent("retrieve")

    result = await agent.ask("Explain the payment policy.")

    assert result.answer == rag_service.result.answer
    assert result.rag_result is rag_service.result
    assert result.compute_result is None
    assert tool_agent.calls == []
    assert len(rag_service.calls) == 1
    assert "only facts that directly" in rag_service.calls[0]
    assert "Do not add unrelated facts" in rag_service.calls[0]
    assert "Explain the payment policy." in rag_service.calls[0]


async def test_both_route_combines_results_and_preserves_citations() -> None:
    citation = SimpleNamespace(label="S1", document="policy.pdf")
    rag_result = RAGResponse(
        answer="The document says payment is due in 30 days [S1].",
        citations=[citation],
        abstained=False,
        retrieved_count=1,
    )
    agent, _, tool_agent, rag_service = make_agent("both")
    rag_service.result = rag_result

    result = await agent.ask("Calculate the invoice total and explain the policy.")

    assert result.route == "both"
    assert "INVOICE DATA" in result.answer
    assert "Invoice total is 132.00." in result.answer
    assert "DOCUMENT CONTEXT" in result.answer
    assert "[S1]" in result.answer
    assert result.citations == [citation]
    assert result.rag_result is rag_result
    assert result.compute_result is tool_agent.result
    assert len(tool_agent.calls) == 1
    assert len(rag_service.calls) == 1


async def test_both_route_scopes_compute_task_to_structured_dataset() -> None:
    agent, _, tool_agent, rag_service = make_agent("both")
    question = (
        "What is the total of invoice NS-2026-204 in the documents, "
        "and how does it compare with the combined total of unpaid invoices "
        "in the structured dataset?"
    )

    await agent.ask(question)

    assert len(tool_agent.calls) == 1
    assert "only the part" in tool_agent.calls[0]
    assert "status_summary" in tool_agent.calls[0]
    assert "NS-2026-204" in tool_agent.calls[0]
    assert "structured invoice dataset" in tool_agent.calls[0]
    assert len(rag_service.calls) == 1
    assert "document collection" in rag_service.calls[0]
    assert "Do not calculate" in rag_service.calls[0]
    assert "Do not compare" in rag_service.calls[0]
    assert question in rag_service.calls[0]


async def test_both_route_keeps_compute_answer_when_rag_abstains() -> None:
    agent, _, _, rag_service = make_agent("both")
    rag_service.result = RAGResponse(
        answer="I don't have enough information to answer.",
        citations=[],
        abstained=True,
        retrieved_count=0,
    )

    result = await agent.ask("Calculate this and find supporting policy.")

    assert "Invoice total is 132.00." in result.answer
    assert "I don't have enough information" in result.answer
    assert result.citations == []
    assert result.rag_result.abstained is True


async def test_failed_compute_is_reported_without_inventing_answer() -> None:
    agent, _, tool_agent, _ = make_agent("compute")
    tool_agent.result = AgentResult("llm_error", None)

    result = await agent.ask("Calculate the invoice total.")

    assert "did not produce a final answer" in result.answer
    assert "llm_error" in result.answer


async def test_compute_route_rejects_final_answer_after_tool_failure() -> None:
    agent, _, tool_agent, _ = make_agent("compute")
    action = AgentAction(
        action="call_tool",
        tool="invoice_query",
        arguments={"operation": "status_summary", "status": "overdue"},
    )
    failed_outcome = ToolOutcome(
        tool="invoice_query",
        ok=False,
        output="invalid_arguments",
        duration_s=0.01,
        error_kind="invalid_arguments",
    )
    tool_agent.result = AgentResult(
        "answered",
        "The payment gateway retry completed at 09:17:45.",
        [AgentStep(index=1, action=action, outcome=failed_outcome)],
    )

    result = await agent.ask("What happened to the payment gateway?")

    assert "no verified result" in result.answer
    assert "09:17:45" not in result.answer


def make_status_result(
    output: str,
    *,
    status: str = "unpaid",
    ok: bool = True,
    truncated: bool = False,
) -> AgentResult:
    action = AgentAction(
        action="call_tool",
        tool="invoice_query",
        arguments={"operation": "status_summary", "status": status},
    )
    outcome = ToolOutcome(
        tool="invoice_query",
        ok=ok,
        output=output,
        duration_s=0.01,
        truncated=truncated,
    )
    step = AgentStep(index=1, action=action, outcome=outcome)
    return AgentResult("answered", "The unpaid total is EUR 999.00.", [step])


def test_verified_status_summary_uses_sql_output_not_model_answer() -> None:
    result = make_status_result(
        '{"status":"unpaid","invoice_count":2,'
        '"totals_by_currency":[{"currency":"USD","invoice_count":2,"total":"85.80"}]}'
    )

    summary = HybridAgent._verified_status_summary(result)

    assert summary == (
        "Structured dataset: 2 unpaid invoice(s); totals by currency: USD 85.80 (2 invoice(s))."
    )
    assert "999.00" not in summary


def test_verified_status_summary_keeps_currencies_separate() -> None:
    result = make_status_result(
        '{"status":"unpaid","invoice_count":2,"totals_by_currency":['
        '{"currency":"EUR","invoice_count":1,"total":"55.00"},'
        '{"currency":"USD","invoice_count":1,"total":"85.80"}]}'
    )

    summary = HybridAgent._verified_status_summary(result)

    assert "EUR 55.00 (1 invoice(s))" in summary
    assert "USD 85.80 (1 invoice(s))" in summary
    assert "140.80" not in summary


@pytest.mark.parametrize(
    ("output", "ok", "truncated", "requested_status"),
    [
        (
            '{"status":"paid","invoice_count":1,"totals_by_currency":[{"currency":"USD","invoice_count":1,"total":"10.00"}]}',
            True,
            False,
            "unpaid",
        ),
        ("not json", True, False, "unpaid"),
        (
            '{"status":"unpaid","invoice_count":-1,"totals_by_currency":[{"currency":"USD","invoice_count":1,"total":"10.00"}]}',
            True,
            False,
            "unpaid",
        ),
        (
            '{"status":"unpaid","invoice_count":1,"totals_by_currency":[{"currency":"USD","invoice_count":1,"total":"NaN"}]}',
            True,
            False,
            "unpaid",
        ),
        (
            '{"status":"unpaid","invoice_count":1,"totals_by_currency":[{"currency":"USD","invoice_count":1,"total":"10.00"}]}',
            False,
            False,
            "unpaid",
        ),
        (
            '{"status":"unpaid","invoice_count":1,"totals_by_currency":[{"currency":"USD","invoice_count":1,"total":"10.00"}]}',
            True,
            True,
            "unpaid",
        ),
    ],
)
def test_verified_status_summary_rejects_untrusted_outcomes(
    output: str,
    ok: bool,
    truncated: bool,
    requested_status: str,
) -> None:
    result = make_status_result(
        output,
        ok=ok,
        truncated=truncated,
        status=requested_status,
    )

    assert HybridAgent._verified_status_summary(result) is None


async def test_empty_question_is_rejected_before_routing() -> None:
    agent, router, tool_agent, rag_service = make_agent("compute")

    with pytest.raises(ValueError, match="question"):
        await agent.ask("   ")

    assert router.calls == []
    assert tool_agent.calls == []
    assert rag_service.calls == []
