"""Tests that evaluation results record the context the judge was given."""

import json

from llmkit.eval_cases import EvalCase
from llmkit.eval_harness import evaluate_case, run_evaluation, write_report
from llmkit.rag import RAGResponse

CASE = EvalCase(id="u", question="Q?", answerable=False, category="out_of_scope")


class FakeRag:
    def __init__(self, response):
        self.response = response

    async def ask(self, question):
        return self.response


def make_response(context):
    return RAGResponse(
        answer="A.",
        citations=[],
        abstained=False,
        retrieved_count=0,
        context=context,
    )


async def test_case_result_records_the_context_sent_to_the_model():
    result = await evaluate_case(CASE, FakeRag(make_response("CTX-TEXT")))

    assert result.context == "CTX-TEXT"


async def test_report_json_includes_context(tmp_path):
    report = await run_evaluation([CASE], FakeRag(make_response("CTX-TEXT")))

    path = write_report(report, tmp_path / "report.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cases"][0]["context"] == "CTX-TEXT"
