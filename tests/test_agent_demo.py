"""Tests for the Day 9 agent demo output."""

from llmkit.agent import AgentAction, AgentResult, AgentStep
from llmkit.agent_demo import print_result


def test_final_answer_with_stray_tool_is_displayed_as_final(capsys) -> None:
    action = AgentAction(
        action="final_answer",
        tool="calculator",
        arguments={},
        final_answer="6848",
    )
    result = AgentResult(
        status="answered",
        answer="6848",
        steps=[AgentStep(index=1, action=action)],
    )

    print_result(result)

    output = capsys.readouterr().out
    assert "step 1: final answer" in output
    assert "calculator(" not in output
    assert "-> skipped" not in output
    assert "answer: 6848" in output
