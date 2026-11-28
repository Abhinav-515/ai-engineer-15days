"""Tests for sliding-window conversation memory."""

from __future__ import annotations

import pytest

from llmkit.memory import ConversationMemory, Turn


def test_starts_empty() -> None:
    memory = ConversationMemory()

    assert memory.is_empty()
    assert len(memory) == 0
    assert memory.turns == []
    assert memory.as_messages() == []
    assert memory.format_for_prompt() == ""


def test_add_turn_preserves_order() -> None:
    memory = ConversationMemory(max_turns=3)
    memory.add_turn("q1", "a1")
    memory.add_turn("q2", "a2")

    assert memory.turns == [Turn("q1", "a1"), Turn("q2", "a2")]
    assert len(memory) == 2
    assert not memory.is_empty()


def test_sliding_window_evicts_oldest_turns() -> None:
    memory = ConversationMemory(max_turns=2)
    for i in range(1, 5):
        memory.add_turn(f"q{i}", f"a{i}")

    assert len(memory) == 2
    assert memory.turns == [Turn("q3", "a3"), Turn("q4", "a4")]


def test_window_never_exceeds_max_turns() -> None:
    memory = ConversationMemory(max_turns=3)
    for i in range(100):
        memory.add_turn(f"q{i}", f"a{i}")

    assert len(memory) == 3


def test_as_messages_alternates_user_and_assistant() -> None:
    memory = ConversationMemory()
    memory.add_turn("q1", "a1")
    memory.add_turn("q2", "a2")

    assert memory.as_messages() == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_format_for_prompt_renders_plain_text() -> None:
    memory = ConversationMemory()
    memory.add_turn("What is the status?", "Unpaid.")
    memory.add_turn("And the total?", "USD 110.")

    assert memory.format_for_prompt() == (
        "User: What is the status?\nAssistant: Unpaid.\nUser: And the total?\nAssistant: USD 110."
    )


def test_turns_property_returns_copy() -> None:
    memory = ConversationMemory()
    memory.add_turn("q1", "a1")

    snapshot = memory.turns
    snapshot.clear()

    assert len(memory) == 1


def test_add_turn_strips_whitespace() -> None:
    memory = ConversationMemory()
    memory.add_turn("  q1  ", "\na1\n")

    assert memory.turns == [Turn("q1", "a1")]


@pytest.mark.parametrize(("question", "answer"), [("", "a"), ("   ", "a"), ("q", ""), ("q", "  ")])
def test_add_turn_rejects_empty_text(question: str, answer: str) -> None:
    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.add_turn(question, answer)

    assert memory.is_empty()


@pytest.mark.parametrize("max_turns", [0, -1])
def test_rejects_non_positive_max_turns(max_turns: int) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        ConversationMemory(max_turns=max_turns)


def test_clear_empties_history() -> None:
    memory = ConversationMemory()
    memory.add_turn("q1", "a1")
    memory.clear()

    assert memory.is_empty()


def test_separate_instances_do_not_share_state() -> None:
    first = ConversationMemory()
    second = ConversationMemory()
    first.add_turn("q1", "a1")

    assert second.is_empty()
