"""Sliding-window conversation memory for multi-turn RAG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from llmkit.client import Messages


@dataclass(frozen=True)
class Turn:
    """One completed exchange: a user question and the assistant's answer."""

    question: str
    answer: str


class ConversationMemory:
    """Keep only the most recent ``max_turns`` exchanges of one conversation.

    Each conversation owns its own instance, so there is no shared global state.
    Old turns are discarded automatically, which bounds prompt size.
    """

    def __init__(self, max_turns: int = 5) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str) -> None:
        """Record a completed exchange, evicting the oldest one if the window is full."""
        question = question.strip()
        answer = answer.strip()
        if not question:
            raise ValueError("question must not be empty")
        if not answer:
            raise ValueError("answer must not be empty")
        self._turns.append(Turn(question=question, answer=answer))

    @property
    def turns(self) -> list[Turn]:
        """Return a copy of the retained turns, oldest first."""
        return list(self._turns)

    def is_empty(self) -> bool:
        return not self._turns

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def as_messages(self) -> Messages:
        """Return history as alternating user/assistant chat messages."""
        messages: Messages = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def format_for_prompt(self) -> str:
        """Render history as plain text, e.g. for a query-rewriting prompt."""
        return "\n".join(f"User: {turn.question}\nAssistant: {turn.answer}" for turn in self._turns)
