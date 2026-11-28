"""Evaluation case schema and JSONL loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

Category = Literal["factual", "multi_chunk", "adversarial", "out_of_scope"]


class EvalCaseError(ValueError):
    """An evaluation dataset is malformed or unreadable."""


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return " ".join(text.split())


class EvalCase(BaseModel):
    """One evaluation question with evidence snippets that mark relevant chunks.

    Evidence is stored as short substrings rather than chunk indices, so labels
    stay valid when chunk sizes change in ablation experiments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    answerable: bool
    category: Category
    evidence: tuple[str, ...] = ()

    @field_validator("id", "question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = normalize_whitespace(value)
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("evidence")
    @classmethod
    def _clean_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(normalize_whitespace(snippet) for snippet in value)
        if any(not snippet for snippet in cleaned):
            raise ValueError("evidence snippets must not be blank")
        if len({snippet.casefold() for snippet in cleaned}) != len(cleaned):
            raise ValueError("evidence snippets must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def _check_answerability(self) -> EvalCase:
        if self.answerable and not self.evidence:
            raise ValueError("answerable cases require at least one evidence snippet")
        if not self.answerable and self.evidence:
            raise ValueError("unanswerable cases must not have evidence snippets")
        return self


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load and validate a JSONL file of evaluation cases."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalCaseError(f"cannot read {file_path}: {exc}") from exc

    cases: list[EvalCase] = []
    first_seen: dict[str, int] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except ValidationError as exc:
            raise EvalCaseError(f"{file_path}:{line_number}: invalid case: {exc}") from exc
        if case.id in first_seen:
            raise EvalCaseError(
                f"{file_path}:{line_number}: duplicate id {case.id!r} "
                f"(first seen on line {first_seen[case.id]})"
            )
        first_seen[case.id] = line_number
        cases.append(case)

    if not cases:
        raise EvalCaseError(f"{file_path}: no evaluation cases found")
    return cases
