"""Records of individual provider attempts, used for batch reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptRecord:
    role: str
    provider: str
    attempt: int
    outcome: str
    error_type: str | None = None
    status_code: int | None = None
    delay_s: float | None = None
