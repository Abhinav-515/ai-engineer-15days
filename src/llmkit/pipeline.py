"""Batch pipeline: files -> load -> clean -> extract (retry + fallback) -> outcomes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from llmkit.attempts import AttemptRecord
from llmkit.client import BaseLLMClient
from llmkit.loaders import DocumentError, load_document
from llmkit.preprocess import clean_text, truncate_text
from llmkit.resilience import FallbackError, extract_with_fallback
from llmkit.schemas import Invoice


@dataclass
class DocumentOutcome:
    source: str
    kind: str
    status: str
    invoice: Invoice | None = None
    provider: str | None = None
    used_fallback: bool = False
    attempts: list[AttemptRecord] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_s: float = 0.0
    truncated: bool = False
    error: str | None = None
    error_stage: str | None = None


@dataclass
class BatchReport:
    outcomes: list[DocumentOutcome]

    @property
    def invoices(self) -> list[Invoice]:
        return [o.invoice for o in self.outcomes if o.invoice is not None]

    def summary(self) -> dict[str, float | int]:
        ok = [o for o in self.outcomes if o.status == "ok"]
        return {
            "total": len(self.outcomes),
            "ok": len(ok),
            "failed": len(self.outcomes) - len(ok),
            "used_fallback": sum(1 for o in ok if o.used_fallback),
            "truncated": sum(1 for o in self.outcomes if o.truncated),
            "total_cost_usd": round(sum(o.cost_usd for o in self.outcomes), 6),
        }


def _kind_from_name(path: Path) -> str:
    return path.suffix.lstrip(".").lower() or "unknown"


async def process_document(
    path: str | Path,
    primary: BaseLLMClient,
    fallback: BaseLLMClient,
    *,
    max_chars: int = 6000,
    **retry_options,
) -> DocumentOutcome:
    """Process one file; never raises, so one bad document cannot stop a batch."""
    path = Path(path)
    source = str(path)
    kind = _kind_from_name(path)
    try:
        try:
            raw = await load_document(path)
        except DocumentError as error:
            return DocumentOutcome(
                source=source, kind=kind, status="failed", error=str(error), error_stage="load"
            )

        kind = raw.kind
        text, truncated = truncate_text(clean_text(raw.text), max_chars)
        options = {"jitter_ratio": 0.25, "fallback_max_attempts": 2, **retry_options}
        try:
            result = await extract_with_fallback(primary, fallback, text, Invoice, **options)
        except FallbackError as error:
            return DocumentOutcome(
                source=source,
                kind=kind,
                status="failed",
                attempts=error.attempts,
                truncated=truncated,
                error=str(error),
                error_stage="extract",
            )

        return DocumentOutcome(
            source=source,
            kind=kind,
            status="ok",
            invoice=result.data,
            provider=result.provider,
            used_fallback=any(a.role == "fallback" for a in result.attempts),
            attempts=result.attempts,
            cost_usd=result.cost_usd,
            latency_s=result.latency_s,
            truncated=truncated,
        )
    except Exception as error:
        return DocumentOutcome(
            source=source,
            kind=kind,
            status="failed",
            error=f"{type(error).__name__}: {error}",
            error_stage="unexpected",
        )


async def run_batch(
    paths: Iterable[str | Path],
    primary: BaseLLMClient,
    fallback: BaseLLMClient,
    *,
    concurrency: int = 2,
    **options,
) -> BatchReport:
    """Process files with bounded concurrency; outcomes keep the input order."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(path: str | Path) -> DocumentOutcome:
        async with semaphore:
            return await process_document(path, primary, fallback, **options)

    outcomes = await asyncio.gather(*(run_one(p) for p in paths))
    return BatchReport(outcomes=list(outcomes))
