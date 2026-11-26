"""Day 4 demo: document extraction, provider comparison, logs and exports."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from llmkit.client import make_client
from llmkit.exporters import invoices_to_frames, write_sqlite
from llmkit.logs import events_to_frame, parse_log
from llmkit.pipeline import run_batch

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "samples" / "day4"


def print_report(provider: str, report) -> None:
    summary = report.summary()
    print(f"\n{'=' * 68}")
    print(f"PROVIDER: {provider.upper()}")
    print("=" * 68)
    print(
        f"Documents: {summary['total']} | "
        f"Successful: {summary['ok']} | "
        f"Failed: {summary['failed']} | "
        f"Fallbacks: {summary['used_fallback']} | "
        f"Cost: ${summary['total_cost_usd']:.6f}"
    )

    for outcome in report.outcomes:
        print(f"\n[{outcome.status.upper()}] {Path(outcome.source).name}")
        print(
            f"  Method: {outcome.method} | "
            f"Provider: {outcome.provider or 'n/a'} | "
            f"Latency: {outcome.latency_s:.2f}s | "
            f"Cost: ${outcome.cost_usd:.6f}"
        )
        if outcome.invoice:
            invoice = outcome.invoice
            print(
                f"  Invoice: {invoice.invoice_number} | "
                f"{invoice.vendor_name} | "
                f"{invoice.currency.value} {invoice.total:.2f} | "
                f"{invoice.status.value} | "
                f"{len(invoice.line_items)} line items"
            )
        if outcome.attempts:
            for attempt in outcome.attempts:
                print(
                    f"  Attempt: {attempt.role} / {attempt.provider} / "
                    f"#{attempt.attempt} / {attempt.outcome}"
                )
        if outcome.note:
            print(f"  Note: {outcome.note}")
        if outcome.error:
            print(f"  Error stage: {outcome.error_stage}")
            print(f"  Error: {outcome.error}")


def export_results(provider: str, report, out_dir: Path) -> None:
    provider_dir = out_dir / provider
    provider_dir.mkdir(parents=True, exist_ok=True)

    invoices_df, items_df = invoices_to_frames(report.invoices)
    invoices_path = provider_dir / "invoices.csv"
    items_path = provider_dir / "line_items.csv"
    invoices_df.to_csv(invoices_path, index=False)
    items_df.to_csv(items_path, index=False)

    if report.invoices:
        db_path = provider_dir / "invoices.sqlite"
        counts = write_sqlite(report.invoices, db_path)
        print(
            f"\nExports ({provider}): {invoices_path}, {items_path}, "
            f"{db_path} | SQLite inserted={counts['inserted']}, "
            f"skipped={counts['skipped']}"
        )
    else:
        print(f"\nExports ({provider}): empty CSV files written; no invoices for SQLite.")


async def run_provider(provider: str, out_dir: Path) -> None:
    client = make_client(provider)
    try:
        report = await run_batch(
            [
                SAMPLES / "invoices.csv",
                SAMPLES / "messy_invoice.txt",
                SAMPLES / "empty_document.txt",
            ],
            client,
            client,
            concurrency=2,
            max_attempts=1,
            fallback_max_attempts=1,
            base_delay_s=0,
        )
        print_report(provider, report)
        export_results(provider, report, out_dir)
    finally:
        await client.aclose()


def run_log_demo(out_dir: Path) -> None:
    log_path = SAMPLES / "application.log"
    result = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
    frame = events_to_frame(result)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / "parsed_events.csv"
    frame.to_csv(output_path, index=False)

    print(f"\n{'=' * 68}")
    print("REGEX-FIRST LOG PARSING")
    print("=" * 68)
    print(f"Regex events: {len(result.parsed) - result.recovered}")
    print(f"Unresolved lines: {len(result.unparsed)}")
    print(f"CSV export: {output_path}")

    for item in result.parsed:
        print(
            f"  Line {item.line_no}: {item.event.level.value} | "
            f"{item.event.message.replace(chr(10), ' / ')}"
        )
    for line_no, line in result.unparsed:
        print(f"  Unresolved line {line_no}: {line}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Day 4 document pipeline demo")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["ollama", "cloud"],
        default=["ollama", "cloud"],
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "day4")
    args = parser.parse_args()

    if not SAMPLES.is_dir():
        raise SystemExit(f"Sample directory not found: {SAMPLES}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for provider in args.providers:
        await run_provider(provider, args.out_dir)

    run_log_demo(args.out_dir)


if __name__ == "__main__":
    asyncio.run(main())
