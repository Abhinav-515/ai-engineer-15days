"""Day 3 demo: resilient invoice extraction with validation and provider fallback."""

import asyncio

from llmkit.client import make_client
from llmkit.resilience import FallbackError, extract_with_fallback
from llmkit.samples import SAMPLES
from llmkit.schemas import Invoice


def check_invoice(inv: Invoice, expected: dict) -> list[str]:
    """Return a list of fields that do not match the expected values."""
    errors = []

    if inv.invoice_number != expected["invoice_number"]:
        errors.append("invoice_number")

    if abs(inv.total - expected["total"]) >= 0.01:
        errors.append("total")

    if inv.currency.value != expected["currency"]:
        errors.append("currency")

    actual_due_date = inv.due_date.isoformat() if inv.due_date is not None else None
    if actual_due_date != expected["due_date"]:
        errors.append("due_date")

    if inv.status.value != expected["status"]:
        errors.append("status")

    return errors


async def main() -> None:
    print("\n=== Resilient extraction: Ollama -> Cloud fallback ===")

    async with make_client("ollama") as primary:
        async with make_client("cloud") as fallback:
            for sample in SAMPLES:
                try:
                    res = await extract_with_fallback(
                        primary,
                        fallback,
                        sample["text"],
                        Invoice,
                        max_attempts=2,
                        base_delay_s=0.5,
                    )
                except FallbackError as error:
                    print(f"[{sample['name']}] FAILED: both providers failed")
                    print(f"    primary: {str(error.primary_error)[:250]}")
                    print(f"    fallback: {str(error.fallback_error)[:250]}")
                    continue

                inv = res.data
                errors = check_invoice(inv, sample["expected"])
                status = "OK" if not errors else "MISMATCH"

                print(
                    f"[{sample['name']}] {status} | "
                    f"source={res.provider} | "
                    f"native_schema={res.used_native_schema} | "
                    f"{res.latency_s:.2f}s | "
                    f"{inv.vendor_name} | "
                    f"{inv.invoice_date} -> {inv.due_date} | "
                    f"{inv.currency.value} {inv.total} | "
                    f"{inv.status.value} | "
                    f"{len(inv.line_items)} items"
                )

                if errors:
                    print(f"    CHECK FAILED: {', '.join(errors)}")


if __name__ == "__main__":
    asyncio.run(main())
