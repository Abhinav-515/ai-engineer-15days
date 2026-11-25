"""Day 3 demo: extract the sample invoices with each provider and check against expectations."""

import asyncio

from llmkit.client import LLMError, make_client
from llmkit.extract import ExtractionError, extract
from llmkit.samples import SAMPLES
from llmkit.schemas import Invoice


async def run(provider: str) -> None:
    print(f"\n=== {provider} ===")
    async with make_client(provider) as client:
        await client.generate([{"role": "user", "content": "Hi"}], max_tokens=16)  # warm-up
        for sample in SAMPLES:
            try:
                res = await extract(client, sample["text"], Invoice)
            except ExtractionError as e:
                print(f"[{sample['name']}] INVALID: {str(e)[:300]}")
                print(f"    raw: {e.raw_text[:200]!r}")
                continue
            except LLMError as e:
                print(f"[{sample['name']}] PROVIDER ERROR: {str(e)[:300]}")
                continue
            inv, exp = res.data, sample["expected"]
            ok = (
                inv.invoice_number == exp["invoice_number"]
                and abs(inv.total - exp["total"]) < 0.01
                and inv.currency.value == exp["currency"]
            )
            print(
                f"[{sample['name']}] {'OK' if ok else 'MISMATCH'} | native_schema="
                f"{res.used_native_schema} | {res.latency_s:.2f}s | "
                f"{inv.vendor_name} | {inv.invoice_date} -> {inv.due_date} | "
                f"{inv.currency.value} {inv.total} | {inv.status.value} | "
                f"{len(inv.line_items)} items"
            )


async def main() -> None:
    for provider in ("ollama", "cloud"):
        await run(provider)


if __name__ == "__main__":
    asyncio.run(main())
