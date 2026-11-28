# Day 4: Document Pipeline Demo Report

## Objective

Demonstrate document loading, invoice extraction, validation, batch error isolation, log parsing and structured exports using local and cloud providers.

## Sample inputs

The demo uses four fixtures in `samples/day4/`:

- `invoices.csv`: structured invoice table
- `messy_invoice.txt`: unstructured invoice with three line items
- `empty_document.txt`: empty input for load-error handling
- `application.log`: timestamped events, a continuation line and an unstructured line

## Observed results

Three demo runs were performed for each provider. The same sample documents were used in each run.

| Metric | Ollama | Cloud |
|---|---:|---:|
| Documents processed per run | 3 | 3 |
| Successful documents per run | 2 | 2 |
| Expected load failures per run | 1 | 1 |
| Cross-provider fallbacks | 0 | 0 |
| Messy invoice extraction latency, run 1 | 13.34 s | 1.04 s |
| Messy invoice extraction latency, run 2 | 7.90 s | 0.82 s |
| Messy invoice extraction latency, run 3 | 7.54 s | 0.74 s |
| Reported extraction cost per run | $0.000000 | $0.000183 |

These are observations from three demo runs on the same machine and sample, not a statistically significant benchmark. The median observed extraction latency was 7.90 seconds for Ollama and 0.82 seconds for cloud.

## Extraction results

Both providers extracted the same messy invoice:

- Invoice number: `NS-2026-204`
- Vendor: `NORTHSTAR OFFICE SOLUTIONS`
- Subtotal: USD 220.00
- Tax: USD 22.00
- Total: USD 242.00
- Payment status: unpaid
- Line items: 3

The structured CSV invoice was parsed deterministically without an LLM call. Both provider runs produced identical invoice exports containing the two valid invoices.

## Error isolation

The empty document was rejected during loading. It did not stop the other documents from being processed or exported.

The demo passes the same provider client as both primary and fallback clients to isolate provider performance. Therefore, these runs do not measure cross-provider fallback behavior. Production fallback behavior is covered separately by the test suite.

## Log parsing

The regex-first parser identified four timestamped events. It attached the indented retry message to the preceding error event and left one unstructured diagnostic line unresolved.

## Exports and idempotency

Each provider run generated `invoices.csv`, `line_items.csv` and `invoices.sqlite`. The log demo generated `parsed_events.csv`.

Repeated runs reported zero new SQLite inserts and skipped two existing invoices, confirming that duplicate records were not inserted.

Generated files are written under `output/day4/` and excluded from version control.

## Validation

At the time of this demo:

- Pytest: 117 tests passed
- Ruff: all checks passed

## Notes and limitations

- Latency figures represent three runs against one LLM-based invoice sample.
- The CSV invoice uses deterministic parsing, so it is not a provider comparison.
- The empty-document failure is an expected negative test.
- Cloud cost is the amount reported by the configured client; actual billing can vary with provider pricing and token usage.
- The unresolved log line is retained rather than assigned an invented timestamp or severity.
