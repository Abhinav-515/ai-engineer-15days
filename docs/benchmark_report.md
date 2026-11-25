# Local vs Cloud LLM Benchmark

- Date: 20261125
- Local hardware: MacBook Air M1, 8 GB
- Runs per task: 3 (after one warm-up call per provider)
- Tasks: 8 across several categories

## Results

| Metric | ollama (llama3.2:3b) | cloud (openai/gpt-oss-20b) |
|---|---|---|
| Runs (errors) | 24 (0) | 24 (0) |
| Auto-check pass rate | 88% | 100% |
| LLM-judge score, mean (1-5) | 4.38 | 5.00 |
| TTFT visible, mean (s) | 0.102 | 0.584 |
| TTFT visible, p95 (s) | 0.257 | 0.718 |
| Total latency, mean (s) | 1.07 | 0.61 |
| Decode tokens/sec, mean (answers >= 80 tokens) | 27.6 | 1316.9 |
| Reasoning tokens (% of output) | 0% | 31% |
| Cost per 1,000 requests (USD) | $0.0000 | $0.0380 |

## Per-task auto-check results (passed / runs)

| Task | ollama | cloud |
|---|---|---|
| capital | 3/3 | 3/3 |
| math | 3/3 | 3/3 |
| json | 3/3 | 3/3 |
| list | 3/3 | 3/3 |
| ball | 0/3 | 3/3 |
| summary | 3/3 | 3/3 |
| code | 3/3 | 3/3 |
| explain | 3/3 | 3/3 |

## Methodology

- Requests run sequentially so providers do not compete for resources.
- TTFT is measured to the first visible answer token (reasoning tokens excluded).
- Decode tokens/sec = (completion_tokens - 1) / (total time - time to first token),
  averaged only over answers with >= 80 tokens (shorter ones are too brief to time).
- HTTP 429 rate-limit responses are retried with backoff; other errors count as failures.
- Cost = token usage x configured per-million prices; local inference is $0.
- Quality: automatic pass/fail checks on every run (primary), plus an LLM judge
  scoring the first run of each task (secondary).

## Caveats

- Few runs per task: p95 is effectively the maximum and not statistically strong.
- Models differ in size (3B local vs 20B cloud), so this compares deployments,
  not architectures.
- The cloud model acts as judge, which can favor its own style.
- Cloud latency includes network distance from the test location.
- Reasoning tokens are billed as output tokens.

## Findings

- **Quality:** the cloud model (gpt-oss-20b) passed 24/24 automatic checks; the local
  llama3.2:3b passed 21/24. The entire gap is the bat-and-ball reasoning task (0/3 local,
  3/3 cloud), a consistent weakness of the 3B model rather than noise. The LLM-judge scores
  (4.38 vs 5.00) agree, but the cloud model is also the judge, so they are secondary.
- **First-token latency:** local is about 5.7x faster (0.102 s vs 0.584 s mean TTFT)
  because there is no network round trip. For interactive, short-answer use, local feels
  more immediate.
- **Total latency:** cloud finishes sooner overall (0.61 s vs 1.07 s mean), and the gap
  grows with answer length (about 4.5 s local vs about 0.6 s cloud on the 100-word task).
- **Decode speed:** local measured 27.6 tokens/s. Cloud decode is far faster, but the timing
  window on short answers is too small for a precise figure (reported 1316.9 tokens/s),
  so treat that number as indicative only.
- **Reasoning overhead:** about 31% of cloud output tokens were hidden reasoning, which is
  billed and adds to latency even at low reasoning effort.
- **Cost:** local inference is $0. Cloud cost is about $0.038 per 1,000 requests at the
  configured per-token prices, which were not re-verified against the provider's current
  price list, so this figure is indicative.
- **Recommendation:** use local for privacy-sensitive, low-latency or offline work where a
  3B model's quality is sufficient; use cloud when reasoning quality or long-answer speed
  matters. Results are from one machine, 8 tasks and 3 runs each, so differences of a few
  percent should not be over-read.
