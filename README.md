# AI Engineer 15-Day Assignment

Applied Generative AI, RAG & Agentic Workflows.

**Author:** Abhinav Talluri

## Setup

1. Create and activate a virtual environment: `python3 -m venv .venv` and `source .venv/bin/activate`.
2. Install dependencies: `pip install -e ".[dev]"`.
3. Copy `.env.example` to `.env` and configure your cloud API key if needed.
4. Never commit `.env`.

## Project Progress

- [x] Day 1: Environment, configuration, secrets, linting and async smoke test
- [x] Day 2: Unified async LLM client and local-vs-cloud benchmarking
- [x] Day 3: Structured extraction, Pydantic validation and provider fallback
- [x] Day 4: Document loaders, preprocessing, resilient batch pipeline, log parsing and exports

## Day 4: Document Processing Pipeline

Features include PDF, CSV, TSV, TXT and LOG loading; text preprocessing; deterministic table extraction; schema-driven LLM extraction with Pydantic validation; retry and fallback handling; per-document error isolation; regex-first log parsing; Pandas CSV exports; and SQLite storage.

### Run the demo

- Local: `python -m llmkit.pipeline_demo --providers ollama`
- Cloud: `python -m llmkit.pipeline_demo --providers cloud`
- Both: `python -m llmkit.pipeline_demo --providers ollama cloud`

The demo reads fixtures from `samples/day4/` and writes generated files to `output/day4/`. Generated outputs are excluded from Git.

See `docs/day4_demo_report.md` for observed results.

## Tests

- Run tests: `pytest -q`
- Run lint: `ruff check .`
