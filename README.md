# AI Engineer 15-Day Assignment

Applied Generative AI, RAG & Agentic Workflows. Author: Abhinav Talluri.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then add your keys
pre-commit install
pytest
```

## Progress

- [x] Day 1: environment, config, secrets, linting, async smoke test
- [ ] Day 2: unified async LLM client + local vs cloud benchmark
