"""Day 2 demo: run one prompt through both providers and print the metrics."""

import asyncio

from llmkit.client import LLMError, LLMResult, make_client

MESSAGES = [{"role": "user", "content": "Explain what a vector database is in 3 sentences."}]


def _fmt(value: float | None, unit: str = "s", digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{unit}"


def show(r: LLMResult) -> None:
    print(f"\n=== {r.provider} / {r.model} ===")
    print(r.text.strip())
    print("-" * 60)
    print(f"TTFT (visible)  : {_fmt(r.ttft_s)}")
    print(f"TTFT (any token): {_fmt(r.ttft_any_s)}")
    print(f"Total time      : {_fmt(r.total_s)}")
    print(f"Tokens/sec      : {_fmt(r.tokens_per_s, '', 1)}")
    print(
        f"Tokens          : prompt={r.usage.prompt_tokens} "
        f"completion={r.usage.completion_tokens} reasoning={r.usage.reasoning_tokens}"
    )
    print(f"Cost            : ${r.cost_usd:.6f}  (estimated usage: {r.usage_estimated})")


async def run(provider: str) -> None:
    async with make_client(provider) as client:
        # Warm-up so model loading / connection setup is not counted in the measurement.
        await client.generate([{"role": "user", "content": "Hi"}], max_tokens=16)
        result = await client.generate(MESSAGES, max_tokens=1024)
    show(result)


async def main() -> None:
    for provider in ("ollama", "cloud"):
        try:
            await run(provider)
        except LLMError as e:
            print(f"\n[{provider}] FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
