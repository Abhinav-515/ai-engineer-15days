"""Day 1 smoke test: async streaming call to local Ollama."""

import asyncio
import json
import time

import httpx

from llmkit.config import get_settings


async def main() -> None:
    s = get_settings()
    payload = {
        "model": s.ollama_model,
        "prompt": "Explain async/await in one sentence.",
        "stream": True,
    }
    start = time.perf_counter()
    first_token_at: float | None = None
    chunks = 0

    async with httpx.AsyncClient(base_url=s.ollama_base_url, timeout=60) as client:
        async with client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    if first_token_at is None:
                        first_token_at = time.perf_counter() - start
                    chunks += 1
                    print(data["response"], end="", flush=True)

    total = time.perf_counter() - start
    print(f"\n\nTTFT: {first_token_at:.3f}s | total: {total:.3f}s | chunks: {chunks}")


if __name__ == "__main__":
    asyncio.run(main())
