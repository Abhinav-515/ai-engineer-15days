import json

import httpx
import pytest

from llmkit.client import LLMError, OllamaClient, OpenAICompatClient

MSGS = [{"role": "user", "content": "hi"}]


async def test_ollama_parses_ndjson_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        lines = [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 2},
        ]
        body = "\n".join(json.dumps(x) for x in lines) + "\n"
        return httpx.Response(200, content=body.encode())

    async with OllamaClient(
        "m", base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        r = await client.generate(MSGS)

    assert r.text == "Hello"
    assert r.usage.prompt_tokens == 10
    assert r.usage.completion_tokens == 2
    assert r.cost_usd == 0
    assert r.ttft_s is not None
    assert not r.usage_estimated


async def test_cloud_parses_sse_reasoning_usage_and_cost():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["reasoning_effort"] == "low"
        events = [
            {"choices": [{"delta": {"reasoning": "think"}}]},
            {"choices": [{"delta": {"content": "Hi"}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 2000,
                    "completion_tokens_details": {"reasoning_tokens": 1500},
                },
            },
        ]
        sse = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode())

    async with OpenAICompatClient(
        "m",
        base_url="http://test/v1",
        price_in_per_m=1.0,
        price_out_per_m=2.0,
        reasoning_effort="low",
        transport=httpx.MockTransport(handler),
    ) as client:
        r = await client.generate(MSGS)

    assert r.text == "Hi"
    assert r.reasoning_text == "think"
    assert r.usage.reasoning_tokens == 1500
    assert r.cost_usd == pytest.approx(0.005)  # 1000*1/1e6 + 2000*2/1e6
    assert r.ttft_any_s is not None and r.ttft_s is not None
    assert r.ttft_any_s <= r.ttft_s  # reasoning arrives before the visible answer


async def test_http_error_includes_response_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "model_not_found"}})

    async with OpenAICompatClient(
        "bad", base_url="http://test/v1", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(LLMError, match="model_not_found"):
            await client.generate(MSGS)
