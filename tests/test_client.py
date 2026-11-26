import json

import httpx
import pytest

from llmkit.client import (
    LLMError,
    OllamaClient,
    OpenAICompatClient,
    _check,
    http_status,
    parse_retry_after,
)

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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5", 5.0),
        ("0", 0.0),
        ("1.5", 1.5),
        (" 7 ", 7.0),
        (None, None),
        ("", None),
        ("-3", None),
        ("inf", None),
        ("nan", None),
        ("Wed, 21 Oct 2026 07:28:00 GMT", None),
        ("soon", None),
    ],
)
def test_parse_retry_after(value, expected):
    assert parse_retry_after(value) == expected


def test_http_status_prefers_attribute():
    assert http_status(LLMError("HTTP 500 from x", status_code=429)) == 429


def test_http_status_falls_back_to_message():
    assert http_status(LLMError("HTTP 503 from x: down")) == 503


def test_http_status_is_none_without_status():
    assert http_status(LLMError("connection reset")) is None


def _response(status: int, **kwargs) -> httpx.Response:
    request = httpx.Request("POST", "http://test/v1/chat")
    return httpx.Response(status, request=request, **kwargs)


async def test_check_captures_status_and_retry_after():
    resp = _response(429, headers={"Retry-After": "7"}, content=b"slow down")
    with pytest.raises(LLMError) as exc_info:
        await _check(resp)
    error = exc_info.value
    assert error.status_code == 429
    assert error.retry_after_s == 7.0
    assert str(error) == "HTTP 429 from http://test/v1/chat: slow down"


async def test_check_without_retry_after_header():
    with pytest.raises(LLMError) as exc_info:
        await _check(_response(503, content=b"down"))
    assert exc_info.value.status_code == 503
    assert exc_info.value.retry_after_s is None


async def test_check_ignores_http_date_retry_after():
    resp = _response(
        429,
        headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"},
        content=b"slow down",
    )
    with pytest.raises(LLMError) as exc_info:
        await _check(resp)
    assert exc_info.value.retry_after_s is None


async def test_check_does_nothing_on_success():
    await _check(_response(200, content=b"ok"))
