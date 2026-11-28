"""Tests for SSE formatting and the streaming adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest

from llmkit.rag import Citation, RAGResponse, RAGStreamEvent
from llmkit.sse import SSE_HEADERS, format_sse, format_sse_comment, sse_stream


def parse(frames: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for frame in frames:
        assert frame.endswith("\n\n")
        lines = frame[:-2].split("\n")
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        assert len(lines) == 2
        parsed.append((lines[0][7:], json.loads(lines[1][6:])))
    return parsed


def make_final() -> RAGStreamEvent:
    citation = Citation("[S1]", "pricing.pdf", 0, "pdf", "recursive", "chunk-1")
    response = RAGResponse(
        answer="It costs $50 [S1].",
        citations=[citation],
        abstained=False,
        retrieved_count=1,
        provider="fake",
    )
    return RAGStreamEvent("final", response=response)


async def source(*events: RAGStreamEvent) -> AsyncGenerator[RAGStreamEvent, None]:
    for event in events:
        yield event


async def collect(gen: AsyncGenerator[str, None]) -> list[str]:
    return [frame async for frame in gen]


def test_headers_are_event_stream() -> None:
    assert SSE_HEADERS["Content-Type"].startswith("text/event-stream")
    assert SSE_HEADERS["Cache-Control"] == "no-cache"
    assert SSE_HEADERS["X-Accel-Buffering"] == "no"


def test_format_sse_basic_frame() -> None:
    assert format_sse("token", {"text": "hi"}) == 'event: token\ndata: {"text": "hi"}\n\n'


def test_format_sse_escapes_newlines_in_payload() -> None:
    frame = format_sse("token", {"text": "line1\nline2\r\n\n"})

    assert frame.count("\n") == 3  # event line, data line, blank line
    assert json.loads(frame.split("data: ")[1])["text"] == "line1\nline2\r\n\n"


def test_format_sse_keeps_unicode() -> None:
    assert "\u20b9500" in format_sse("token", {"text": "\u20b9500"})


@pytest.mark.parametrize("name", ["", "a\nb", "a\rb"])
def test_format_sse_rejects_bad_event_names(name: str) -> None:
    with pytest.raises(ValueError):
        format_sse(name, {})


def test_comment_frame() -> None:
    assert format_sse_comment("ping") == ": ping\n\n"
    assert format_sse_comment("a\nb") == ": a b\n\n"


async def test_tokens_final_and_done_in_order() -> None:
    frames = await collect(
        sse_stream(
            source(
                RAGStreamEvent("token", text="It costs "),
                RAGStreamEvent("token", text="$50 [S1]."),
                make_final(),
            )
        )
    )

    parsed = parse(frames)
    assert [name for name, _ in parsed] == ["token", "token", "final", "done"]
    assert parsed[0][1] == {"text": "It costs "}
    final = parsed[2][1]
    assert final["answer"] == "It costs $50 [S1]."
    assert final["citations"][0]["label"] == "[S1]"
    assert final["abstained"] is False


async def test_error_event_is_followed_by_done() -> None:
    frames = await collect(
        sse_stream(source(RAGStreamEvent("token", text="x"), RAGStreamEvent("error", error="bad")))
    )

    parsed = parse(frames)
    assert [name for name, _ in parsed] == ["token", "error", "done"]
    assert parsed[1][1] == {"message": "bad"}


async def test_nothing_is_emitted_after_final() -> None:
    frames = await collect(sse_stream(source(make_final(), RAGStreamEvent("token", text="late"))))

    assert [name for name, _ in parse(frames)] == ["final", "done"]


async def test_unexpected_exception_becomes_generic_error_event() -> None:
    async def broken() -> AsyncGenerator[RAGStreamEvent, None]:
        yield RAGStreamEvent("token", text="partial")
        raise RuntimeError("secret internal detail")

    frames = await collect(sse_stream(broken()))

    parsed = parse(frames)
    assert [name for name, _ in parsed] == ["token", "error", "done"]
    assert parsed[1][1] == {"message": "internal error: RuntimeError"}
    assert "secret" not in "".join(frames)


async def test_final_without_response_is_malformed_error() -> None:
    frames = await collect(sse_stream(source(RAGStreamEvent("final"))))

    parsed = parse(frames)
    assert [name for name, _ in parsed] == ["error", "done"]
    assert parsed[0][1] == {"message": "malformed stream event"}


async def test_frames_are_yielded_before_source_finishes() -> None:
    finished = False

    async def slow() -> AsyncGenerator[RAGStreamEvent, None]:
        nonlocal finished
        yield RAGStreamEvent("token", text="a")
        yield RAGStreamEvent("token", text="b")
        finished = True

    gen = sse_stream(slow())
    first = await gen.__anext__()

    assert parse([first])[0][0] == "token"
    assert finished is False
    await gen.aclose()


async def test_closing_stream_closes_the_source() -> None:
    closed = False

    async def upstream() -> AsyncGenerator[RAGStreamEvent, None]:
        nonlocal closed
        try:
            while True:
                yield RAGStreamEvent("token", text="x")
        finally:
            closed = True

    gen = sse_stream(upstream())
    await gen.__anext__()
    await gen.aclose()

    assert closed
