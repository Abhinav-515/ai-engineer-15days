"""Server-Sent Events (SSE) formatting for streamed RAG answers.

This module is framework-agnostic: it yields plain strings and exposes the response
headers, so any web layer (for example FastAPI's ``StreamingResponse``) can serve it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import asdict
from typing import Any

from llmkit.rag import RAGStreamEvent

SSE_HEADERS: dict[str, str] = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # stop reverse proxies such as nginx from buffering
}

SSE_MEDIA_TYPE = "text/event-stream"


def format_sse(event: str, data: Any) -> str:
    """Format one SSE frame. The payload is single-line JSON, so it cannot break framing."""
    if not event or "\n" in event or "\r" in event:
        raise ValueError("event name must be a non-empty single line")
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def format_sse_comment(comment: str = "keepalive") -> str:
    """Format an SSE comment frame, useful as a keep-alive ping."""
    return f": {comment.replace(chr(10), ' ')}\n\n"


def rag_event_to_sse(event: RAGStreamEvent) -> str:
    """Convert one RAG stream event into an SSE frame."""
    if event.type == "rewrite":
        return format_sse("rewrite", {"query": event.text})
    if event.type == "token":
        return format_sse("token", {"text": event.text})
    if event.type == "final" and event.response is not None:
        payload = asdict(event.response)
        payload.pop("context", None)
        payload.pop("retrieved", None)
        return format_sse("final", payload)
    if event.type == "error":
        return format_sse("error", {"message": event.error or "unknown error"})
    return format_sse("error", {"message": "malformed stream event"})


async def sse_stream(
    events: AsyncGenerator[RAGStreamEvent, None],
) -> AsyncGenerator[str, None]:
    """Yield SSE frames as events arrive, ending with exactly one ``done`` frame.

    Frames are produced one at a time, so the full answer is never buffered. Exceptions
    from the source become a generic ``error`` frame; internal details are not leaked.
    """
    try:
        async with aclosing(events):
            async for event in events:
                yield rag_event_to_sse(event)
                if event.type in ("final", "error"):
                    break
    except Exception as exc:
        yield format_sse("error", {"message": f"internal error: {type(exc).__name__}"})
    yield format_sse("done", {})
