"""Benchmark task set: prompts with objective, automatic pass/fail checks."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    prompt: str
    check: Callable[[str], bool]
    max_tokens: int = 512


def _words(text: str) -> int:
    return len(text.split())


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _check_json(text: str) -> bool:
    try:
        obj = json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    age = obj.get("age")
    return isinstance(obj.get("name"), str) and isinstance(age, int) and not isinstance(age, bool)


def _check_list(text: str) -> bool:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return len(lines) == 3 and not any(re.match(r"^(\d+[.)]|[-*\u2022])", ln) for ln in lines)


def _check_ball(text: str) -> bool:
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    return "5" in nums or "0.05" in nums


_SUMMARY_SOURCE = (
    "A vector database stores data as high-dimensional numeric vectors called embeddings. "
    "It uses approximate nearest-neighbor indexes such as HNSW to find the vectors most "
    "similar to a query vector quickly. This makes it useful for semantic search, "
    "recommendations, and retrieval-augmented generation."
)

TASKS: list[Task] = [
    Task(
        "capital",
        "factual",
        "What is the capital of Australia? Answer with just the city name.",
        lambda t: "canberra" in t.lower(),
        max_tokens=256,
    ),
    Task(
        "math",
        "math",
        "What is 17 * 23? Answer with just the number.",
        lambda t: "391" in t.replace(",", ""),
        max_tokens=256,
    ),
    Task(
        "json",
        "structured",
        "Return a JSON object with keys name (string) and age (integer) for a fictional "
        "person. Output only the JSON.",
        _check_json,
    ),
    Task(
        "list",
        "instruction",
        "List exactly 3 fruits, one per line, with no numbering, bullets, or extra text.",
        _check_list,
        max_tokens=256,
    ),
    Task(
        "ball",
        "reasoning",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Answer in cents, with just the number.",
        _check_ball,
    ),
    Task(
        "summary",
        "summarization",
        f"Summarize the following text in one sentence of at most 30 words:\n\n{_SUMMARY_SOURCE}",
        lambda t: _words(t) <= 40 and "vector" in t.lower(),
    ),
    Task(
        "code",
        "code",
        "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric "
        "characters. Output only the code.",
        lambda t: "def is_palindrome" in t and "return" in t,
    ),
    Task(
        "explain",
        "long-form",
        "Explain the difference between TCP and UDP in about 100 words.",
        lambda t: 60 <= _words(t) <= 160 and "tcp" in t.lower() and "udp" in t.lower(),
        max_tokens=600,
    ),
]
