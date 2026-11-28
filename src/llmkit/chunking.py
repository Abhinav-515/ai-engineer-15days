"""Text chunking strategies for retrieval-augmented generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Minimal interface required by semantic chunking."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Convert a sequence of texts into embedding vectors."""


def _validate_limits(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")


def fixed_size_chunks(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into fixed character windows with optional overlap."""
    _validate_limits(chunk_size, overlap)
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap

    return chunks


def _split_recursive(text: str, separators: tuple[str, ...], size: int) -> list[str]:
    """Recursively split oversized text using progressively smaller separators."""
    if len(text) <= size:
        return [text] if text else []

    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    separator = separators[0]

    # The final empty separator means split by individual character windows.
    if not separator:
        return [text[i : i + size] for i in range(0, len(text), size)]

    if separator not in text:
        return _split_recursive(text, separators[1:], size)

    pieces = text.split(separator)
    result = []

    for piece in pieces:
        if not piece:
            continue
        if len(piece) <= size:
            result.append(piece)
        else:
            result.extend(_split_recursive(piece, separators[1:], size))

    return result


def recursive_chunks(
    text: str,
    *,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Split text hierarchically, preferring paragraph and sentence boundaries."""
    _validate_limits(chunk_size, overlap)
    text = text.strip()
    if not text:
        return []

    pieces = _split_recursive(text, separators, chunk_size)
    chunks = []
    current = ""

    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if overlap and current:
            prefix = current[-overlap:]
            current = f"{prefix} {piece}".strip()
            if len(current) > chunk_size:
                current = piece
        else:
            current = piece

    if current:
        chunks.append(current)

    return chunks


def _sentences(text: str) -> list[str]:
    """Split text into simple sentence-like units."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero dimension")

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def semantic_chunks(
    text: str,
    embedder: Embedder,
    *,
    similarity_threshold: float = 0.65,
    max_chunk_size: int = 1000,
) -> list[str]:
    """Group adjacent sentences while their embedding similarity remains high."""
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    sentences = _sentences(text)
    if not sentences:
        return []

    vectors = embedder.embed(sentences)
    if len(vectors) != len(sentences):
        raise ValueError("Embedder returned a different number of vectors")

    chunks = []
    current = sentences[0]

    for index in range(1, len(sentences)):
        sentence = sentences[index]
        similarity = _cosine_similarity(vectors[index - 1], vectors[index])
        candidate = f"{current} {sentence}"

        if similarity >= similarity_threshold and len(candidate) <= max_chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


async def async_semantic_chunks(
    text: str,
    embedder,
    *,
    similarity_threshold: float = 0.65,
    max_chunk_size: int = 1000,
) -> list[str]:
    """Async version of semantic chunking for network-based embedding clients."""
    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    sentences = _sentences(text)
    if not sentences:
        return []

    vectors = await embedder.embed(sentences)
    if len(vectors) != len(sentences):
        raise ValueError("Embedder returned a different number of vectors")

    chunks = []
    current = sentences[0]

    for index in range(1, len(sentences)):
        sentence = sentences[index]
        similarity = _cosine_similarity(vectors[index - 1], vectors[index])
        candidate = f"{current} {sentence}"

        if similarity >= similarity_threshold and len(candidate) <= max_chunk_size:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks
