import pytest

from llmkit.chunking import (
    fixed_size_chunks,
    recursive_chunks,
    semantic_chunks,
)


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            if "finance" in text.lower() or "invoice" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_fixed_size_chunks_respect_size_and_overlap():
    chunks = fixed_size_chunks("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_fixed_size_chunks_empty_text():
    assert fixed_size_chunks("   ") == []


def test_fixed_size_chunks_reject_invalid_limits():
    with pytest.raises(ValueError):
        fixed_size_chunks("hello", chunk_size=10, overlap=10)


def test_recursive_chunks_prefer_paragraph_boundaries():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = recursive_chunks(text, chunk_size=25, overlap=0)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 25 for chunk in chunks)


def test_recursive_chunks_empty_text():
    assert recursive_chunks("  ") == []


def test_semantic_chunks_group_related_sentences():
    text = (
        "Finance invoice processing is important. "
        "Finance invoice records need validation. "
        "Sports teams play football. "
        "Sports players train every day."
    )
    chunks = semantic_chunks(
        text,
        FakeEmbedder(),
        similarity_threshold=0.8,
        max_chunk_size=200,
    )
    assert len(chunks) == 2
    assert "Finance invoice" in chunks[0]
    assert "Sports teams" in chunks[1]


def test_semantic_chunks_reject_invalid_threshold():
    with pytest.raises(ValueError):
        semantic_chunks("Hello world.", FakeEmbedder(), similarity_threshold=2)


def test_semantic_chunks_empty_text():
    assert semantic_chunks(" ", FakeEmbedder()) == []


def test_recursive_chunks_falls_back_to_character_windows():
    from llmkit.chunking import recursive_chunks

    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = recursive_chunks(
        text,
        chunk_size=5,
        overlap=0,
        separators=("",),
    )

    assert chunks == ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
    assert all(len(chunk) <= 5 for chunk in chunks)
