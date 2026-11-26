import httpx
import pytest

from llmkit.config import Settings
from llmkit.embeddings import EmbeddingError, OllamaEmbedder


def make_embedder(handler):
    return OllamaEmbedder(
        transport=httpx.MockTransport(handler),
        settings=Settings(_env_file=None),
    )


async def test_embed_sends_batch_and_returns_vectors():
    def handler(request):
        assert request.url.path == "/api/embed"
        assert request.read()
        import json

        payload = json.loads(request.content)
        assert payload["model"] == "nomic-embed-text"
        assert payload["input"] == ["first text", "second text"]
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    async with make_embedder(handler) as embedder:
        vectors = await embedder.embed(["first text", "second text"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_empty_input_skips_http_request():
    def handler(request):
        pytest.fail("No request should be made for empty input")

    async with make_embedder(handler) as embedder:
        assert await embedder.embed([]) == []


async def test_embed_rejects_blank_input():
    async with make_embedder(lambda request: httpx.Response(200)) as embedder:
        with pytest.raises(ValueError, match="non-empty"):
            await embedder.embed(["valid", " "])


async def test_embed_rejects_http_errors():
    def handler(request):
        return httpx.Response(404, json={"error": "model not found"})

    async with make_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError, match="HTTP 404"):
            await embedder.embed(["hello"])


async def test_embed_rejects_wrong_number_of_vectors():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    async with make_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError, match="number of embeddings"):
            await embedder.embed(["one", "two"])


async def test_embed_rejects_non_finite_values():
    def handler(request):
        return httpx.Response(200, content=b'{"embeddings": [[0.1, 1e999]]}')

    async with make_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError, match="non-finite"):
            await embedder.embed(["hello"])


async def test_embed_rejects_inconsistent_dimensions():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1], [0.2, 0.3]]})

    async with make_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError, match="inconsistent"):
            await embedder.embed(["one", "two"])


async def test_embed_rejects_empty_vector():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[]]})

    async with make_embedder(handler) as embedder:
        with pytest.raises(EmbeddingError, match="empty"):
            await embedder.embed(["hello"])


def test_embedding_model_has_default():
    settings = Settings(_env_file=None)
    assert settings.ollama_embedding_model == "nomic-embed-text"
