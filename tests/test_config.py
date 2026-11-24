from llmkit.config import Settings


def test_defaults_and_secret_masking(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    s = Settings(_env_file=None)
    assert s.ollama_base_url == "http://localhost:11434"
    assert "sk-test-123" not in repr(s)
    assert s.openai_api_key.get_secret_value() == "sk-test-123"
