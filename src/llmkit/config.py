"""Centralised, validated configuration loaded from environment / .env."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Cloud provider (any OpenAI-compatible API: Groq, OpenAI, Together, ...)
    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str = "https://api.groq.com/openai/v1"
    openai_model: str = "openai/gpt-oss-20b"
    cloud_reasoning_effort: str = "low"  # "" disables; only for reasoning models
    cloud_price_in_per_m: float = 0.075  # USD per 1M input tokens (verify on provider site)
    cloud_price_out_per_m: float = 0.30  # USD per 1M output tokens

    # Local provider
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"


@lru_cache
def get_settings() -> Settings:
    return Settings()
