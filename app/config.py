from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Knowledge Assistant"
    data_dir: Path = Path("data")
    uploads_dir: Path = Path("data/uploads")
    index_path: Path = Path("data/index/vector_store.json")
    embedding_provider: str = "local"
    llm_provider: str = "local"
    openai_api_key: Optional[str] = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    llama_base_url: str = "http://localhost:11434"
    llama_model: str = "llama3.1"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.index_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
