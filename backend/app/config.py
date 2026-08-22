"""
Application configuration.

Loads all settings from environment variables / .env file.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — single source of truth for the entire app."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq API ──────────────────────────────────────────────────────
    groq_api_key: str = ""

    # Models
    groq_vision_model: str = "qwen/qwen3.6-27b"
    groq_text_model: str = "groq/compound-mini"
    groq_whisper_model: str = "whisper-large-v3-turbo"
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "troy"

    # ── RAG ───────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "./knowledge_base/faiss_index"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    rag_rerank_top_n: int = 3

    # ── Agent ─────────────────────────────────────────────────────────
    confidence_threshold: float = 0.70
    max_agent_iterations: int = 8

    # ── Paths ─────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    knowledge_base_raw_dir: str = "./knowledge_base/raw"
    log_dir: str = "./logs"

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — loaded once, reused everywhere."""
    return Settings()
