from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant
    # Embedded (local file) by default — set qdrant_url to use a server.
    qdrant_url: str | None = None
    qdrant_path: Path = ROOT / "qdrant_storage"
    qdrant_collection: str = "scifact"

    # LLM routing
    rag_evals_default_model: str = "gpt-5-mini"
    rag_evals_judge_model: str = "claude-haiku-4-5"
    rag_evals_third_judge: str = "gemini/gemini-3-flash"
    rag_evals_backend: str = "auto"  # auto | live | mock

    # Provider keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # ML models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Thresholds
    threshold_recall_at_10: float = 0.85
    threshold_mrr: float = 0.6
    threshold_filter_false_exclusion: float = 0.02
    threshold_faithfulness: float = 0.85

    # Paths
    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / "data" / "cache"
    golden_dir: Path = ROOT / "data" / "golden"


settings = Settings()
