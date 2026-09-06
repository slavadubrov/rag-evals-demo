from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

# Pydantic-settings reads .env; the OpenAI SDK reads keys from os.environ.
# Mirror the file into the process environment so both paths see the same
# values regardless of CWD.
load_dotenv(ROOT / ".env", override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Qdrant
    # Embedded (local file) by default — set qdrant_url to use a server.
    qdrant_url: str | None = None
    qdrant_path: Path = ROOT / "qdrant_storage"
    qdrant_collection: str = "scifact"

    # LLM routing
    rag_evals_default_model: str = "gpt-5.6-luna"
    rag_evals_judge_model: str = "gpt-5.6-luna"
    rag_evals_third_judge: str = "gpt-5.6-terra"
    rag_evals_backend: Literal["auto", "live", "mock"] = "mock"  # auto | live | mock

    # Provider keys
    openai_api_key: SecretStr | None = None

    # ML models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_timeout: float = Field(default=60.0, gt=0)
    llm_max_tokens: int = Field(default=4096, gt=0)
    llm_max_calls: int = Field(default=200, gt=0)
    reasoning_effort: str = "low"

    # Thresholds
    threshold_recall_at_10: float = Field(default=0.85, ge=0, le=1)
    threshold_mrr: float = Field(default=0.6, ge=0, le=1)
    threshold_filter_false_exclusion: float = Field(default=0.02, ge=0, le=1)
    threshold_faithfulness: float = Field(default=0.85, ge=0, le=1)

    # Paths
    data_dir: Path = ROOT / "data"
    cache_dir: Path = ROOT / "data" / "cache"
    golden_dir: Path = ROOT / "data" / "golden"


settings = Settings()
