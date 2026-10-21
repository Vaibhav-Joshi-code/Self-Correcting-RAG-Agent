"""Validated configuration; environment variables override the project .env."""
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)
    euri_api_key: SecretStr = Field(default=SecretStr(""), validation_alias=AliasChoices("EURI_API_KEY", "EURIAI_API_KEY"))
    euri_base_url: str = "https://api.euron.one/api/v1/euri"
    chat_model: str = "gpt-4.1-mini"
    evaluator_model: str = ""
    embedding_model: str = "text-embedding-3-small"
    data_dir: Path = BASE_DIR / "data"
    chroma_dir: Path = BASE_DIR / "chroma_data"
    collection_name: str = "self_correcting_rag"
    chunk_size: int = Field(default=900, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0)
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    top_k: int = Field(default=4, ge=1, le=20)
    max_retrieval_k: int = Field(default=12, ge=1, le=50)
    max_retries: int = Field(default=2, ge=0, le=5)
    min_retrieval_relevance: int = Field(default=3, ge=1, le=5)
    min_grounding: int = Field(default=4, ge=1, le=5)
    min_completeness: int = Field(default=4, ge=1, le=5)
    max_hallucination_risk: int = Field(default=2, ge=1, le=5)
    request_timeout_seconds: float = Field(default=45, gt=0, le=180)
    provider_max_retries: int = Field(default=2, ge=0, le=4)
    max_output_tokens: int = Field(default=1800, ge=256, le=8192)
    max_context_chars: int = Field(default=18000, ge=1000, le=100000)
    langsmith_tracing: bool = True
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = "self-correcting-rag-agent"
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", validation_alias=AliasChoices("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"))
    api_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def validate_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.max_retrieval_k < self.top_k:
            raise ValueError("MAX_RETRIEVAL_K must be at least TOP_K")
        if not self.euri_base_url.startswith("https://"):
            raise ValueError("EURI_BASE_URL must use HTTPS")
        if self.langsmith_tracing and not self.langsmith_api_key.get_secret_value():
            raise ValueError("LANGSMITH_API_KEY is required when tracing is enabled")
        for field in ("data_dir", "chroma_dir"):
            setattr(self, field, (BASE_DIR / getattr(self, field)).resolve())
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
