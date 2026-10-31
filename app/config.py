from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings come from environment variables or .env (names are case-insensitive)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://support:support@localhost:5433/support"

    # Local models, so retrieval and its evals run with no API key and no per-query cost.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    chunk_chars: int = 1200
    chunk_overlap_chars: int = 200

    candidates: int = 30  # results taken from each retriever before fusion and reranking
    rrf_k: int = 60       # reciprocal rank fusion constant

    # Any OpenAI-compatible chat endpoint: OpenAI, Azure OpenAI's v1 endpoint, Groq or Ollama.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    judge_model: str = ""  # model for the answer evals; defaults to llm_model
    llm_input_price_per_mtok: float = 0.0   # USD per million tokens, for cost reporting
    llm_output_price_per_mtok: float = 0.0

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
