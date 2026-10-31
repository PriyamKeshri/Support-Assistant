"""Local embedding and reranking models, loaded once per process."""
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings


class Embedder:
    def __init__(self, model_name: str, query_instruction: str):
        self.model = SentenceTransformer(model_name)
        self.query_instruction = query_instruction

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=32, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        # BGE models retrieve better when queries carry this instruction prefix.
        return self.model.encode(self.query_instruction + query, normalize_embeddings=True)


class Reranker:
    """A cross-encoder reads the query and passage together, so it judges relevance more
    precisely than comparing two separately computed embeddings, but it is too slow to run
    over the whole corpus. It reorders the top candidates from hybrid search."""

    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    return Embedder(settings.embedding_model, settings.query_instruction)


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(get_settings().reranker_model)
