"""Four retrieval modes over the same index, so the evals can compare them like for like.

- keyword:  Postgres full-text search (tsvector + GIN index), ranked with ts_rank_cd
- semantic: pgvector cosine distance over bge-small embeddings (HNSW index)
- hybrid:   both, fused in SQL with reciprocal rank fusion (RRF)
- rerank:   hybrid candidates reordered by a cross-encoder
"""
import time
from dataclasses import dataclass
from enum import StrEnum

import psycopg

from app.config import Settings
from app.models import Embedder, Reranker


class Mode(StrEnum):
    keyword = "keyword"
    semantic = "semantic"
    hybrid = "hybrid"
    rerank = "rerank"


@dataclass
class Hit:
    chunk_id: int
    doc_code: str
    doc_title: str
    section_no: int
    section_title: str
    page: int
    content: str
    score: float

    @property
    def label(self) -> str:
        """The section-level ID the eval set uses, e.g. IT-OPS-003#4."""
        return f"{self.doc_code}#{self.section_no}"


# Questions are natural language, so match ANY of their terms (an OR query) and let
# ranking sort it out. plainto_tsquery would AND every word and miss most questions.
OR_QUERY = """
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS query
    FROM unnest(to_tsvector('english', %(query)s))
"""

KEYWORD_SQL = f"""
WITH q AS ({OR_QUERY})
SELECT c.id, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks c, q
WHERE c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(limit)s
"""

SEMANTIC_SQL = """
SELECT id, 1 - (embedding <=> %(embedding)s) AS score
FROM chunks
ORDER BY embedding <=> %(embedding)s
LIMIT %(limit)s
"""

# RRF scores each chunk by 1 / (k + rank) in every list it appears in. It needs only ranks,
# so the two retrievers' incomparable scores (ts_rank_cd vs cosine) never have to be mixed.
HYBRID_SQL = f"""
WITH q AS ({OR_QUERY}),
semantic AS (
    SELECT id, rank() OVER (ORDER BY embedding <=> %(embedding)s) AS rank
    FROM chunks
    ORDER BY embedding <=> %(embedding)s
    LIMIT %(candidates)s
),
keyword AS (
    SELECT c.id, rank() OVER (ORDER BY ts_rank_cd(c.tsv, q.query) DESC) AS rank
    FROM chunks c, q
    WHERE c.tsv @@ q.query
    ORDER BY ts_rank_cd(c.tsv, q.query) DESC
    LIMIT %(candidates)s
)
SELECT coalesce(s.id, k.id) AS id,
       coalesce(1.0 / (%(rrf_k)s + s.rank), 0.0) + coalesce(1.0 / (%(rrf_k)s + k.rank), 0.0) AS score
FROM semantic s
FULL OUTER JOIN keyword k ON s.id = k.id
ORDER BY score DESC, id
LIMIT %(limit)s
"""

FETCH_SQL = """
SELECT c.id, d.code, d.title, c.section_no, c.section_title, c.page, c.content
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.id = ANY(%(ids)s)
"""


def _fetch(conn: psycopg.Connection, scored: list[tuple[int, float]]) -> list[Hit]:
    if not scored:
        return []
    rows = {row[0]: row for row in conn.execute(FETCH_SQL, {"ids": [cid for cid, _ in scored]})}
    return [Hit(*rows[cid], score=float(score)) for cid, score in scored]


def search(
    conn: psycopg.Connection,
    query: str,
    mode: Mode,
    k: int,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
) -> list[Hit]:
    params = {"query": query, "limit": k, "candidates": settings.candidates, "rrf_k": settings.rrf_k}
    if mode == Mode.keyword:
        return _fetch(conn, conn.execute(KEYWORD_SQL, params).fetchall())

    params["embedding"] = embedder.embed_query(query)
    if mode == Mode.semantic:
        return _fetch(conn, conn.execute(SEMANTIC_SQL, params).fetchall())
    if mode == Mode.hybrid:
        return _fetch(conn, conn.execute(HYBRID_SQL, params).fetchall())

    params["limit"] = settings.candidates
    candidates = _fetch(conn, conn.execute(HYBRID_SQL, params).fetchall())
    scores = reranker.score(query, [f"{h.doc_title} > {h.section_title}\n{h.content}" for h in candidates])
    for hit, score in zip(candidates, scores):
        hit.score = score
    return sorted(candidates, key=lambda h: h.score, reverse=True)[:k]


def timed_search(*args, **kwargs) -> tuple[list[Hit], float]:
    """search() plus its wall-clock time in milliseconds."""
    start = time.perf_counter()
    hits = search(*args, **kwargs)
    return hits, (time.perf_counter() - start) * 1000
