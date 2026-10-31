from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool

from app.answer import generate_answer
from app.config import Settings, get_settings
from app.db import create_pool
from app.models import get_embedder, get_reranker
from app.search import Hit, Mode, timed_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.pool = create_pool(settings.database_url)
    # Load both models at startup so the first request doesn't pay for it.
    get_embedder()
    get_reranker()
    yield
    app.state.pool.close()


app = FastAPI(
    title="Agentic Support Assistant",
    description="IT help desk assistant over company policy PDFs: hybrid search on Postgres + pgvector.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pool(request: Request) -> ConnectionPool:
    return request.app.state.pool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, examples=["My VPN says VPN-412, what do I do?"])
    mode: Mode = Mode.rerank
    k: int = Field(default=5, ge=1, le=20)


class Result(BaseModel):
    doc_code: str
    doc_title: str
    section: str
    page: int
    score: float
    content: str

    @classmethod
    def from_hit(cls, hit: Hit) -> "Result":
        return cls(doc_code=hit.doc_code, doc_title=hit.doc_title, page=hit.page, score=round(hit.score, 4),
                   section=f"{hit.section_no}. {hit.section_title}", content=hit.content)


class SearchResponse(BaseModel):
    mode: Mode
    latency_ms: float
    results: list[Result]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, examples=["I lost my laptop at the airport. What now?"])
    k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Result]
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@app.get("/health")
def health(pool: ConnectionPool = Depends(get_pool)) -> dict:
    with pool.connection() as conn:
        documents, chunks = conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)").fetchone()
    return {"status": "ok", "documents": documents, "chunks": chunks}


@app.post("/search")
def search_endpoint(body: SearchRequest, pool: ConnectionPool = Depends(get_pool),
                    settings: Settings = Depends(get_settings)) -> SearchResponse:
    with pool.connection() as conn:
        hits, ms = timed_search(conn, body.query, body.mode, body.k, settings, get_embedder(), get_reranker())
    return SearchResponse(mode=body.mode, latency_ms=round(ms, 1), results=[Result.from_hit(h) for h in hits])


@app.post("/ask")
def ask_endpoint(body: AskRequest, pool: ConnectionPool = Depends(get_pool),
                 settings: Settings = Depends(get_settings)) -> AskResponse:
    if not settings.llm_configured:
        raise HTTPException(503, "No LLM configured. Set LLM_BASE_URL and LLM_MODEL (see .env.example); "
                                 "/search works without one.")
    with pool.connection() as conn:
        hits, retrieval_ms = timed_search(conn, body.question, Mode.rerank, body.k, settings,
                                          get_embedder(), get_reranker())
    answer = generate_answer(body.question, hits, settings)
    return AskResponse(
        answer=answer.text,
        abstained=answer.abstained,
        citations=[Result.from_hit(h) for h in answer.citations],
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(answer.generation_ms, 1),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        cost_usd=round(answer.cost_usd(settings), 6),
    )
