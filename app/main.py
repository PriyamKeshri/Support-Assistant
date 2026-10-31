import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool

from app.answer import AnswerStream, generate_answer
from app.config import Settings, get_settings
from app.db import create_pool
from app.models import get_embedder, get_reranker
from app.search import Hit, Mode, timed_search

log = logging.getLogger(__name__)
CHAT_PAGE = Path(__file__).with_name("static") / "index.html"


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


def require_llm(settings: Settings) -> None:
    if not settings.llm_configured:
        raise HTTPException(503, "No LLM configured. Set LLM_BASE_URL and LLM_MODEL (see .env.example); "
                                 "/search works without one.")


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


class Citation(Result):
    n: int  # the number the answer uses for this excerpt, as in [n]

    @classmethod
    def numbered(cls, n: int, hit: Hit) -> "Citation":
        return cls(n=n, **Result.from_hit(hit).model_dump())


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
    citations: list[Citation]
    retrieval_ms: float
    generation_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(CHAT_PAGE)


@app.get("/health")
def health(pool: ConnectionPool = Depends(get_pool), settings: Settings = Depends(get_settings)) -> dict:
    with pool.connection() as conn:
        documents, chunks = conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)").fetchone()
    return {"status": "ok", "documents": documents, "chunks": chunks,
            "llm_model": settings.llm_model if settings.llm_configured else None}


@app.post("/search")
def search_endpoint(body: SearchRequest, pool: ConnectionPool = Depends(get_pool),
                    settings: Settings = Depends(get_settings)) -> SearchResponse:
    with pool.connection() as conn:
        hits, ms = timed_search(conn, body.query, body.mode, body.k, settings, get_embedder(), get_reranker())
    return SearchResponse(mode=body.mode, latency_ms=round(ms, 1), results=[Result.from_hit(h) for h in hits])


@app.post("/ask")
def ask_endpoint(body: AskRequest, pool: ConnectionPool = Depends(get_pool),
                 settings: Settings = Depends(get_settings)) -> AskResponse:
    require_llm(settings)
    with pool.connection() as conn:
        hits, retrieval_ms = timed_search(conn, body.question, Mode.rerank, body.k, settings,
                                          get_embedder(), get_reranker())
    answer = generate_answer(body.question, hits, settings)
    return AskResponse(
        answer=answer.text,
        abstained=answer.abstained,
        citations=[Citation.numbered(n, answer.context[n - 1]) for n in answer.cited],
        retrieval_ms=round(retrieval_ms, 1),
        generation_ms=round(answer.generation_ms, 1),
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        cost_usd=round(answer.cost_usd(settings), 6),
    )


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/ask/stream")
def ask_stream_endpoint(body: AskRequest, pool: ConnectionPool = Depends(get_pool),
                        settings: Settings = Depends(get_settings)) -> StreamingResponse:
    """Like /ask, but streamed as server-sent events: `sources` (the numbered excerpts), then a
    `token` event per piece of the answer as the model writes it, then `done` with the cited
    numbers, timing and token counts. A failure after streaming has started arrives as `error`."""
    require_llm(settings)
    with pool.connection() as conn:
        hits, retrieval_ms = timed_search(conn, body.question, Mode.rerank, body.k, settings,
                                          get_embedder(), get_reranker())

    def events():
        yield sse("sources", [Citation.numbered(n, h).model_dump() for n, h in enumerate(hits, start=1)])
        stream = AnswerStream(body.question, hits, settings)
        try:
            for text in stream:
                yield sse("token", {"text": text})
        except Exception as exc:  # the 200 status is already sent, so report the failure in the stream
            log.exception("answer generation failed")
            yield sse("error", {"detail": f"The model call failed: {exc}"})
            return
        answer = stream.answer
        yield sse("done", {
            "answer": answer.text,
            "cited": answer.cited,
            "abstained": answer.abstained,
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(answer.generation_ms, 1),
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
            "cost_usd": round(answer.cost_usd(settings), 6),
        })

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
