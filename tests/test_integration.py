"""Runs against the real database. Start it and ingest first:  make db ingest"""
import psycopg
import pytest
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.models import get_embedder, get_reranker
from app.search import Mode, search
from evals.metrics import load_questions

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def conn():
    settings = get_settings()
    try:
        connection = psycopg.connect(settings.database_url, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (make db)")
    with connection:
        if connection.execute("SELECT to_regclass('chunks')").fetchone()[0] is None:
            pytest.skip("no chunks table (make ingest)")
        register_vector(connection)
        yield connection


def run(conn, query, mode, k=5):
    return search(conn, query, mode, k, get_settings(), get_embedder(), get_reranker())


def test_every_labelled_section_exists(conn):
    stored = {f"{code}#{n}" for code, n in conn.execute(
        "SELECT d.code, c.section_no FROM chunks c JOIN documents d ON d.id = c.document_id")}
    for q in load_questions():
        assert set(q.relevant) <= stored, f"{q.id} points at a section that wasn't ingested"


def test_keyword_search_finds_exact_error_codes(conn):
    hits = run(conn, "VPN-412", Mode.keyword)
    assert hits[0].label == "IT-OPS-003#4"


def test_semantic_search_handles_paraphrase(conn):
    hits = run(conn, "someone nicked my work computer from the train", Mode.semantic)
    assert "IT-OPS-004#5" in [h.label for h in hits[:3]]


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_returns_ranked_hits(conn, mode):
    hits = run(conn, "How long is the account locked after failed sign-ins?", mode)
    assert 0 < len(hits) <= 5
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)


def test_stopword_only_query_does_not_crash(conn):
    assert run(conn, "the and of", Mode.keyword) == []
    assert len(run(conn, "the and of", Mode.hybrid)) > 0  # semantic side still answers


def test_api_search_endpoint(conn):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").json()["chunks"] > 0
        response = client.post("/search", json={"query": "What does error VPN-809 mean?", "k": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "rerank"
        assert body["results"][0]["doc_code"] == "IT-OPS-003"
