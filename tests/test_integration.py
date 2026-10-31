"""Runs against the real database. Start it and ingest first:  make db ingest"""
import json
from types import SimpleNamespace

import psycopg
import pytest
from pgvector.psycopg import register_vector

from app.config import Settings, get_settings
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


@pytest.fixture
def api(conn):
    """The FastAPI app with an LLM configured; tests swap the model itself for fake_llm."""
    from fastapi.testclient import TestClient

    from app.main import app

    app.dependency_overrides[get_settings] = lambda: Settings(llm_base_url="http://fake", llm_model="fake")
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def parse_events(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_chat_page_is_served(api):
    response = api.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Kestrel IT Help Desk" in response.text


def test_ask_numbers_its_citations(api, fake_llm):
    fake_llm("Report it within 2 hours [2]. IT locks it remotely [1].")
    body = api.post("/ask", json={"question": "I lost my laptop at the airport"}).json()
    assert [c["n"] for c in body["citations"]] == [1, 2]
    assert body["prompt_tokens"] == 120


def test_ask_stream_sends_sources_then_tokens_then_done(api, fake_llm):
    reply = "Your network is blocking UDP port 51820, so switch networks [1]."
    fake_llm(reply)
    response = api.post("/ask/stream", json={"question": "What does error VPN-809 mean?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_events(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and set(names[1:-1]) == {"token"}
    sources = events[0][1]
    assert [s["n"] for s in sources] == [1, 2, 3, 4, 5]
    assert sources[0]["doc_code"] == "IT-OPS-003"
    assert "".join(data["text"] for name, data in events if name == "token") == reply
    done = events[-1][1]
    assert done["cited"] == [1] and not done["abstained"] and done["completion_tokens"] == 30


def test_model_failure_mid_stream_arrives_as_an_error_event(api, monkeypatch):
    def unreachable(**kwargs):
        raise ConnectionError("model server is down")
    broken = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=unreachable)))
    monkeypatch.setattr("app.answer.get_client", lambda base_url, api_key: broken)

    events = parse_events(api.post("/ask/stream", json={"question": "How do I fix VPN-127?"}).text)
    assert [name for name, _ in events] == ["sources", "error"]
    assert "model server is down" in events[1][1]["detail"]


def test_ask_without_a_model_is_a_clear_503(conn):
    from fastapi.testclient import TestClient

    from app.main import app

    app.dependency_overrides[get_settings] = lambda: Settings(llm_base_url="", llm_model="")
    try:
        with TestClient(app) as client:
            response = client.post("/ask/stream", json={"question": "anything"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503
    assert "No LLM configured" in response.json()["detail"]
