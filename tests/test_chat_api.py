"""The agent's HTTP endpoints, with a scripted model but the real MCP tools, database and
Postgres checkpointer. Needs Postgres (make db ingest)."""
import uuid

import psycopg
import pytest

from app.config import Settings, get_settings
from app.tickets import seed
from helpers import ScriptedLLM, parse_events

pytestmark = pytest.mark.integration


@pytest.fixture
def chat(monkeypatch):
    """Yields (client, script): call script(*replies) before a request to set the model's replies."""
    from fastapi.testclient import TestClient

    from app import agent, main

    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2) as conn:
            seed(conn)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (make db)")

    llm = ScriptedLLM()

    def script(*replies):
        llm.replies.extend(replies)
        return llm

    monkeypatch.setattr(main, "open_agent", lambda settings, checkpointer: agent.open_agent(settings, checkpointer,
                                                                                            llm=llm))
    main.app.dependency_overrides[get_settings] = lambda: Settings(llm_base_url="http://fake", llm_model="fake")
    try:
        with TestClient(main.app) as client:
            yield client, script
    finally:
        main.app.dependency_overrides.clear()


def post(client, path, body):
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    return parse_events(response.text)


def test_a_ticket_is_created_only_after_approval(chat):
    client, script = chat
    script([("create_ticket", {"title": "Laptop stolen", "description": "Stolen from my car.",
                               "category": "Hardware", "priority": "P1"})], "Created INC-1007 for you.")
    thread = uuid.uuid4().hex
    events = post(client, "/chat", {"thread_id": thread, "message": "My laptop was stolen, raise a ticket."})
    name, approval = events[-1]
    assert name == "approval" and approval["actions"][0]["summary"] == "Create a P1 Hardware ticket: Laptop stolen"
    assert [t["number"] for t in client.get("/tickets").json()["tickets"]].count("INC-1007") == 0

    events = post(client, "/chat/resume", {"thread_id": thread, "decisions": [
        {"id": approval["actions"][0]["id"], "action": "approve"}]})
    tool = next(data for name, data in events if name == "tool")
    assert (tool["name"], tool["status"]) == ("create_ticket", "done")
    assert events[-1][0] == "done" and events[-1][1]["answer"] == "Created INC-1007 for you."

    mine = client.get("/tickets").json()["tickets"]
    assert mine[0]["number"] == "INC-1007" and mine[0]["priority"] == "P1"
    with psycopg.connect(get_settings().database_url) as conn:
        approver = conn.execute("""SELECT approved_by FROM ticket_events e JOIN tickets t ON t.id = e.ticket_id
                                   WHERE t.number = 'INC-1007'""").fetchone()[0]
    assert approver == get_settings().demo_user


def test_conversations_are_remembered_between_requests(chat):
    client, script = chat
    llm = script("Hi Alex.", "You said hello.")
    thread = uuid.uuid4().hex
    post(client, "/chat", {"thread_id": thread, "message": "Hello"})
    post(client, "/chat", {"thread_id": thread, "message": "What did I just say?"})
    assert [m["role"] for m in llm.requests[-1]["messages"]] == ["system", "user", "assistant", "user"]


def test_resuming_without_a_pending_approval_is_reported(chat):
    client, _ = chat
    events = post(client, "/chat/resume", {"thread_id": uuid.uuid4().hex,
                                           "decisions": [{"id": "call_x", "action": "approve"}]})
    assert events == [("error", {"type": "error",
                                 "detail": "Nothing in this conversation is waiting for approval."})]


def test_requests_are_validated(chat):
    client, _ = chat
    assert client.post("/chat", json={"thread_id": "no spaces!", "message": "hi"}).status_code == 422
    assert client.post("/chat/resume", json={"thread_id": uuid.uuid4().hex, "decisions": []}).status_code == 422
