"""Talks to the MCP server through a real MCP client, in-process. Needs Postgres (make db ingest)."""
import json

import anyio
import psycopg
import pytest
from mcp import Client

from app.config import get_settings
from app.mcp_server import server
from app.tickets import seed

pytestmark = pytest.mark.integration
ALEX = get_settings().demo_user


@pytest.fixture(autouse=True)
def demo_tickets():
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2) as conn:
            seed(conn)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (make db)")


def call(name: str, args: dict):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, args)
    result = anyio.run(go)
    return result.is_error, result.content[0].text


def test_tools_are_listed_with_read_only_annotations():
    async def go():
        async with Client(server) as client:
            return (await client.list_tools()).tools
    tools = {t.name: t for t in anyio.run(go)}
    assert set(tools) == {"search_policies", "list_my_tickets", "get_ticket", "create_ticket", "update_ticket"}
    read_only = {name for name, t in tools.items() if t.annotations and t.annotations.read_only_hint}
    assert read_only == {"search_policies", "list_my_tickets", "get_ticket"}
    priority = tools["create_ticket"].input_schema["properties"]["priority"]
    assert priority["enum"] == ["P1", "P2", "P3", "P4"]
    assert "lost or stolen device" in tools["create_ticket"].description


def test_search_policies_finds_the_right_section():
    is_error, text = call("search_policies", {"query": "What does error VPN-809 mean?", "k": 3})
    results = json.loads(text)["results"]
    assert not is_error and results[0]["policy"] == "IT-OPS-003" and results[0]["section"].startswith("4.")


def test_ticket_tools_respect_ownership():
    is_error, text = call("list_my_tickets", {"requester": ALEX, "status": "Open"})
    assert not is_error and [t["number"] for t in json.loads(text)["tickets"]] == ["INC-1002"]

    is_error, text = call("get_ticket", {"requester": ALEX, "number": "INC-1004"})
    assert is_error and "No ticket INC-1004" in text


def test_changes_need_an_approver_and_valid_values():
    args = {"requester": ALEX, "title": "Laptop stolen", "description": "From my car.", "category": "Hardware",
            "priority": "P1"}
    is_error, text = call("create_ticket", {**args, "approved_by": ""})
    assert is_error and "approval" in text

    is_error, text = call("create_ticket", {**args, "approved_by": ALEX, "priority": "P0"})
    assert is_error and "priority" in text  # rejected by the schema before the tool runs

    is_error, text = call("create_ticket", {**args, "approved_by": ALEX})
    assert not is_error and json.loads(text)["number"] == "INC-1007"
