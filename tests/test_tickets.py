"""Runs against the real database (make db). Each test starts from the demo tickets."""
import psycopg
import pytest

from app.config import get_settings
from app.tickets import TicketError, create_ticket, get_ticket, list_tickets, seed, update_ticket

pytestmark = pytest.mark.integration
ALEX = get_settings().demo_user
JORDAN = "jordan.lee@kestrel.example"


@pytest.fixture
def conn():
    try:
        connection = psycopg.connect(get_settings().database_url, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (make db)")
    with connection:
        seed(connection)
        yield connection


def test_employees_see_only_their_own_tickets(conn):
    assert [t["number"] for t in list_tickets(conn, ALEX)] == ["INC-1002", "INC-1001", "INC-1003", "INC-1005"]
    assert [t["number"] for t in list_tickets(conn, ALEX, status="Open")] == ["INC-1002"]
    assert [t["number"] for t in list_tickets(conn, JORDAN)] == ["INC-1004", "INC-1006"]


def test_ticket_details_include_history(conn):
    ticket = get_ticket(conn, ALEX, "inc-1003")
    assert ticket["status"] == "Waiting on User"
    assert [e["action"] for e in ticket["history"]] == ["created", "commented"]
    assert "different cable" in ticket["history"][1]["comment"]


def test_someone_elses_ticket_looks_like_it_does_not_exist(conn):
    with pytest.raises(TicketError, match="No ticket INC-1004"):
        get_ticket(conn, ALEX, "INC-1004")
    with pytest.raises(TicketError, match="No ticket INC-1004"):
        update_ticket(conn, ALEX, ALEX, "INC-1004", status="Closed")


def test_create_needs_approval_and_valid_values(conn):
    with pytest.raises(TicketError, match="approval"):
        create_ticket(conn, ALEX, "", "Laptop stolen", "Stolen from my car.", "Hardware", "P1")
    with pytest.raises(TicketError, match="category must be one of"):
        create_ticket(conn, ALEX, ALEX, "Laptop stolen", "Stolen from my car.", "Hardwares", "P1")

    created = create_ticket(conn, ALEX, ALEX, "Laptop stolen", "Stolen from my car.", "Hardware", "P1")
    assert (created["number"], created["status"], created["priority"]) == ("INC-1007", "Open", "P1")
    event = conn.execute("SELECT action, actor, approved_by FROM ticket_events ORDER BY id DESC LIMIT 1").fetchone()
    assert event == ("created", ALEX, ALEX)


def test_update_records_what_changed(conn):
    closed = update_ticket(conn, ALEX, ALEX, "INC-1002", status="Closed", comment="Got access another way.")
    assert closed["status"] == "Closed"
    assert closed["changed"] == {"status": ["Open", "Closed"]}
    history = get_ticket(conn, ALEX, "INC-1002")["history"]
    assert history[-1]["changes"] == {"status": ["Open", "Closed"]}
    assert history[-1]["comment"] == "Got access another way."

    comment_only = update_ticket(conn, ALEX, ALEX, "INC-1003", comment="A new cable fixed it.")
    assert comment_only["changed"] == {} and comment_only["comment_added"] == "A new cable fixed it."


def test_update_refuses_what_employees_cannot_do(conn):
    with pytest.raises(TicketError, match="status must be one of: Open, Closed"):
        update_ticket(conn, ALEX, ALEX, "INC-1001", status="Resolved")
    with pytest.raises(TicketError, match="Nothing to change"):
        update_ticket(conn, ALEX, ALEX, "INC-1001", priority="P3")
    with pytest.raises(TicketError, match="approval"):
        update_ticket(conn, ALEX, "", "INC-1001", comment="hi")
