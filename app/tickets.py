"""The help desk's tickets: storage, access rules and the audit trail.

Employees see and change only their own tickets. Creating or changing a ticket needs the
name of whoever approved it, and every change is recorded in ticket_events.

    python -m app.tickets seed     # reset the tickets to the demo data
"""
import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal, get_args

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import get_settings

SCHEMA = Path(__file__).with_name("tickets.sql")

Category = Literal["Account and Password", "Access Request", "Hardware", "Software",
                   "Network/VPN", "Email and Collaboration", "Security Incident"]
Priority = Literal["P1", "P2", "P3", "P4"]
Status = Literal["Open", "In Progress", "Waiting on User", "Resolved", "Closed"]
# Employees can close their own tickets or reopen them; the other states belong to the Service Desk.
EmployeeStatus = Literal["Open", "Closed"]

CATEGORIES, PRIORITIES, STATUSES = get_args(Category), get_args(Priority), get_args(Status)


class TicketError(Exception):
    """A request the help desk refuses. The message is safe to show the employee."""


def init_tickets(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text())
    conn.commit()


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="minutes")


def _summary(row: dict) -> dict:
    return {
        "number": row["number"], "title": row["title"], "category": row["category"],
        "priority": row["priority"], "status": row["status"], "updated": _iso(row["updated_at"]),
    }


def _check(value, allowed: tuple, field: str) -> None:
    if value is not None and value not in allowed:
        raise TicketError(f"{field} must be one of: {', '.join(allowed)}")


def list_tickets(conn: psycopg.Connection, requester: str, status: str | None = None) -> list[dict]:
    _check(status, STATUSES, "status")
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """SELECT * FROM tickets
               WHERE requester = %(requester)s AND (%(status)s::text IS NULL OR status = %(status)s)
               ORDER BY updated_at DESC, id DESC""",
            {"requester": requester, "status": status},
        ).fetchall()
    return [_summary(r) for r in rows]


def get_ticket(conn: psycopg.Connection, requester: str, number: str) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute("SELECT * FROM tickets WHERE number = %s AND requester = %s",
                          (number.strip().upper(), requester)).fetchone()
        if row is None:
            # The same answer for someone else's ticket, so the tool can't reveal that it exists.
            raise TicketError(f"No ticket {number} found for {requester}.")
        events = cur.execute(
            "SELECT * FROM ticket_events WHERE ticket_id = %s ORDER BY created_at, id", (row["id"],)).fetchall()
    return {
        **_summary(row),
        "description": row["description"],
        "created": _iso(row["created_at"]),
        "history": [{"when": _iso(e["created_at"]), "action": e["action"], "by": e["actor"],
                     **({"changes": e["changes"]} if e["changes"] else {}),
                     **({"comment": e["comment"]} if e["comment"] else {})} for e in events],
    }


def create_ticket(conn: psycopg.Connection, requester: str, approved_by: str, title: str, description: str,
                  category: str, priority: str) -> dict:
    if not approved_by:
        raise TicketError("Creating a ticket needs the employee's approval.")
    _check(category, CATEGORIES, "category")
    _check(priority, PRIORITIES, "priority")
    title, description = title.strip(), description.strip()
    if not 3 <= len(title) <= 120:
        raise TicketError("title must be 3 to 120 characters")
    if not description:
        raise TicketError("description must not be empty")
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            """INSERT INTO tickets (requester, title, description, category, priority)
               VALUES (%s, %s, %s, %s, %s) RETURNING *""",
            (requester, title, description, category, priority),
        ).fetchone()
        cur.execute("INSERT INTO ticket_events (ticket_id, action, actor, approved_by) VALUES (%s, 'created', %s, %s)",
                    (row["id"], requester, approved_by))
    return _summary(row)


def update_ticket(conn: psycopg.Connection, requester: str, approved_by: str, number: str,
                  status: str | None = None, priority: str | None = None, comment: str | None = None) -> dict:
    if not approved_by:
        raise TicketError("Changing a ticket needs the employee's approval.")
    _check(status, get_args(EmployeeStatus), "status")
    _check(priority, PRIORITIES, "priority")
    comment = (comment or "").strip() or None
    with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute("SELECT * FROM tickets WHERE number = %s AND requester = %s FOR UPDATE",
                          (number.strip().upper(), requester)).fetchone()
        if row is None:
            raise TicketError(f"No ticket {number} found for {requester}.")
        changes = {field: [row[field], new] for field, new in (("status", status), ("priority", priority))
                   if new is not None and new != row[field]}
        if not changes and not comment:
            raise TicketError("Nothing to change: give a new status, a new priority or a comment.")
        if changes:
            row = cur.execute(
                """UPDATE tickets SET status = coalesce(%s, status), priority = coalesce(%s, priority),
                   updated_at = now() WHERE id = %s RETURNING *""",
                (status, priority, row["id"]),
            ).fetchone()
        else:
            cur.execute("UPDATE tickets SET updated_at = now() WHERE id = %s", (row["id"],))
        cur.execute(
            """INSERT INTO ticket_events (ticket_id, action, changes, comment, actor, approved_by)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (row["id"], "updated" if changes else "commented", Jsonb(changes), comment, requester, approved_by),
        )
    return {**_summary(row), "changed": changes, **({"comment_added": comment} if comment else {})}


# ───────────────────────────── demo data ─────────────────────────────

DEMO_TICKETS = [
    # requester, title, description, category, priority, status, days ago, Service Desk comment
    ("alex", "Outlook keeps asking for my password", "Outlook prompts for my password every few minutes.",
     "Account and Password", "P3", "In Progress", 2, "We've cleared your cached credentials. Please restart Outlook."),
    ("alex", "Access to the finance dashboard", "I need read access to the finance dashboard for the Q4 review.",
     "Access Request", "P4", "Open", 1, None),
    ("alex", "Second monitor not detected", "My second monitor stopped working after the last update.",
     "Hardware", "P3", "Waiting on User", 4, "Please try a different cable and reply with the result."),
    ("jordan", "VPN-809 on hotel Wi-Fi", "KestrelConnect fails with VPN-809 at the hotel.",
     "Network/VPN", "P3", "Resolved", 6, "Hotel network blocks UDP 51820; a mobile hotspot works."),
    ("alex", "Teams camera not working", "Teams can't find my camera in meetings.",
     "Software", "P3", "Resolved", 9, "Camera permission for Teams was off; turned it back on."),
    ("jordan", "Suspicious invoice email", "Got an invoice email with a strange attachment. Reported it.",
     "Security Incident", "P2", "Closed", 12, "Confirmed phishing; the sender is blocked."),
]


def seed(conn: psycopg.Connection) -> list[str]:
    """Resets the tickets to the demo data and returns the new ticket numbers."""
    settings = get_settings()
    people = {"alex": settings.demo_user, "jordan": "jordan.lee@kestrel.example"}
    init_tickets(conn)
    numbers = []
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("TRUNCATE tickets, ticket_events RESTART IDENTITY")
        cur.execute("ALTER SEQUENCE ticket_number_seq RESTART WITH 1001")
        for person, title, description, category, priority, status, days, note in DEMO_TICKETS:
            ticket_id, number = cur.execute(
                """INSERT INTO tickets (requester, title, description, category, priority, status,
                                        created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, now() - make_interval(days => %s),
                           now() - make_interval(days => %s) + interval '3 hours')
                   RETURNING id, number""",
                (people[person], title, description, category, priority, status, days, days),
            ).fetchone()
            cur.execute("""INSERT INTO ticket_events (ticket_id, action, actor, approved_by, created_at)
                           VALUES (%s, 'created', %s, %s, now() - make_interval(days => %s))""",
                        (ticket_id, people[person], people[person], days))
            if note:
                cur.execute("""INSERT INTO ticket_events (ticket_id, action, comment, actor, created_at)
                               VALUES (%s, 'commented', %s, 'servicedesk@kestrel.example',
                                       now() - make_interval(days => %s) + interval '3 hours')""",
                            (ticket_id, note, days))
            numbers.append(number)
    return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["seed"])
    parser.parse_args()
    with psycopg.connect(get_settings().database_url) as conn:
        numbers = seed(conn)
    print(f"Seeded {len(numbers)} tickets: {', '.join(numbers)}")


if __name__ == "__main__":
    main()
