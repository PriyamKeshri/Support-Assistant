"""The help desk's tools, served over the Model Context Protocol (MCP).

The agent connects to this server in-process. Any MCP client, such as Claude Desktop or the
MCP Inspector, can run it over stdio:

    python -m app.mcp_server

Tools that only read are annotated read-only. The agent asks the employee to approve every
other tool call before running it, so adding a new tool can't skip that step by accident.

`requester` and `approved_by` identify the employee. The agent fills them in from the
session and hides them from the model, so the model can't act as someone else.
"""
from functools import lru_cache
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from psycopg_pool import ConnectionPool
from pydantic import Field

from app import tickets
from app.config import get_settings
from app.db import create_pool
from app.models import get_embedder, get_reranker
from app.search import Mode, search
from app.tickets import Category, EmployeeStatus, Priority, Status, TicketError

server = MCPServer(
    "kestrel-helpdesk",
    instructions="IT help desk tools for Kestrel Analytics: search the IT policies, and look up, "
                 "create or update the employee's own help desk tickets.",
)

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
CHANGES_DATA = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

Requester = Annotated[str, Field(description="Email of the employee the request is for.")]
ApprovedBy = Annotated[str, Field(description="Email of the person who approved this change.")]
TicketNumber = Annotated[str, Field(description="Ticket number, for example INC-1001.")]


@lru_cache
def pool() -> ConnectionPool:
    return create_pool(get_settings().database_url)


def _ticket_call(fn, *args, **kwargs):
    """Runs a ticket operation, turning a refusal into an error the model can read and act on."""
    try:
        with pool().connection() as conn:
            return fn(conn, *args, **kwargs)
    except TicketError as exc:
        raise ToolError(str(exc)) from exc


@server.tool(annotations=READ_ONLY)
def search_policies(
    query: Annotated[str, Field(description="What to look for, in the employee's words or as keywords.")],
    k: Annotated[int, Field(ge=1, le=8, description="How many sections to return.")] = 5,
) -> dict:
    """Search Kestrel's IT policies with hybrid keyword and semantic search, reranked.

    Returns the best-matching policy sections, best first, each with its policy code,
    section, page and text."""
    settings = get_settings()
    with pool().connection() as conn:
        hits = search(conn, query, Mode.rerank, k, settings, get_embedder(), get_reranker())
    return {"results": [{"policy": h.doc_code, "title": h.doc_title, "section": f"{h.section_no}. {h.section_title}",
                         "page": h.page, "text": h.content} for h in hits]}


@server.tool(annotations=READ_ONLY)
def list_my_tickets(requester: Requester, status: Status | None = None) -> dict:
    """List the employee's own help desk tickets, most recently updated first, optionally
    only those with one status."""
    return {"tickets": _ticket_call(tickets.list_tickets, requester, status)}


@server.tool(annotations=READ_ONLY)
def get_ticket(requester: Requester, number: TicketNumber) -> dict:
    """Get one of the employee's tickets, with its description and full history of changes
    and Service Desk comments."""
    return _ticket_call(tickets.get_ticket, requester, number)


@server.tool(annotations=CHANGES_DATA)
def create_ticket(
    requester: Requester,
    approved_by: ApprovedBy,
    title: Annotated[str, Field(description="Short summary, 3 to 120 characters.")],
    description: Annotated[str, Field(description="What happened, in the employee's words, with any error codes.")],
    category: Category,
    priority: Priority,
) -> dict:
    """Create a help desk ticket for the employee.

    Category, from the Service Desk standard (IT-SVC-006):
    - Account and Password: signing in, passwords, MFA, locked accounts.
    - Access Request: access to a system, application, data or shared mailbox.
    - Hardware: laptops, monitors, keyboards and other devices, including lost or stolen ones.
    - Software: applications that crash or misbehave, and installing software.
    - Network/VPN: the VPN, Wi-Fi and connectivity.
    - Email and Collaboration: Outlook, Teams, mailboxes and calendars.
    - Security Incident: phishing, malware, a compromised account or a data breach.

    Priority, from the Service Desk standard (IT-SVC-006):
    - P1 Critical: a company-wide outage, a security incident in progress, or a lost or stolen device.
    - P2 High: a team or a business-critical process is blocked.
    - P3 Normal: one person is affected and a workaround exists.
    - P4 Low: service requests and questions.
    Security Incident tickets are always P1 or P2."""
    return _ticket_call(tickets.create_ticket, requester, approved_by, title, description, category, priority)


@server.tool(annotations=CHANGES_DATA)
def update_ticket(
    requester: Requester,
    approved_by: ApprovedBy,
    number: TicketNumber,
    status: Annotated[EmployeeStatus | None, Field(description="Closed to close the ticket, Open to reopen it.")] = None,
    priority: Priority | None = None,
    comment: Annotated[str | None, Field(description="A note to add to the ticket.")] = None,
) -> dict:
    """Change one of the employee's tickets: close or reopen it, change its priority, or add
    a comment. Give at least one of these."""
    return _ticket_call(tickets.update_ticket, requester, approved_by, number, status, priority, comment)


if __name__ == "__main__":
    server.run()
