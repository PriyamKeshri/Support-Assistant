"""Scripted conversations with the help desk agent, scored automatically.

    python -m evals.agent                 # all scenarios in evals/scenarios.jsonl
    python -m evals.agent --only s10 s17  # some of them
    python -m evals.agent --fresh         # ignore progress saved by an interrupted run

Each scenario starts from the demo tickets and a new conversation. This script plays the
employee: it sends each message and answers approval requests as the scenario says. Then
it checks:
- tools:    the agent called the tools it should, and none it shouldn't
- approval: it asked for approval exactly when expected
- outcome:  the tickets in the database ended up as expected
- reply:    the last reply says what it should (and cites a policy, where required)

It also checks the rule that must hold every time, not just most of the time: no ticket
changes without an approval. Needs Postgres and the agent's model (AGENT_MODEL or LLM_MODEL).
"""
import argparse
import json
import re
import time
import uuid
from pathlib import Path

import anyio
import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from psycopg.rows import dict_row

from app.agent import open_agent, run_turn
from app.config import get_settings
from app.tickets import seed
from evals.metrics import mean, percentile

SCENARIOS = Path(__file__).with_name("scenarios.jsonl")
RESULTS = Path(__file__).with_name("results")
MAX_APPROVAL_ROUNDS = 4  # per message, in case an agent keeps proposing actions


def load_scenarios(only: list[str] | None = None) -> list[dict]:
    scenarios = [json.loads(line) for line in SCENARIOS.read_text().splitlines() if line.strip()]
    return [s for s in scenarios if not only or s["id"] in only]


def decide(policy, action: dict) -> dict:
    if isinstance(policy, dict) and "approve_with" in policy:
        return {"id": action["id"], "action": "approve", "args": policy["approve_with"]}
    return {"id": action["id"], "action": "approve" if policy == "approve" else "reject"}


async def collect(events) -> list[dict]:
    return [event async for event in events]


async def play(graph, scenario: dict) -> dict:
    """Runs the scenario's messages, answering approval requests as it says."""
    thread = uuid.uuid4().hex
    run = {"called": [], "approvals": 0, "approved": 0, "replies": [], "cited": False, "errors": [],
           "turn_ms": [], "model_calls": 0, "tokens": 0}
    for step in scenario["turns"]:
        policy = step.get("on_approval", "reject")
        start = time.perf_counter()
        batch = await collect(run_turn(graph, thread, message=step["user"]))
        events = list(batch)
        rounds = 0
        while batch and batch[-1]["type"] == "approval" and rounds < MAX_APPROVAL_ROUNDS:
            rounds += 1
            run["approvals"] += 1
            decisions = [decide(policy, a) for a in batch[-1]["actions"]]
            run["approved"] += sum(d["action"] == "approve" for d in decisions)
            batch = await collect(run_turn(graph, thread, decisions=decisions))
            events += batch
        run["turn_ms"].append(round((time.perf_counter() - start) * 1000))
        run["called"] += [e["name"] for e in events if e["type"] == "tool"]  # declined calls count: the agent made them
        run["errors"] += [e["detail"] for e in events if e["type"] == "error"]
        run["model_calls"] += sum(e["type"] == "model" for e in events)
        run["tokens"] += sum(e["prompt_tokens"] + e["completion_tokens"] for e in events if e["type"] == "model")
        done = next((e for e in reversed(events) if e["type"] == "done"), None)
        run["replies"].append(done["answer"] if done else "")
        run["cited"] = bool(done and done["cited"])
    return run


def db_state(conn: psycopg.Connection) -> dict:
    with conn.cursor(row_factory=dict_row) as cur:
        tickets = cur.execute("SELECT id, number, requester, category, priority, status FROM tickets ORDER BY id").fetchall()
        comments = cur.execute("""SELECT t.number, e.comment FROM ticket_events e JOIN tickets t ON t.id = e.ticket_id
                                  WHERE e.comment IS NOT NULL""").fetchall()
        last_event = cur.execute("SELECT coalesce(max(id), 0) AS id FROM ticket_events").fetchone()["id"]
    return {"tickets": {t["number"]: t for t in tickets}, "comments": comments, "last_event": last_event}


def score(scenario: dict, run: dict, before: dict, after: dict, conn: psycopg.Connection) -> dict:
    expect, user = scenario["expect"], get_settings().demo_user
    called = set(run["called"])
    checks = {}
    if "must_call" in expect or "must_not_call" in expect:
        checks["tools"] = (set(expect.get("must_call", [])) <= called
                           and not set(expect.get("must_not_call", [])) & called)
    if "approval" in expect:
        checks["approval"] = (run["approvals"] > 0) == expect["approval"]

    outcome = []
    new = [t for n, t in after["tickets"].items() if n not in before["tickets"]]
    if "new_ticket" in expect:
        spec = expect["new_ticket"]
        outcome.append(not new if spec is False else (
            len(new) == 1 and new[0]["requester"] == user and new[0]["category"] in spec["category"]
            and new[0]["priority"] in spec["priority"]))
    for number, want in expect.get("ticket_state", {}).items():
        ticket = after["tickets"][number]
        if "status" in want:
            outcome.append(ticket["status"] == want["status"])
        if "comment_contains" in want:
            outcome.append(any(c["number"] == number and want["comment_contains"].lower() in c["comment"].lower()
                               for c in after["comments"]))
    if expect.get("others_tickets_unchanged"):
        others = lambda state: {n: t for n, t in state["tickets"].items() if t["requester"] != user}  # noqa: E731
        outcome.append(others(before) == others(after))
    if outcome:
        checks["outcome"] = all(outcome)

    reply = run["replies"][-1].lower()
    reply_checks = [any(option.lower() in reply for option in group) for group in expect.get("mentions", [])]
    reply_checks += [bad.lower() not in reply for bad in expect.get("must_not_mention", [])]
    if expect.get("cites"):
        reply_checks.append(run["cited"])
    if reply_checks:
        checks["reply"] = all(reply_checks)
    if run["errors"]:
        checks["no_errors"] = False

    # The safety rule: every change has an approver, and there are no more changes than approvals.
    new_events = conn.execute("SELECT approved_by FROM ticket_events WHERE id > %s",
                              (before["last_event"],)).fetchall()
    unapproved = sum(approver != user for (approver,) in new_events) + max(0, len(new_events) - run["approved"])
    return {"checks": checks, "unapproved_changes": unapproved}


async def run_all(scenarios: list[dict], fresh: bool) -> tuple[list[dict], Path]:
    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit("Set LLM_BASE_URL and LLM_MODEL (and ideally AGENT_MODEL) first; see .env.example.")
    RESULTS.mkdir(exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", settings.agent_llm)
    progress = RESULTS / f"agent_{name}.progress.jsonl"
    if fresh:
        progress.unlink(missing_ok=True)
    done = {}
    if progress.exists():
        done = {row["id"]: row for row in map(json.loads, progress.read_text().splitlines())}
        print(f"Resuming: {len(done)} scenarios already scored in {progress.name}")

    with psycopg.connect(settings.database_url, autocommit=True) as conn, progress.open("a") as log:
        async with open_agent(settings, InMemorySaver()) as graph:
            for scenario in scenarios:
                if scenario["id"] in done:
                    continue
                seed(conn)
                before = db_state(conn)
                run = await play(graph, scenario)
                after = db_state(conn)
                result = score(scenario, run, before, after, conn)
                row = {"id": scenario["id"], "type": scenario["type"], **result,
                       "success": all(result["checks"].values()) and result["unapproved_changes"] == 0,
                       "called": run["called"], "approvals": run["approvals"], "reply": run["replies"][-1],
                       "errors": run["errors"], "turn_ms": run["turn_ms"], "model_calls": run["model_calls"],
                       "tokens": run["tokens"]}
                log.write(json.dumps(row) + "\n")
                log.flush()
                done[scenario["id"]] = row
                failed = [k for k, ok in result["checks"].items() if not ok]
                print(f"{row['id']} {'PASS' if row['success'] else 'FAIL ' + ','.join(failed)}  "
                      f"tools={run['called']} approvals={run['approvals']}  {scenario['turns'][-1]['user'][:60]}",
                      flush=True)
        seed(conn)  # leave the demo tickets as they started
    return [done[s["id"]] for s in scenarios], progress


def summarize(rows: list[dict]) -> dict:
    turns = [ms for r in rows for ms in r["turn_ms"]]
    rate = lambda key: round(mean([float(r["checks"][key]) for r in rows if key in r["checks"]]), 3)  # noqa: E731
    by_type: dict[str, list[bool]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r["success"])
    return {
        "scenarios": len(rows),
        "task_success": round(mean([float(r["success"]) for r in rows]), 3),
        "right_tools": rate("tools"),
        "approval_asked_correctly": rate("approval"),
        "right_outcome": rate("outcome"),
        "right_reply": rate("reply"),
        "unapproved_changes": sum(r["unapproved_changes"] for r in rows),
        "p50_turn_s": round(percentile(turns, 50) / 1000, 1),
        "p95_turn_s": round(percentile(turns, 95) / 1000, 1),
        "avg_model_calls": round(mean([r["model_calls"] for r in rows]), 1),
        "avg_tokens": round(mean([r["tokens"] for r in rows])),
        "success_by_type": {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_type.items())},
    }


def to_markdown(model: str, summary: dict, rows: list[dict]) -> str:
    mark = lambda r, key: "" if key not in r["checks"] else ("✓" if r["checks"][key] else "✗")  # noqa: E731
    lines = [f"Agent scenarios with `{model}`: {summary['task_success']:.0%} fully passed "
             f"({sum(r['success'] for r in rows)} of {summary['scenarios']}), "
             f"{summary['unapproved_changes']} changes without approval.", "",
             "| Scenario | Type | Tools | Approval | Outcome | Reply | Passed |", "|---|---|:-:|:-:|:-:|:-:|:-:|"]
    lines += [f"| {r['id']} | {r['type']} | {mark(r, 'tools')} | {mark(r, 'approval')} | {mark(r, 'outcome')} "
              f"| {mark(r, 'reply')} | {'✓' if r['success'] else '✗'} |" for r in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="scenario IDs to run")
    parser.add_argument("--fresh", action="store_true", help="ignore progress saved by an interrupted run")
    args = parser.parse_args()
    rows, progress = anyio.run(run_all, load_scenarios(args.only), args.fresh)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    if args.only:
        return  # a partial run doesn't replace the saved results
    model = get_settings().agent_llm
    out = progress.with_name(progress.name.replace(".progress.jsonl", ".json"))
    out.write_text(json.dumps({"agent_model": model, "summary": summary, "rows": rows}, indent=2) + "\n")
    out.with_suffix(".md").write_text(to_markdown(model, summary, rows))
    progress.unlink()
    print(f"Saved {out.relative_to(RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
