"""Measures answer quality end to end: retrieval (rerank mode) + generation + an LLM judge.

    python -m evals.answers                 # all questions
    python -m evals.answers --limit 10      # quick smoke run

Needs LLM_BASE_URL and LLM_MODEL. JUDGE_MODEL defaults to LLM_MODEL; a stronger judge
than the model being graded gives more trustworthy scores.

What it reports:
- correctness:     answerable questions whose answer matches the reference (refusals count as wrong)
- groundedness:    answers whose every claim is supported by the retrieved excerpts
- hallucination:   1 - groundedness
- false refusals:  answerable questions the assistant declined
- abstention:      unanswerable questions the assistant correctly declined
- latency, tokens and cost per question
"""
import argparse
import json
import re
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.answer import format_context, generate_answer, get_client
from app.config import get_settings
from app.models import get_embedder, get_reranker
from app.search import Mode, timed_search
from evals.metrics import load_questions, mean, percentile

RESULTS = Path(__file__).with_name("results")
CONTEXT_K = 5

JUDGE_PROMPT = """You are grading an IT help desk assistant.

Policy excerpts the assistant was given:
{context}

Question: {question}
Reference answer: {reference}
Assistant's answer: {answer}

Reply with a JSON object and nothing else:
{{"grounded": true or false,       // every factual claim in the assistant's answer is supported by the excerpts
  "unsupported_claims": [strings], // the claims that are not supported, empty if none
  "correct": true or false}}       // the assistant's answer agrees with the reference answer on the key facts"""


def judge(question, reference: str, answer_text: str, context: str, settings) -> dict:
    client = get_client(settings.llm_base_url, settings.llm_api_key)
    response = client.chat.completions.create(
        model=settings.judge_model or settings.llm_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            context=context, question=question, reference=reference or "(not answerable from the policies)",
            answer=answer_text)}],
    )
    raw = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    return {
        "grounded": bool(verdict.get("grounded", False)),
        "unsupported_claims": verdict.get("unsupported_claims", []),
        "correct": bool(verdict.get("correct", False)),
        "judge_parsed": bool(verdict),
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in rows if not r["abstained"]]
    totals = [r["retrieval_ms"] + r["generation_ms"] for r in rows]
    groundedness = mean([float(r["grounded"]) for r in answered])
    return {
        "correctness": round(mean([float(bool(r["correct"])) for r in answerable]), 3),
        "groundedness": round(groundedness, 3),
        "hallucination_rate": round(1 - groundedness, 3) if answered else 0.0,
        "false_refusal_rate": round(mean([float(r["abstained"]) for r in answerable]), 3),
        "abstention_on_unanswerable": round(mean([float(r["abstained"]) for r in unanswerable]), 3),
        "p50_total_ms": round(percentile(totals, 50), 1),
        "p95_total_ms": round(percentile(totals, 95), 1),
        "avg_prompt_tokens": round(mean([r["prompt_tokens"] for r in rows])),
        "avg_completion_tokens": round(mean([r["completion_tokens"] for r in rows])),
        "avg_cost_usd": round(mean([r["cost_usd"] for r in rows]), 6),
        "unparsed_judge_replies": sum(1 for r in answered if not r["judge_parsed"]),
    }


def run(limit: int | None, fresh: bool) -> tuple[dict, Path]:
    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit("Set LLM_BASE_URL and LLM_MODEL first (see .env.example).")
    judge_model = settings.judge_model or settings.llm_model
    embedder, reranker = get_embedder(), get_reranker()
    questions = load_questions()[:limit] if limit else load_questions()

    # Each graded question is appended to a progress file straight away, so a run that dies
    # halfway (model server restarted, laptop asleep) resumes where it stopped.
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", settings.llm_model if judge_model == settings.llm_model
                  else f"{settings.llm_model}_judged_by_{judge_model}")
    RESULTS.mkdir(exist_ok=True)
    progress = RESULTS / f"answers_{name}.progress.jsonl"
    if fresh:
        progress.unlink(missing_ok=True)
    done = {}
    if progress.exists():
        done = {row["id"]: row for row in map(json.loads, progress.read_text().splitlines())}
        print(f"Resuming: {len(done)} questions already graded in {progress.name}")

    with psycopg.connect(settings.database_url) as conn, progress.open("a") as log:
        register_vector(conn)
        for q in questions:
            if q.id in done:
                continue
            hits, retrieval_ms = timed_search(conn, q.question, Mode.rerank, CONTEXT_K, settings, embedder, reranker)
            answer = generate_answer(q.question, hits, settings)
            verdict = {"grounded": None, "unsupported_claims": [], "correct": None, "judge_parsed": None}
            if not answer.abstained:
                verdict = judge(q.question, q.answer, answer.text, format_context(hits), settings)
            row = {
                "id": q.id, "type": q.type, "question": q.question, "answer": answer.text,
                "abstained": answer.abstained, **verdict,
                "retrieval_ms": round(retrieval_ms, 1), "generation_ms": round(answer.generation_ms, 1),
                "prompt_tokens": answer.prompt_tokens, "completion_tokens": answer.completion_tokens,
                "cost_usd": answer.cost_usd(settings),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            done[q.id] = row
            print(f"{q.id} {'ABSTAIN' if answer.abstained else 'answer '} "
                  f"correct={verdict['correct']} grounded={verdict['grounded']}  {q.question}", flush=True)

    rows = [done[q.id] for q in questions]
    report = {"llm_model": settings.llm_model, "judge_model": judge_model, "questions": len(rows),
              "summary": summarize(rows), "rows": rows}
    return report, progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--fresh", action="store_true", help="ignore progress saved by an interrupted run")
    args = parser.parse_args()
    report, progress = run(args.limit, args.fresh)
    out = progress.with_name(progress.name.replace(".progress.jsonl", ".json"))
    out.write_text(json.dumps(report, indent=2) + "\n")
    progress.unlink()
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {out.relative_to(RESULTS.parent.parent)}")


if __name__ == "__main__":
    main()
