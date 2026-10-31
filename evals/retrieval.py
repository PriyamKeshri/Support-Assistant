"""Compares the four retrieval modes on the labelled question set.

    python -m evals.retrieval

Writes evals/results/retrieval.json (per-question detail) and retrieval.md (the tables
the README quotes). Needs Postgres with the policies ingested; no API key needed.
"""
import argparse
import json
import platform
from collections import defaultdict
from datetime import date
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.models import get_embedder, get_reranker
from app.search import Mode, timed_search
from evals.metrics import (hit_at_k, load_questions, mean, ndcg_at_k, percentile, recall_at_k,
                           reciprocal_rank, unique_in_order)

RESULTS = Path(__file__).with_name("results")
DEPTH = 10  # chunks retrieved per question; metrics are cut at 1, 5 and 10


def score(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    return {
        "hit@1": hit_at_k(ranked, relevant, 1),
        "recall@5": recall_at_k(ranked, relevant, 5),
        "mrr@10": reciprocal_rank(ranked, relevant, 10),
        "ndcg@10": ndcg_at_k(ranked, relevant, 10),
    }


def run(modes: list[Mode]) -> dict:
    settings = get_settings()
    embedder, reranker = get_embedder(), get_reranker()
    questions = [q for q in load_questions() if q.answerable]
    report = {
        "date": date.today().isoformat(),
        "machine": f"{platform.system()} {platform.machine()}",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "questions": len(questions),
        "modes": {},
    }
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        for mode in modes:
            timed_search(conn, "warm-up query", mode, DEPTH, settings, embedder, reranker)
            rows, latencies, by_type = [], [], defaultdict(list)
            for q in questions:
                hits, ms = timed_search(conn, q.question, mode, DEPTH, settings, embedder, reranker)
                ranked = unique_in_order([h.label for h in hits])
                metrics = score(ranked, set(q.relevant))
                latencies.append(ms)
                by_type[q.type].append(metrics["recall@5"])
                rows.append({"id": q.id, "type": q.type, "relevant": q.relevant, "top5": ranked[:5],
                             "latency_ms": round(ms, 1), **metrics})
            report["modes"][mode.value] = {
                "summary": {
                    **{m: round(mean([r[m] for r in rows]), 3) for m in ("hit@1", "recall@5", "mrr@10", "ndcg@10")},
                    "p50_ms": round(percentile(latencies, 50), 1),
                    "p95_ms": round(percentile(latencies, 95), 1),
                },
                "recall@5_by_type": {t: round(mean(v), 3) for t, v in sorted(by_type.items())},
                "questions": rows,
            }
    return report


def to_markdown(report: dict) -> str:
    modes = report["modes"]
    types = sorted({t for m in modes.values() for t in m["recall@5_by_type"]})
    counts = defaultdict(int)
    for row in next(iter(modes.values()))["questions"]:
        counts[row["type"]] += 1
    lines = [
        f"Retrieval on {report['questions']} labelled questions · {report['date']} · {report['machine']}",
        f"Embeddings: `{report['embedding_model']}` · reranker: `{report['reranker_model']}`",
        "",
        "| Mode | Hit@1 | Recall@5 | MRR@10 | nDCG@10 | p50 latency | p95 latency |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in modes.items():
        s = m["summary"]
        lines.append(f"| {name} | {s['hit@1']:.2f} | {s['recall@5']:.2f} | {s['mrr@10']:.2f} | {s['ndcg@10']:.2f} "
                     f"| {s['p50_ms']:.1f} ms | {s['p95_ms']:.1f} ms |")
    lines += [
        "",
        "Recall@5 by question type:",
        "",
        "| Mode | " + " | ".join(f"{t} ({counts[t]})" for t in types) + " |",
        "|---|" + "---|" * len(types),
    ]
    for name, m in modes.items():
        lines.append(f"| {name} | " + " | ".join(f"{m['recall@5_by_type'].get(t, 0):.2f}" for t in types) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", choices=[m.value for m in Mode], default=[m.value for m in Mode])
    args = parser.parse_args()
    report = run([Mode(m) for m in args.modes])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = to_markdown(report)
    (RESULTS / "retrieval.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
