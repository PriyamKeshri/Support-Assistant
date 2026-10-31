import pytest

from evals.answers import summarize


def row(type_="lookup", abstained=False, correct=True, grounded=True, ms=100.0):
    return {"type": type_, "abstained": abstained, "correct": None if abstained else correct,
            "grounded": None if abstained else grounded, "judge_parsed": None if abstained else True,
            "retrieval_ms": ms, "generation_ms": ms, "prompt_tokens": 300, "completion_tokens": 50,
            "cost_usd": 0.0}


def test_summary_counts_refusals_and_hallucinations():
    rows = [
        row(),                                            # right and grounded
        row(correct=False, grounded=False),               # wrong and made something up
        row(abstained=True),                              # refused an answerable question
        row("unanswerable", abstained=True),              # correctly declined
        row("unanswerable", correct=False, grounded=False),  # answered what it shouldn't
    ]
    s = summarize(rows)
    assert s["correctness"] == pytest.approx(1 / 3, abs=1e-3)       # refusals count as wrong
    assert s["false_refusal_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert s["abstention_on_unanswerable"] == 0.5
    assert s["groundedness"] == pytest.approx(1 / 3, abs=1e-3)       # over the 3 answers actually given
    assert s["hallucination_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["p50_total_ms"] == 200.0
