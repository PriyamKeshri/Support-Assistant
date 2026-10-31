import math

import pytest

from evals.metrics import (hit_at_k, load_questions, ndcg_at_k, percentile, recall_at_k, reciprocal_rank,
                           unique_in_order)

RANKED = ["A#1", "B#2", "C#3", "D#4"]


def test_unique_in_order_keeps_first_occurrence():
    assert unique_in_order(["A#1", "B#2", "A#1", "C#3"]) == ["A#1", "B#2", "C#3"]


def test_hit_and_recall():
    assert hit_at_k(RANKED, {"B#2"}, 1) == 0.0
    assert hit_at_k(RANKED, {"B#2"}, 2) == 1.0
    assert recall_at_k(RANKED, {"B#2", "Z#9"}, 5) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"C#3"}, 10) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"C#3"}, 2) == 0.0


def test_ndcg():
    assert ndcg_at_k(RANKED, {"A#1"}, 10) == 1.0
    assert ndcg_at_k(RANKED, {"B#2"}, 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(RANKED, {"A#1", "B#2"}, 10) == 1.0


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile([7.0], 95) == 7.0


def test_question_set_is_well_formed():
    questions = load_questions()
    assert len({q.id for q in questions}) == len(questions)
    assert len(questions) >= 60
    for q in questions:
        assert q.type in {"lookup", "paraphrase", "exact_term", "multi_section", "unanswerable"}
        assert q.answerable == (q.type != "unanswerable")
        assert all("#" in label for label in q.relevant)
