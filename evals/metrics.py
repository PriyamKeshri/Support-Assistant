"""Retrieval metrics over ranked lists of section labels (e.g. "IT-OPS-003#4").

A relevant section counts once, however many of its chunks were retrieved,
so a ranked list of chunks is first reduced to unique sections in order."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

QUESTIONS = Path(__file__).with_name("questions.jsonl")


@dataclass
class Question:
    id: str
    type: str
    question: str
    relevant: list[str]
    answer: str

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [Question(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def unique_in_order(labels: list[str]) -> list[str]:
    return list(dict.fromkeys(labels))


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(label in relevant for label in ranked[:k]))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    for i, label in enumerate(ranked[:k], start=1):
        if label in relevant:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, label in enumerate(ranked[:k]) if label in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 100]."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
