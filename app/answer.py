"""Answers a question from retrieved policy excerpts, with numbered citations."""
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.config import Settings
from app.search import Hit

NO_ANSWER = "I couldn't find this in the IT policies. Please raise a ticket at help.kestrel.example."

SYSTEM_PROMPT = f"""You are the IT help desk assistant for Kestrel Analytics.
Answer the employee's question using only the numbered policy excerpts provided.

Rules:
- Support every statement with the excerpt number in square brackets, like [2].
- If the excerpts do not answer the question, reply with exactly this sentence and nothing else:
  {NO_ANSWER}
- Keep it short: at most five sentences, or a short list of steps.
- Never invent policy details, numbers, contacts or links."""

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2], or [1, 3]


@dataclass
class Answer:
    text: str
    cited: list[int]    # excerpt numbers the answer cites, as in its [n] markers
    context: list[Hit]  # the excerpts the model was given, numbered from 1
    abstained: bool
    generation_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def citations(self) -> list[Hit]:
        return [self.context[n - 1] for n in self.cited]

    def cost_usd(self, settings: Settings) -> float:
        return (self.prompt_tokens * settings.llm_input_price_per_mtok
                + self.completion_tokens * settings.llm_output_price_per_mtok) / 1_000_000


def is_abstention(text: str) -> bool:
    return "couldn't find this in the it policies" in text.lower().replace("’", "'")


@lru_cache
def get_client(base_url: str, api_key: str) -> OpenAI:
    # Local servers such as Ollama ignore the key, but the client requires one.
    return OpenAI(base_url=base_url, api_key=api_key or "not-needed")


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.doc_title} ({h.doc_code}), section {h.section_no}. {h.section_title}, page {h.page}\n{h.content}"
        for i, h in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Policy excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"},
    ]


def finish_answer(raw_text: str, hits: list[Hit], generation_ms: float, usage) -> Answer:
    """Turns the model's raw reply into an Answer: drops reasoning blocks and collects the
    citations that point at a real excerpt."""
    text = THINK_BLOCK.sub("", raw_text).strip()
    return Answer(
        text=text,
        cited=sorted({int(n) for group in CITATION.findall(text) for n in group.split(",")
                      if 1 <= int(n) <= len(hits)}),
        context=hits,
        abstained=is_abstention(text),
        generation_ms=generation_ms,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


def generate_answer(question: str, hits: list[Hit], settings: Settings) -> Answer:
    client = get_client(settings.llm_base_url, settings.llm_api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.llm_model, temperature=0, messages=build_messages(question, hits))
    elapsed = (time.perf_counter() - start) * 1000
    return finish_answer(response.choices[0].message.content or "", hits, elapsed, response.usage)


class AnswerStream:
    """Iterating yields the answer text as the model writes it. Once iteration ends,
    `answer` holds the finished Answer, built exactly as generate_answer builds it."""

    def __init__(self, question: str, hits: list[Hit], settings: Settings):
        self.question, self.hits, self.settings = question, hits, settings
        self.answer: Answer | None = None

    def __iter__(self) -> Iterator[str]:
        client = get_client(self.settings.llm_base_url, self.settings.llm_api_key)
        start = time.perf_counter()
        chunks = client.chat.completions.create(
            model=self.settings.llm_model,
            temperature=0,
            messages=build_messages(self.question, self.hits),
            stream=True,
            stream_options={"include_usage": True},  # token counts arrive with the last chunk
        )
        parts, usage = [], None
        for chunk in chunks:
            usage = chunk.usage or usage
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
                yield delta
        self.answer = finish_answer("".join(parts), self.hits, (time.perf_counter() - start) * 1000, usage)
