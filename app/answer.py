"""Answers a question from retrieved policy excerpts, with numbered citations."""
import re
import time
from dataclasses import dataclass, field
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
CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    citations: list[Hit]
    abstained: bool
    generation_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    context: list[Hit] = field(default_factory=list)

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


def generate_answer(question: str, hits: list[Hit], settings: Settings) -> Answer:
    client = get_client(settings.llm_base_url, settings.llm_api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    elapsed = (time.perf_counter() - start) * 1000
    text = THINK_BLOCK.sub("", response.choices[0].message.content or "").strip()
    cited = sorted({int(n) for n in CITATION.findall(text) if 1 <= int(n) <= len(hits)})
    usage = response.usage
    return Answer(
        text=text,
        citations=[hits[n - 1] for n in cited],
        abstained=is_abstention(text),
        generation_ms=elapsed,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        context=hits,
    )
