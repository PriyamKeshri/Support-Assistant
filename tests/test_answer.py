from types import SimpleNamespace

from app.answer import NO_ANSWER, AnswerStream, finish_answer, generate_answer
from app.config import Settings
from app.search import Hit

HITS = [Hit(i, "IT-OPS-004", "Device and Endpoint Policy", i, f"Section {i}", 1, f"text {i}", 1.0)
        for i in (1, 2, 3)]
SETTINGS = Settings(llm_base_url="http://fake", llm_model="fake",
                    llm_input_price_per_mtok=1.0, llm_output_price_per_mtok=2.0)
USAGE = SimpleNamespace(prompt_tokens=100, completion_tokens=20)


def test_citations_are_collected_and_checked():
    answer = finish_answer("Report it [2]. Then lock it [1, 3]. Again [2] and an invalid [7].", HITS, 5.0, USAGE)
    assert answer.cited == [1, 2, 3]
    assert [h.section_no for h in answer.citations] == [1, 2, 3]
    assert not answer.abstained


def test_reasoning_blocks_are_dropped():
    answer = finish_answer("<think>Which excerpt fits?</think>\nUse Company Portal [1].", HITS, 5.0, None)
    assert answer.text == "Use Company Portal [1]."
    assert answer.prompt_tokens == 0


def test_abstention_is_detected():
    answer = finish_answer(NO_ANSWER, HITS, 5.0, USAGE)
    assert answer.abstained
    assert answer.cited == []


def test_generate_answer_reports_tokens_and_cost(fake_llm):
    fake_llm("Report it within 2 hours [1].")
    answer = generate_answer("I lost my laptop", HITS, SETTINGS)
    assert answer.cited == [1]
    assert (answer.prompt_tokens, answer.completion_tokens) == (120, 30)
    assert answer.cost_usd(SETTINGS) == (120 * 1.0 + 30 * 2.0) / 1_000_000


def test_stream_yields_pieces_then_builds_the_same_answer(fake_llm):
    reply = "Switch to a mobile hotspot [2]. Hotel networks often block the port [2]."
    fake = fake_llm(reply)
    stream = AnswerStream("What does VPN-809 mean?", HITS, SETTINGS)
    pieces = list(stream)
    assert len(pieces) > 1 and "".join(pieces) == reply
    assert stream.answer.text == reply
    assert stream.answer.cited == [2]
    assert stream.answer.completion_tokens == 30
    assert fake.calls[0]["stream"] is True
