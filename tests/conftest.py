from types import SimpleNamespace

import pytest


class FakeLLM:
    """Stands in for the OpenAI client: replies with fixed text, streamed in small pieces."""

    def __init__(self, reply: str):
        self.reply = reply
        self.chat = SimpleNamespace(completions=self)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
        if not kwargs.get("stream"):
            message = SimpleNamespace(content=self.reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)
        pieces = [self.reply[i:i + 7] for i in range(0, len(self.reply), 7)]
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))], usage=None)
                  for p in pieces]
        return iter(chunks + [SimpleNamespace(choices=[], usage=usage)])  # usage comes last, as in OpenAI's API


@pytest.fixture
def fake_llm(monkeypatch):
    """Call with the reply text; the model client is replaced for the rest of the test."""
    def install(reply: str) -> FakeLLM:
        fake = FakeLLM(reply)
        monkeypatch.setattr("app.answer.get_client", lambda base_url, api_key: fake)
        return fake
    return install
