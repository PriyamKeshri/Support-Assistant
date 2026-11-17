"""Shared test doubles and parsers."""
import json
from types import SimpleNamespace


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))],
                           usage=None)


class ScriptedLLM:
    """An async, streaming stand-in for the OpenAI client that replays scripted replies in
    order. A reply is text, or a list of (tool name, arguments) tool calls."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)

        async def stream():
            if isinstance(reply, str):
                for i in range(0, len(reply), 8):
                    yield _chunk(content=reply[i:i + 8])
            else:
                for i, (name, args) in enumerate(reply):
                    arguments = args if isinstance(args, str) else json.dumps(args)
                    function = SimpleNamespace(name=name, arguments=arguments)
                    yield _chunk(tool_calls=[SimpleNamespace(index=i, id=f"call_{len(self.requests)}_{i}",
                                                             function=function)])
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=50, completion_tokens=10))
        return stream()


def parse_events(body: str) -> list[tuple[str, object]]:
    """Splits a server-sent events body into (event name, decoded data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((fields["event"], json.loads(fields["data"])))
    return events
