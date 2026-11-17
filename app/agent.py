"""The help desk agent: a LangGraph state machine around one model and the MCP tools.

    agent ──calls a tool that changes data──> review ──> tools ──> agent ...
      │   └──calls only read-only tools─────────────────> tools ──> agent ...
      └──replies──> done

`review` pauses the graph until the employee approves or rejects each call to a tool that
isn't read-only. `tools` runs read-only calls straight away, and any other call only when an
approval is recorded for that exact call, so a model that skips or fakes the approval step
changes nothing. When every proposed change is rejected, `declined` replies with fixed text
instead of asking the model, so a rejection is never reported as done. A reply that cites
policy excerpts nobody retrieved is withdrawn: the agent searches the policies itself and the
model answers again. Conversations, including a pending approval, are saved by the checkpointer.
"""
import json
import operator
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from mcp import Client
from openai import AsyncOpenAI

from app.config import Settings

INJECTED = ("requester", "approved_by")  # filled in from the session, never by the model
HIDDEN = (*INJECTED, "k")                # not offered to the model; search uses its default result count
MAX_HISTORY = 40      # most recent messages sent to the model
RECURSION_LIMIT = 24  # graph steps per turn, about eight rounds of tool calls
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
GROUNDING_CALL = "ground_"  # ID prefix of searches the agent runs itself, after a reply cited nothing real
NOT_APPROVED = "Not done: the employee did not approve this action, so nothing was changed."
MOVED_ON = "Not done: the employee sent a new message instead of approving this action."


def system_prompt(settings: Settings) -> str:
    return f"""You are the IT help desk agent for Kestrel Analytics, helping {settings.demo_user_name} \
({settings.demo_user}). Today is {date.today():%A %d %B %Y}.

You can search Kestrel's IT policies, and look up, create and update this employee's help desk tickets.

- For questions about IT rules, deadlines or procedures, and for "how do I fix this?" questions, call \
search_policies first and answer only from what it returns, with the steps it gives. Cite each statement with \
the excerpt number in square brackets, like [2]. If the policies don't cover the question, say so and offer to \
raise a ticket. Never invent policy details, numbers or contacts.
- Call create_ticket when the employee asks for a ticket, or when the policies say the problem must be reported, \
such as a lost or stolen device or a security incident. Use a short title, a description in their words, and \
the category and priority the Service Desk standard sets out (see the tool). Don't ask them to confirm first: \
the app shows them the details and asks for approval before anything is saved.
- For a lost or stolen device or a security incident, call search_policies first to find what the employee \
must do themselves, then call create_ticket. Once the ticket exists, tell them those steps, with citations.
- Use list_my_tickets and get_ticket for questions about their tickets. Ticket numbers look like INC-1001.
- Keep replies short: at most five sentences, or a short list."""


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]  # the conversation, in OpenAI's message format
    sources: Annotated[list[dict], operator.add]   # policy excerpts found so far, numbered across the conversation
    decisions: dict                                 # the employee's decision for each call awaiting approval


class ToolBox:
    """The MCP server's tools: as the model sees them, and as the agent calls them."""

    def __init__(self, client: Client, tools: list):
        self.client = client
        self.tools = {t.name: t for t in tools}

    @classmethod
    @asynccontextmanager
    async def connect(cls, server=None):
        if server is None:
            from app.mcp_server import server  # loads the search models, so only when needed
        async with Client(server) as client:
            yield cls(client, (await client.list_tools()).tools)

    def needs_approval(self, name: str) -> bool:
        tool = self.tools.get(name)
        return tool is not None and not (tool.annotations and tool.annotations.read_only_hint)

    def visible_params(self, name: str) -> set[str]:
        return set(self.tools[name].input_schema.get("properties", {})) - set(HIDDEN)

    def openai_schemas(self) -> list[dict]:
        schemas = []
        for tool in self.tools.values():
            params = json.loads(json.dumps(tool.input_schema))
            for name in HIDDEN:
                params.get("properties", {}).pop(name, None)
            params["required"] = [r for r in params.get("required", []) if r not in HIDDEN]
            schemas.append({"type": "function", "function": {
                "name": tool.name, "description": tool.description or "", "parameters": params}})
        return schemas

    async def call(self, name: str, args: dict, *, requester: str, approved_by: str = "") -> tuple[bool, Any]:
        params = self.tools[name].input_schema.get("properties", {})
        arguments = {k: v for k, v in args.items() if k not in HIDDEN}
        if "requester" in params:
            arguments["requester"] = requester
        if "approved_by" in params:
            arguments["approved_by"] = approved_by
        result = await self.client.call_tool(name, arguments)
        text = "".join(getattr(part, "text", "") for part in result.content)
        if result.is_error:
            return False, text
        try:
            return True, json.loads(text)
        except json.JSONDecodeError:
            return True, text


def describe(name: str, args: dict) -> str:
    """One line telling the employee what an action will do."""
    if name == "create_ticket":
        return f"Create a {args.get('priority', '?')} {args.get('category', '?')} ticket: {args.get('title', '')}"
    if name == "update_ticket":
        parts = [f"set status to {args['status']}" if args.get("status") else "",
                 f"set priority to {args['priority']}" if args.get("priority") else "",
                 f'add the comment "{args["comment"]}"' if args.get("comment") else ""]
        return f"Update {args.get('number', '?')}: " + ", ".join(p for p in parts if p)
    return f"Run {name}"


def declined_reply(calls: list[dict]) -> str:
    """What to say when the employee rejected every proposed change. Written here rather than by
    the model, so a rejection can never be reported as done."""
    parts = []
    for call in calls:
        try:
            number = json.loads(call["function"]["arguments"] or "{}").get("number")
        except json.JSONDecodeError:
            number = None
        if call["function"]["name"] == "update_ticket" and number:
            parts.append(f"I haven't changed {number}")
        elif call["function"]["name"] == "create_ticket":
            parts.append("I haven't created the ticket")
        else:
            parts.append("I haven't made that change")
    return f"OK, {' and '.join(dict.fromkeys(parts))}. Nothing was saved. Is there anything else I can help with?"


def prepare(messages: list[dict]) -> list[dict]:
    """The history to send the model: tool calls left unanswered (because the employee moved on
    instead of approving) get a reply saying so, and long conversations keep only recent turns,
    cut at a user message so a tool call is never separated from its result."""
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    fixed = []
    for message in messages:
        fixed.append(message)
        fixed += [{"role": "tool", "tool_call_id": c["id"], "content": MOVED_ON}
                  for c in message.get("tool_calls") or [] if c["id"] not in answered]
    starts = [i for i, m in enumerate(fixed) if m["role"] == "user"]
    fits = [i for i in starts if len(fixed) - i <= MAX_HISTORY]
    start = fits[0] if fits else (starts[-1] if starts else 0)
    return fixed[start:]


def format_sources(sources: list[dict]) -> str:
    excerpts = "\n\n".join(f"[{s['n']}] {s['doc_title']} ({s['doc_code']}), section {s['section']}, "
                           f"page {s['page']}\n{s['content']}" for s in sources)
    return f"{excerpts}\n\nCite these excerpts by number, like [{sources[0]['n']}]." if sources else ""



def cited_numbers(text: str, sources: list[dict]) -> list[int]:
    known = {s["n"] for s in sources}
    return sorted({int(n) for group in CITATION.findall(text) for n in group.split(",") if int(n) in known})


def build_graph(toolbox: ToolBox, llm: AsyncOpenAI, settings: Settings, checkpointer):
    user = settings.demo_user
    tool_schemas = toolbox.openai_schemas()
    extra = {"reasoning_effort": settings.agent_reasoning_effort} if settings.agent_reasoning_effort else None

    async def agent(state: AgentState) -> dict:
        write = get_stream_writer()
        start = time.perf_counter()
        stream = await llm.chat.completions.create(
            model=settings.agent_llm,
            messages=[{"role": "system", "content": system_prompt(settings)}, *prepare(state["messages"])],
            tools=tool_schemas,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra,
        )
        text, calls, usage = [], {}, None
        async for chunk in stream:
            usage = chunk.usage or usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text.append(delta.content)
                write({"type": "token", "text": delta.content})
            for part in delta.tool_calls or []:
                slot = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                slot["id"] = slot["id"] or part.id or ""
                if part.function:
                    slot["name"] += part.function.name or ""
                    slot["arguments"] += part.function.arguments or ""
        message = {"role": "assistant", "content": THINK_BLOCK.sub("", "".join(text)).strip() or None}
        if calls:
            message["tool_calls"] = [
                {"id": c["id"] or f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for _, c in sorted(calls.items())
            ]
        if not calls and needs_grounding(message["content"] or "", state):
            # The reply cites policy excerpts that were never retrieved. Withdraw it, search the
            # policies with the employee's own words, and let the model answer again.
            write({"type": "retract"})
            question = next(m["content"] for m in reversed(state["messages"]) if m["role"] == "user")
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"{GROUNDING_CALL}{uuid.uuid4().hex[:8]}", "type": "function",
                "function": {"name": "search_policies", "arguments": json.dumps({"query": question})}}]}
        write({"type": "model", "ms": round((time.perf_counter() - start) * 1000),
               "prompt_tokens": usage.prompt_tokens if usage else 0,
               "completion_tokens": usage.completion_tokens if usage else 0,
               "tool_calls": [c["function"]["name"] for c in message.get("tool_calls", [])]})
        return {"messages": [message]}

    def needs_grounding(text: str, state: AgentState) -> bool:
        """True when the reply cites excerpts that don't exist, or talks about searching the
        policies instead of doing it (small models sometimes write a tool call out as text).
        At most once per turn, so a model that keeps doing it can't loop forever."""
        if "search_policies" not in toolbox.tools:
            return False
        last_user = max(i for i, m in enumerate(state["messages"]) if m["role"] == "user")
        this_turn = [m for m in state["messages"][last_user:] if m["role"] == "tool"]
        if any(m["tool_call_id"].startswith(GROUNDING_CALL) for m in this_turn):
            return False
        cited = {int(n) for group in CITATION.findall(text) for n in group.split(",")}
        invented = bool(cited - {s["n"] for s in state["sources"]})
        narrated = "search_policies" in text and not this_turn
        return invented or narrated

    def after_agent(state: AgentState) -> str:
        calls = state["messages"][-1].get("tool_calls") or []
        if not calls:
            return END
        return "review" if any(toolbox.needs_approval(c["function"]["name"]) for c in calls) else "tools"

    async def review(state: AgentState) -> dict:
        # Re-runs from the top when the graph resumes, so it must do nothing before interrupt().
        proposals = []
        for call in state["messages"][-1]["tool_calls"]:
            name = call["function"]["name"]
            if not toolbox.needs_approval(name):
                continue
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                continue  # tools() reports the bad arguments to the model instead
            args = {k: v for k, v in args.items() if k not in INJECTED}
            proposals.append({"id": call["id"], "tool": name, "args": args, "summary": describe(name, args)})
        if not proposals:
            return {"decisions": {}}
        reply = interrupt({"actions": proposals})
        decisions = {}
        for item in (reply or {}).get("decisions", []):
            proposal = next((p for p in proposals if p["id"] == item.get("id")), None)
            if proposal is None:
                continue
            edits = {k: v for k, v in (item.get("args") or {}).items()
                     if k in toolbox.visible_params(proposal["tool"])}
            decisions[proposal["id"]] = {"action": "approve" if item.get("action") == "approve" else "reject",
                                         "args": edits}
        return {"decisions": decisions}

    async def tools(state: AgentState) -> dict:
        write = get_stream_writer()
        replies, new_sources = [], []
        next_n = len(state["sources"]) + 1
        for call in state["messages"][-1]["tool_calls"]:
            name, start = call["function"]["name"], time.perf_counter()
            status, args = "error", {}
            try:
                args = {k: v for k, v in json.loads(call["function"]["arguments"] or "{}").items() if k not in INJECTED}
            except json.JSONDecodeError:
                content = "Error: the arguments were not valid JSON."
            else:
                if name not in toolbox.tools:
                    content = f"Error: there is no tool named {name}."
                elif toolbox.needs_approval(name):
                    decision = state["decisions"].get(call["id"])
                    if decision is None or decision["action"] != "approve":
                        status, content = "declined", NOT_APPROVED
                    else:
                        args = {**args, **decision["args"]}
                        ok, result = await toolbox.call(name, args, requester=user, approved_by=user)
                        status = "done" if ok else "error"
                        content = json.dumps(result) if ok else f"Error: {result}"
                else:
                    ok, result = await toolbox.call(name, args, requester=user)
                    status = "done" if ok else "error"
                    if ok and name == "search_policies":
                        found = [{"n": next_n + i, "doc_code": r["policy"], "doc_title": r["title"],
                                  "section": r["section"], "page": r["page"], "content": r["text"]}
                                 for i, r in enumerate(result.get("results", []))]
                        next_n += len(found)
                        new_sources += found
                        content = format_sources(found) or "No matching policy sections."
                    else:
                        content = json.dumps(result) if ok else f"Error: {result}"
            write({"type": "tool", "id": call["id"], "name": name, "args": args, "status": status,
                   "ms": round((time.perf_counter() - start) * 1000),
                   **({"result": content[:400]} if status == "error" else {})})
            replies.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        if new_sources:
            write({"type": "sources", "items": new_sources})
        return {"messages": replies, "sources": new_sources, "decisions": {}}

    def after_tools(state: AgentState) -> str:
        calls = next(m for m in reversed(state["messages"]) if m["role"] == "assistant")["tool_calls"]
        replies = state["messages"][-len(calls):]
        return "declined" if all(r["content"] == NOT_APPROVED for r in replies) else "agent"

    async def declined(state: AgentState) -> dict:
        calls = next(m for m in reversed(state["messages"]) if m["role"] == "assistant")["tool_calls"]
        text = declined_reply(calls)
        get_stream_writer()({"type": "token", "text": text})
        return {"messages": [{"role": "assistant", "content": text}]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("review", review)
    graph.add_node("tools", tools)
    graph.add_node("declined", declined)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["review", "tools", END])
    graph.add_edge("review", "tools")
    graph.add_conditional_edges("tools", after_tools, ["agent", "declined"])
    graph.add_edge("declined", END)
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_agent(settings: Settings, checkpointer, server=None, llm: AsyncOpenAI | None = None):
    """Connects to the MCP tools and the model for one request, and yields the compiled graph."""
    async with ToolBox.connect(server) as toolbox:
        if llm is not None:
            yield build_graph(toolbox, llm, settings, checkpointer)
            return
        async with AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "not-needed") as client:
            yield build_graph(toolbox, client, settings, checkpointer)


async def run_turn(graph, thread_id: str, message: str | None = None,
                   decisions: list[dict] | None = None) -> AsyncIterator[dict]:
    """Runs one turn, from a new message or from the employee's decisions on pending actions,
    and yields events for the chat page: token, model, tool, sources, then approval or done.
    Failures arrive as an error event."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    pending = (await graph.aget_state(config)).interrupts
    if decisions is not None:
        if not pending:
            yield {"type": "error", "detail": "Nothing in this conversation is waiting for approval."}
            return
        inputs = Command(resume={"decisions": decisions})
    else:
        if pending:
            yield {"type": "cancelled", "ids": [a["id"] for a in pending[0].value["actions"]]}
        inputs = {"messages": [{"role": "user", "content": message}], "decisions": {}}

    started, model_calls, tool_calls, prompt_tokens, completion_tokens = time.perf_counter(), 0, 0, 0, 0
    try:
        async for mode, chunk in graph.astream(inputs, config, stream_mode=["custom", "updates"]):
            if mode == "custom":
                if chunk["type"] == "model":
                    model_calls += 1
                    prompt_tokens += chunk["prompt_tokens"]
                    completion_tokens += chunk["completion_tokens"]
                elif chunk["type"] == "tool":
                    tool_calls += 1
                yield chunk
            elif "__interrupt__" in chunk:
                yield {"type": "approval", "actions": chunk["__interrupt__"][0].value["actions"]}
    except GraphRecursionError:
        yield {"type": "error", "detail": "I couldn't finish this within the step limit. Please try rephrasing."}
        return

    state = await graph.aget_state(config)
    if state.interrupts:
        return  # waiting for approval; the approval event has been sent
    answer = state.values["messages"][-1].get("content") or ""
    cited = cited_numbers(answer, state.values["sources"])
    yield {"type": "done", "answer": answer, "cited": cited,
           "sources": [s for s in state.values["sources"] if s["n"] in cited],
           "ms": round((time.perf_counter() - started) * 1000), "model_calls": model_calls,
           "tool_calls": tool_calls, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
