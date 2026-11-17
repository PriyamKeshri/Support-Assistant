"""The agent's graph with a scripted model and fake tools: no Ollama, no database."""
from types import SimpleNamespace

import anyio
from langgraph.checkpoint.memory import InMemorySaver

from app.agent import MAX_HISTORY, MOVED_ON, NOT_APPROVED, ToolBox, build_graph, declined_reply, prepare, run_turn
from app.config import Settings
from helpers import ScriptedLLM

SETTINGS = Settings(llm_base_url="http://fake", llm_model="fake", demo_user="alex@kestrel.example")


def tool(name, read_only, *params):
    schema = {"type": "object", "properties": {p: {"type": "string"} for p in params}, "required": list(params)}
    return SimpleNamespace(name=name, description=f"{name} tool", input_schema=schema,
                           annotations=SimpleNamespace(read_only_hint=read_only))


class FakeToolBox(ToolBox):
    def __init__(self):
        super().__init__(client=None, tools=[
            tool("search_policies", True, "query"),
            tool("get_ticket", True, "requester", "number"),
            tool("create_ticket", False, "requester", "approved_by", "title", "description", "category", "priority"),
        ])
        self.calls = []

    async def call(self, name, args, *, requester, approved_by=""):
        self.calls.append({"name": name, "args": args, "requester": requester, "approved_by": approved_by})
        if name == "search_policies":
            return True, {"results": [{"policy": "IT-OPS-004", "title": "Device and Endpoint Policy",
                                       "section": "5. Lost or Stolen Devices", "page": 1,
                                       "text": "Report a lost or stolen laptop within 2 hours."}]}
        return True, {"number": "INC-1007", **args}


def setup(*replies):
    toolbox, llm = FakeToolBox(), ScriptedLLM(*replies)
    return toolbox, llm, build_graph(toolbox, llm, SETTINGS, InMemorySaver())


def turn(graph, thread="t1", **kwargs) -> list[dict]:
    async def collect():
        return [event async for event in run_turn(graph, thread, **kwargs)]
    return anyio.run(collect)


TICKET = {"title": "Laptop stolen", "description": "Stolen from my car.", "category": "Hardware", "priority": "P1"}


def test_read_only_tools_run_without_approval_and_sources_are_numbered():
    toolbox, llm, graph = setup([("search_policies", {"query": "stolen laptop"})], "Report it within 2 hours [1].")
    events = turn(graph, message="My laptop was stolen, what do I do?")
    kinds = [e["type"] for e in events]
    assert "approval" not in kinds and kinds[-1] == "done"
    assert toolbox.calls == [{"name": "search_policies", "args": {"query": "stolen laptop"},
                              "requester": "alex@kestrel.example", "approved_by": ""}]
    done = events[-1]
    assert done["answer"] == "Report it within 2 hours [1]." and done["cited"] == [1]
    assert done["sources"][0]["doc_code"] == "IT-OPS-004" and done["model_calls"] == 2
    assert "".join(e["text"] for e in events if e["type"] == "token") == done["answer"]


def test_changes_wait_for_approval_then_run_with_the_employee_as_approver():
    toolbox, llm, graph = setup([("create_ticket", TICKET)], "Created INC-1007.")
    events = turn(graph, message="My laptop was stolen, raise a ticket")
    assert events[-1]["type"] == "approval" and toolbox.calls == []
    action = events[-1]["actions"][0]
    assert action["summary"] == "Create a P1 Hardware ticket: Laptop stolen"

    events = turn(graph, decisions=[{"id": action["id"], "action": "approve"}])
    assert toolbox.calls == [{"name": "create_ticket", "args": TICKET, "requester": "alex@kestrel.example",
                              "approved_by": "alex@kestrel.example"}]
    assert events[-1] == {**events[-1], "type": "done", "answer": "Created INC-1007."}


def test_rejected_actions_change_nothing_and_get_a_fixed_reply():
    toolbox, llm, graph = setup([("create_ticket", TICKET)])
    action = turn(graph, message="raise a ticket")[-1]["actions"][0]
    events = turn(graph, decisions=[{"id": action["id"], "action": "reject"}])
    assert toolbox.calls == [] and len(llm.requests) == 1  # the model isn't asked to report the rejection
    assert events[-1]["answer"] == ("OK, I haven't created the ticket. Nothing was saved. "
                                    "Is there anything else I can help with?")


def test_a_partly_rejected_batch_goes_back_to_the_model():
    second = {**TICKET, "title": "Monitor broken", "category": "Hardware", "priority": "P3"}
    toolbox, llm, graph = setup([("create_ticket", TICKET), ("create_ticket", second)], "Created one of them.")
    first_id, second_id = (a["id"] for a in turn(graph, message="two tickets please")[-1]["actions"])
    events = turn(graph, decisions=[{"id": first_id, "action": "approve"}, {"id": second_id, "action": "reject"}])
    assert [c["args"]["title"] for c in toolbox.calls] == ["Laptop stolen"]
    tool_replies = [m["content"] for m in llm.requests[-1]["messages"] if m["role"] == "tool"]
    assert tool_replies[1] == NOT_APPROVED and events[-1]["answer"] == "Created one of them."


def test_approver_edits_apply_only_to_fields_the_model_may_set():
    toolbox, llm, graph = setup([("create_ticket", {**TICKET, "requester": "jordan@kestrel.example"})], "Done.")
    action = turn(graph, message="raise a ticket")[-1]["actions"][0]
    assert "requester" not in action["args"]
    turn(graph, decisions=[{"id": action["id"], "action": "approve",
                            "args": {"priority": "P2", "requester": "mallory@evil.example", "approved_by": "x"}}])
    call = toolbox.calls[0]
    assert call["args"]["priority"] == "P2" and "requester" not in call["args"]
    assert call["requester"] == call["approved_by"] == "alex@kestrel.example"


def test_a_new_message_cancels_the_pending_action():
    toolbox, llm, graph = setup([("create_ticket", TICKET)], "No problem.")
    turn(graph, message="raise a ticket")
    events = turn(graph, message="Never mind, it turned up.")
    assert events[0]["type"] == "cancelled" and events[-1]["type"] == "done"
    assert toolbox.calls == []
    sent = llm.requests[-1]["messages"]
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "tool", "user"]
    assert sent[3]["content"] == MOVED_ON


def test_resuming_with_nothing_pending_is_an_error():
    toolbox, llm, graph = setup("Hello!")
    turn(graph, message="hi")
    events = turn(graph, decisions=[{"id": "call_x", "action": "approve"}])
    assert events == [{"type": "error", "detail": "Nothing in this conversation is waiting for approval."}]


def test_unreadable_arguments_go_back_to_the_model_without_asking_for_approval():
    toolbox, llm, graph = setup([("create_ticket", "{not json")], "Sorry, let me try again.")
    events = turn(graph, message="raise a ticket")
    assert "approval" not in [e["type"] for e in events] and toolbox.calls == []
    tool_event = next(e for e in events if e["type"] == "tool")
    assert tool_event["status"] == "error"


def test_conversation_memory_carries_across_turns():
    toolbox, llm, graph = setup("Hi Alex.", "You asked about your laptop.")
    turn(graph, message="My laptop is broken")
    turn(graph, message="What did I just ask about?")
    roles = [m["role"] for m in llm.requests[-1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_long_histories_are_cut_at_a_user_message():
    history = []
    for i in range(30):
        history += [{"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "function": {}}]},
                    {"role": "tool", "tool_call_id": f"c{i}", "content": "r"}]
    kept = prepare(history)
    assert len(kept) <= MAX_HISTORY and kept[0]["role"] == "user"


def test_identity_and_tuning_parameters_are_not_offered_to_the_model():
    toolbox = ToolBox(client=None, tools=[tool("search_policies", True, "query", "k"),
                                          tool("create_ticket", False, "requester", "approved_by", "title")])
    offered = {s["function"]["name"]: s["function"]["parameters"] for s in toolbox.openai_schemas()}
    assert list(offered["search_policies"]["properties"]) == ["query"]
    assert list(offered["create_ticket"]["properties"]) == ["title"]
    assert offered["create_ticket"]["required"] == ["title"]


def test_rejection_replies_name_what_was_left_alone():
    calls = [{"function": {"name": "update_ticket", "arguments": '{"number": "INC-1001", "status": "Closed"}'}}]
    assert declined_reply(calls) == ("OK, I haven't changed INC-1001. Nothing was saved. "
                                     "Is there anything else I can help with?")


def test_a_reply_citing_unretrieved_policies_is_withdrawn_and_grounded():
    toolbox, llm, graph = setup("No, that's not allowed [1].", "Customer data is Restricted [1].")
    events = turn(graph, message="Can I paste customer data into ChatGPT?")
    kinds = [e["type"] for e in events]
    assert kinds.index("retract") < kinds.index("tool")
    assert toolbox.calls[0]["name"] == "search_policies"
    assert toolbox.calls[0]["args"] == {"query": "Can I paste customer data into ChatGPT?"}
    assert events[-1]["answer"] == "Customer data is Restricted [1]." and events[-1]["cited"] == [1]


def test_the_grounding_search_happens_at_most_once_per_turn():
    toolbox, llm, graph = setup("Not allowed [1].", "Still citing a source that isn't there [9].")
    events = turn(graph, message="Is it allowed?")
    assert [c["name"] for c in toolbox.calls] == ["search_policies"]
    assert [e["type"] for e in events].count("retract") == 1
    assert events[-1]["answer"] == "Still citing a source that isn't there [9]." and events[-1]["cited"] == []


def test_a_search_written_out_as_text_is_actually_run():
    toolbox, llm, graph = setup('<query> {"type": "search_policies", "query": "phishing"} </query>',
                                "Use the Report Phishing button in Outlook [1].")
    events = turn(graph, message="How do I report a phishing email?")
    assert "retract" in [e["type"] for e in events]
    assert toolbox.calls[0]["args"] == {"query": "How do I report a phishing email?"}
    assert events[-1]["answer"] == "Use the Report Phishing button in Outlook [1]."
