"""Write the synthetic TRAIL fixture: two Parquet files laid out like the pinned `data/` directory.

Every record below is written by hand in the real schema (a trace with a
`spans` tree of OTel spans with flattened OpenInference attributes, and an
annotation file with `errors` and `scores`). No text comes from the dataset.
Run: python make_fixture.py  (needs pyarrow); it overwrites the two files.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
G1, G2, G4 = "0000000000000000000000000000aaa1", "0000000000000000000000000000aaa2", "0000000000000000000000000000aaa4"
S1, S2 = "0000000000000000000000000000bbb1", "0000000000000000000000000000bbb2"
BASE = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
RESOURCE = {"service.name": "synthetic/app", "telemetry.sdk.language": "python"}
SCOPE = ("openinference.instrumentation.smolagents", "0.1.6")
PAT = {"pat.app": "Synth", "pat.project.id": "p-1", "pat.project.name": "synthetic"}


def ts(tenths):
    return (BASE + timedelta(milliseconds=100 * tenths)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def span(trace, sid, parent, name, at, dur, attrs=None, kind=None, children=(), status="Ok", message="", events=(),
         logs=(), scope=SCOPE, timestamp=None):
    a = dict(PAT)
    if kind:
        a["openinference.span.kind"] = kind
    a.update(attrs or {})
    return {
        "timestamp": timestamp if timestamp is not None else ts(at), "trace_id": trace, "span_id": sid,
        "parent_span_id": parent, "trace_state": "", "span_name": name, "span_kind": "Internal",
        "service_name": "synthetic/app", "resource_attributes": dict(RESOURCE), "scope_name": scope[0],
        "scope_version": scope[1], "span_attributes": a, "duration": dur, "status_code": status,
        "status_message": message, "events": list(events), "links": [], "logs": list(logs),
        "child_spans": list(children),
    }


def wrapper(trace, sid, parent, name, at, dur, arguments, children=()):
    log = {"timestamp": ts(at).rstrip("Z"), "trace_id": trace, "span_id": sid, "trace_flags": 1, "severity_text": "INFO",
           "severity_number": 9, "service_name": "synthetic/app",
           "body": {"function.arguments": arguments, "function.name": name, "function.output": "<null>"},
           "log_attributes": {"pat.log.type": "trace"}, "evaluations": [], "annotations": []}
    return span(trace, sid, parent, name, at, dur, children=children, logs=[log], scope=("patronus.sdk", ""))


def llm(trace, sid, parent, at, dur, inputs, output=None, calls=(), model="synth-model", tokens=(10, 5), tools=(),
        status="Ok", message="", events=(), timestamp=None):
    """`inputs` is a list of (role or None, content); `output` the output text or None for an absent content."""
    a = {"llm.model_name": model, "input.mime_type": "application/json", "input.value": "{\"messages\": []}"}
    if model is None:
        del a["llm.model_name"]
    for i, (role, content) in enumerate(inputs):
        if role is not None:
            a[f"llm.input_messages.{i}.message.role"] = role
        a[f"llm.input_messages.{i}.message.content"] = content
    if output is not None or calls:
        a["llm.output_messages.0.message.role"] = "assistant"
    if output is not None:
        a["llm.output_messages.0.message.content"] = output
    for k, (cid, fname, args) in enumerate(calls):
        base = f"llm.output_messages.0.message.tool_calls.{k}.tool_call"
        if cid is not None:
            a[f"{base}.id"] = cid
        a[f"{base}.function.name"] = fname
        a[f"{base}.function.arguments"] = args
    for k, definition in enumerate(tools):
        a[f"llm.tools.{k}.tool.json_schema"] = json.dumps(definition)
    if tokens:
        a["llm.token_count.prompt"], a["llm.token_count.completion"] = str(tokens[0]), str(tokens[1])
        a["llm.token_count.total"] = str(tokens[0] + tokens[1])
    return span(trace, sid, parent, "LiteLLMModel.__call__", at, dur, a, "LLM", status=status, message=message,
                events=events, timestamp=timestamp)


def chain(trace, sid, parent, n, at, dur, children, status="Ok", message="", events=()):
    return span(trace, sid, parent, f"Step {n}", at, dur, {"input.value": "ActionStep(synthetic)"}, "CHAIN", children,
                status=status, message=message, events=events)


def tool(trace, sid, parent, name, at, dur, with_output=True, kwargs=None, args=(), output="tool output text"):
    """A TOOL span; `kwargs` and `args` are what the tool was called with (a query by default)."""
    kw = {"query": "synthetic fact"} if kwargs is None else kwargs
    a = {"tool.name": name, "tool.description": f"Synthetic {name}.",
         "tool.parameters": json.dumps({"query": {"type": "string", "description": "query"}}),
         "input.value": json.dumps({"args": list(args), "sanitize_inputs_outputs": False, "kwargs": kw})}
    if with_output:
        a["output.value"] = output
    return span(trace, sid, parent, name.title(), at, dur, a, "TOOL")


def agent(trace, sid, parent, name, at, dur, tools_names, children, task=None):
    a = {"smolagents.tools_names": json.dumps(tools_names), "smolagents.max_steps": "12",
         "llm.token_count.prompt": "100", "llm.token_count.completion": "50", "llm.token_count.total": "150"}
    if task is not None:
        a["input.value"] = json.dumps({"task": task, "stream": False})
    return span(trace, sid, parent, name, at, dur, a, "AGENT", children)


SYS = "You are a synthetic assistant. Solve the task with code."
TASK_USER = "New task:\nSynthetic question one."
FACTS_PROMPT = "Below I will present a synthetic task. Build a survey of facts."
FACTS = "### 1. Facts given\n- the question is synthetic"
PLAN_PROMPT = "Now make a plan for the synthetic task."
PLAN = "1. Ask the helper agent.\n2. Answer.\n<end_plan>"
FACTS_RENDER = "[FACTS LIST]:\nHere are the facts that I know so far:\n```\n" + FACTS + "\n```"
PLAN_RENDER = "[PLAN]:\nHere is the plan of action that I will follow:\n```\n1. Ask the helper agent.\n2. Answer.\n```"
OUT1 = "Thought: delegate to the helper.\nCode:\n```py\nsearch_agent(task='q')\n```"
CALL1 = ("Calling tools:\n[{'id': 'call_1', 'type': 'function', 'function': "
         "{'name': 'python_interpreter', 'arguments': \"search_agent(task='q')\"}}]")
RESP1 = "Call id: call_1\nObservation:\nExecution logs:\nA\nLast output from code snippet:\nA"
OUT2 = "Thought: finish.\nCode:\n```py\nfinal_answer('A')\n```"
PARSE_ERROR = "Error:\nError in code parsing:\nsynthetic parse failure"
SUB_SYS = "You are the helper with tools."
SUB_USER = "New task:\nYou are the helper. Find a synthetic fact."
SUB_FACTS_PROMPT = "Below I will present a synthetic sub task. Build a survey of facts."
SUB_FACTS = "- helper facts"
SUB_FACTS_RENDER = "[FACTS LIST]:\nHere are the facts that I know so far:\n```\n" + SUB_FACTS + "\n```"
SUB_CALL = ("Calling tools:\n[{'id': 'call_sub1', 'type': 'function', 'function': "
            "{'name': 'web_search', 'arguments': {'query': 'synthetic fact'}}}]")
SUB_RESP = "Call id: call_sub1\nObservation:\nThe synthetic fact is A."
WEB_SEARCH = {"type": "function", "function": {"name": "web_search", "description": "Search the synthetic web.",
              "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "query"}},
                             "required": ["query"]}}}
FINAL_ANSWER = {"type": "function", "function": {"name": "final_answer", "description": "Gives the final answer.",
                "parameters": {"type": "object", "properties": {"answer": {"type": "string", "description": "answer"}},
                               "required": ["answer"]}}}
EXCEPTION = {"Timestamp": ts(82).rstrip("Z"), "Name": "exception",
             "Attributes": {"exception.escaped": "False", "exception.message": "synthetic failure",
                            "exception.stacktrace": "Traceback (synthetic)", "exception.type": "BadRequestError"}}


def gaia_one():
    t = G1
    manager_first = [("system", SYS), ("user", TASK_USER), ("assistant", FACTS_RENDER), ("assistant", PLAN_RENDER)]
    sub_first = [("system", SUB_SYS), ("user", SUB_USER), ("assistant", SUB_FACTS_RENDER)]
    sub_agent = agent(t, "a000000000000008", "a000000000000006", "ToolCallingAgent.run", 12, "PT3.5S",
                      ["web_search", "final_answer"], [
        llm(t, "a000000000000009", "a000000000000008", 12, "PT0.5S", [("user", SUB_FACTS_PROMPT)], SUB_FACTS),
        chain(t, "a000000000000010", "a000000000000008", 1, 14, "PT1S", [
            llm(t, "a000000000000011", "a000000000000010", 14, "PT0.8S", sub_first, None,
                [("call_sub1", "web_search", json.dumps({"query": "synthetic fact"}))],
                tools=[WEB_SEARCH, FINAL_ANSWER]),
            tool(t, "a000000000000012", "a000000000000010", "web_search", 22, "PT0.1S")]),
        chain(t, "a000000000000013", "a000000000000008", 2, 25, "PT1S", [
            llm(t, "a000000000000014", "a000000000000013", 25, "PT0.9S", sub_first + [
                ("tool-call", SUB_CALL), ("tool-response", SUB_RESP)], None,
                [("call_sub2", "final_answer", json.dumps({"answer": "A"}))], tools=[WEB_SEARCH, FINAL_ANSWER])])])
    manager = agent(t, "a000000000000003", "a000000000000002", "CodeAgent.run", 1, "PT9S", ["final_answer", "helper_tool"], [
        llm(t, "a000000000000004", "a000000000000003", 1, "PT0.5S", [("user", FACTS_PROMPT)], FACTS),
        llm(t, "a000000000000005", "a000000000000003", 7, "PT0.5S", [("user", PLAN_PROMPT)], PLAN),
        chain(t, "a000000000000006", "a000000000000003", 1, 10, "PT5S", [
            llm(t, "a000000000000007", "a000000000000006", 10, "PT0.5S", manager_first, OUT1),
            sub_agent]),
        chain(t, "a000000000000015", "a000000000000003", 2, 60, "PT2S", [
            llm(t, "a000000000000016", "a000000000000015", 60, "PT0.7S", manager_first + [
                ("assistant", OUT1 + "<end_code>"), ("tool-call", CALL1), ("tool-response", RESP1),
                ("tool-response", PARSE_ERROR)], OUT2),
            tool(t, "a000000000000017", "a000000000000015", "final_answer", 67, "PT0.01S", with_output=False)]),
        chain(t, "a000000000000019", "a000000000000003", 3, 80, "PT1S", [
            llm(t, "a000000000000018", "a000000000000019", 80, "PT0.3S", manager_first, None, status="Error",
                message="BadRequestError: synthetic failure", events=[EXCEPTION])],
            status="Error", message="AgentGenerationError: synthetic failure", events=[EXCEPTION])],
        task="Synthetic task text for the manager.")
    final = llm(t, "a000000000000020", "a000000000000002", 92, "PT0.4S", [
        ("system", "Earlier you were asked the following synthetic question."), ("user", TASK_USER),
        ("user", FACTS_RENDER), ("user", CALL1), ("user", RESP1),
        ("user", "Read the conversation above and give a FINAL ANSWER.")], "FINAL ANSWER: A")
    root = wrapper(t, "a000000000000001", None, "main", 0, "PT20S", {}, [
        wrapper(t, "a000000000000099", "a000000000000001", "get_examples_to_answer", 0, "PT0.1S", {"eval_ds": "Dataset(...)"}),
        wrapper(t, "a000000000000002", "a000000000000001", "answer_single_question", 1, "PT15S",
                {"example": {"task_id": "synthetic-task-1", "question": "Synthetic question one.", "true_answer": "A",
                             "Annotator Metadata": {}}, "model_id": "synth-model"}, [manager, final])])
    labels = {"trace_id": t, "errors": [
        {"category": "Formatting Errors", "location": "a000000000000007", "evidence": "search_agent(task='q')",
         "description": "The code block has no final answer call.", "impact": "LOW"},
        {"category": "Tool-related", "location": "a000000000000012", "evidence": "tool output text",
         "description": "The tool result was misread.", "impact": "HIGH"},
        {"category": "Goal Deviation", "location": "a000000000000011", "evidence": "web_search",
         "description": "Searched an unrelated query.", "impact": "MEDIUM"},
        {"category": "Language-only", "location": "a000000000000011", "evidence": "synthetic fact",
         "description": "Unclear wording.", "impact": "LOW"},
        {"category": "Resource Abuse", "location": "Span ID not found for this shard", "evidence": "n/a",
         "description": "The location names no span.", "impact": "MEDIUM"}],
        "scores": [{"reliability_score": 4, "reliability_reasoning": "Mostly reliable.", "security_score": 5,
                    "security_reasoning": "No issue.", "instruction_adherence_score": 3,
                    "instruction_adherence_reasoning": "One slip.", "plan_opt_score": 2,
                    "plan_opt_reasoning": "Plan was loose.", "overall": 3.5}]}
    return {"trace_id": t, "spans": [root]}, json.dumps(labels, indent=2)


def gaia_two():
    """No harness wrapper: one AGENT root with odd messages, a repeated span, an orphan root, bad timestamps."""
    t = G2
    odd = [("system", "S"), ("user", "U"), ("assistant", "A1 matches no output"), ("function", "F body"),
           (None, "R role-less"), ("tool-call", "Calling tools: not a list"),
           ("tool-response", "Call id: call_x\nObservation:\nlate result")]
    nested = llm(t, "b000000000000004", "b000000000000003", 4, "PT0.1S", [("user", "dup")], "dup out")
    root = agent(t, "b000000000000001", None, "CodeAgent.run", 0, "PT1M2.25S", ["tool_a"], [
        llm(t, "b000000000000002", "b000000000000001", 1, "PT0.2S", odd, "Thought: odd.",
            [(None, "mystery", "not json")], model=None, tokens=None),
        chain(t, "b000000000000003", "b000000000000001", 1, 3, "PT1H0M0.5S", [nested, nested]),
        llm(t, "b000000000000005", "b000000000000001", 9, "PT0.2S", [(None, "A1 matches no output")] + odd[3:] + [
            ("assistant", "unrelated role-ful text"),
            ("tool-call", "Calling tools:\n[{'id': 'call_9', 'type': 'function', 'function': "
                          "{'name': 'python_interpreter', 'arguments': 'print(1)'}}]"),
            ("tool-response", None)], "Thought: tail.", tokens=(1, 1), timestamp="unknown")])
    root["span_attributes"]["smolagents.task"] = "Fallback task text."
    orphan = span(t, "b000000000000006", "ffffffffffffffff", "create_agent", 0, "PT0.01S", scope=("patronus.sdk", ""))
    labels_text = ("{\n \"errors\": [\n  {\"category\": \"Instruction Non-compliance\", \"location\": "
                   "\"b000000000000002\", \"evidence\": \"x, y\", \"description\": \"d\", \"impact\": \"LOW\"},\n"
                   "  {\"category\": \"Resource Abuse\", \"location\": \"b000000000000003\", \"evidence\": \"e\",\n"
                   "   \"Observation: ...\": \"stray key\", \"within the same output.\": \"stray text\", "
                   "\"impact\": \"HIGH\"},\n ],\n \"scores\": [{\"overall\": 1}],\n}")
    return {"trace_id": t, "spans": [root, orphan]}, labels_text


def eid(step, part):
    """Span id of the Step CHAIN (part 0), its LLM span (1) or one of its TOOL spans (2 and up) in `gaia_four`."""
    return f"e{0:011d}{step:02d}{part:02d}"


def gaia_four():
    """One ToolCallingAgent whose steps show when a TOOL span is tied to a call and when it is not.

    Step 1 and 2: one call and one TOOL span with the same name and arguments, under two Step CHAIN spans (both tie).
    Step 3: two identical calls and two identical TOOL spans (none tie). Step 4: one call and two identical TOOL
    spans (none tie). Step 5: TOOL spans whose argument value or tool name differs from the call's, or that was also
    given a positional argument (none tie).
    Step 6: the arguments are equal with the keys in another order (ties). Step 7: the call's arguments are not
    JSON, so ATIF holds a stand-in {} that never matches a TOOL span called with {} (none). Step 8: the call
    carries the object {} (ties). Step 9 and 10: a history-message call with the name and arguments of a TOOL span
    (never tied), and a TOOL span whose id appears twice with different content (never tied). Step 11: two identical
    calls and one TOOL span (none tie).
    """
    t = G4
    agent_id = "e000000000000003"
    first = [("system", SUB_SYS), ("user", SUB_USER)]
    args = lambda **kw: json.dumps(kw)  # noqa: E731
    out_h = "Thought: search.\nCode:\n```py\nweb_search(query='hist')\n```"
    call_h = ("Calling tools:\n[{'id': 'call_hist', 'type': 'function', 'function': "
              "{'name': 'web_search', 'arguments': {'query': 'hist'}}}]")
    resp_h = "Call id: call_hist\nObservation:\nhistory result"
    history = [("assistant", out_h + "<end_code>"), ("tool-call", call_h), ("tool-response", resp_h)]

    def step(k, calls, tools, inputs=first, output=None):
        at = 10 * k
        return chain(t, eid(k, 0), agent_id, k, at, "PT0.9S",
                     [llm(t, eid(k, 1), eid(k, 0), at, "PT0.4S", inputs, output, calls)] + tools)

    def call_of(k, j, name, kwargs, **kw):
        return tool(t, eid(k, j), eid(k, 0), name, 10 * k + j, "PT0.1S", kwargs=kwargs, **kw)

    twin = [("call_twin1", "visit_page", args(url="u-twin")), ("call_twin2", "visit_page", args(url="u-twin"))]
    steps = [
        step(1, [("call_tie", "web_search", args(query="alpha"))], [call_of(1, 2, "web_search", {"query": "alpha"})]),
        step(2, [("call_tie2", "web_search", args(query="alpha"))], [call_of(2, 2, "web_search", {"query": "alpha"})]),
        step(3, twin, [call_of(3, 2, "visit_page", {"url": "u-twin"}), call_of(3, 3, "visit_page", {"url": "u-twin"})]),
        step(4, [("call_one", "find_on_page_ctrl_f", args(search_string="s"))],
             [call_of(4, 2, "find_on_page_ctrl_f", {"search_string": "s"}),
              call_of(4, 3, "find_on_page_ctrl_f", {"search_string": "s"})]),
        step(5, [("call_diff", "visit_page", args(url="u-call"))],
             [call_of(5, 2, "visit_page", {"url": "u-other"}), call_of(5, 3, "page_down", {"url": "u-call"}),
              call_of(5, 4, "visit_page", {"url": "u-call"}, args=["u-call"])]),
        step(6, [("call_order", "find_archived_url", args(url="u-arch", date="20230831"))],
             [call_of(6, 2, "find_archived_url", {"date": "20230831", "url": "u-arch"})]),
        step(7, [("call_raw", "page_down", "not json")], [call_of(7, 2, "page_down", {})]),
        step(8, [("call_empty", "page_up", "{}")], [call_of(8, 2, "page_up", {})]),
        step(9, [], [call_of(9, 2, "web_search", {"query": "hist"})], output=out_h),
        step(10, [("call_dup", "inspect_file_as_text", args(file_path="f", question="q"))],
             [call_of(10, 2, "inspect_file_as_text", {"file_path": "f", "question": "q"}),
              call_of(10, 2, "inspect_file_as_text", {"file_path": "f", "question": "q"}, output="another output")],
             inputs=first + history),
        step(11, [("call_pair1", "page_down", args(q="x")), ("call_pair2", "page_down", args(q="x"))],
             [call_of(11, 2, "page_down", {"q": "x"})]),
    ]
    agent_span = agent(t, agent_id, "e000000000000002", "ToolCallingAgent.run", 1, "PT20S",
                       ["web_search", "visit_page", "final_answer"], steps, task="Synthetic tool task.")
    root = wrapper(t, "e000000000000001", None, "main", 0, "PT30S", {}, [
        wrapper(t, "e000000000000002", "e000000000000001", "answer_single_question", 1, "PT25S",
                {"example": {"task_id": "synthetic-task-4", "question": "Synthetic tool question.", "true_answer": "B"}},
                [agent_span])])
    return {"trace_id": t, "spans": [root]}, json.dumps({"trace_id": t, "errors": [], "scores": []})


def swe_one():
    t = S1
    o1, o2, o3 = "Plan: look at the repo.\nCode:\n```py\nprint('one')\n```", "Next.\nCode:\n```py\nprint('two')\n```", "Done."
    c = lambda n: (f"Calling tools:\n[{{'id': 'call_{n}', 'type': 'function', 'function': "
                   f"{{'name': 'python_interpreter', 'arguments': \"print('{n}')\"}}}}]")
    r = lambda n: f"Call id: call_{n}\nObservation:\nExecution logs:\n{n}\nLast output from code snippet:\nNone"
    first = [("system", "SWE system prompt."), ("user", "Fix the synthetic issue.")]
    second = first + [("assistant", o1 + "<end_code>"), ("tool-call", c(1)), ("tool-response", r(1))]
    window = [(None, o1 + "<end_code>"), ("tool-call", c(1)), ("tool-response", r(1)),
              ("assistant", o2 + "<end_code>"), ("tool-call", c(2)), ("tool-response", r(2))]
    llms = [llm(t, "c000000000000003", "c000000000000002", 1, "PT0.5S", first, o1, model="synth-claude", tokens=(7, 3)),
            llm(t, "c000000000000004", "c000000000000002", 3, "PT0.5S", second, o2, model="synth-claude", tokens=(8, 4)),
            llm(t, "c000000000000005", "c000000000000002", 5, "PT0.5S", window, o3, model="synth-claude", tokens=None)]
    llms[2]["span_attributes"]["llm.token_count.prompt"] = "many"
    llms[2]["span_attributes"]["llm.token_count.completion"] = "3"
    final = tool(t, "c000000000000006", "c000000000000002", "final_answer", 7, "PT0.01S", with_output=False)
    root = wrapper(t, "c000000000000001", None, "process_item", 0, "PT9S",
                   {"item": {"instance_id": "synth__repo-1", "repo": "synth/repo", "problem_statement": "Synthetic issue."},
                    "item_index": 1}, [
        agent(t, "c000000000000002", "c000000000000001", "CodeAgent.run", 1, "PT8S", ["final_answer"], llms + [final],
              task="Fix the synthetic issue.")])
    labels = {"trace_id": t, "errors": [
        {"category": "Context Handling Failures", "location": "c000000000000004", "evidence": "e", "description": "d",
         "impact": "MEDIUM"}], "scores": [{"reliability_score": 1, "reliability_reasoning": "r", "overall": 1}]}
    return {"trace_id": t, "spans": [root]}, json.dumps(labels)


def swe_two():
    """The AGENT span is absent from the file: root spans point at a parent that is not there."""
    t = S2
    steps = [chain(t, f"d00000000000000{n}", "eeeeeeeeeeeeeeee", n, n, "PT0.5S", [
        llm(t, f"d00000000000010{n}", f"d00000000000000{n}", n, "PT0.4S",
            [("system", "SWE system prompt."), ("user", "Fix the other synthetic issue.")] * 1, f"Step {n} text.")])
        for n in (1, 2)]
    return {"trace_id": t, "spans": steps}, json.dumps({"errors": [], "scores": []})


def main():
    g1, g2, g4 = gaia_one(), gaia_two(), gaia_four()
    gaia = {"trace": [json.dumps(g1[0], indent=4), json.dumps(g2[0]), "this row is not JSON", json.dumps(g4[0])],
            "labels": [g1[1], g2[1], None, g4[1]]}
    s1, s2 = swe_one(), swe_two()
    swe = {"trace": [json.dumps(s1[0]), json.dumps(s2[0])], "labels": [s1[1], s2[1]]}
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    schema = pa.schema([("trace", pa.string()), ("labels", pa.string())])
    for name, cols in (("gaia-00000-of-00001-33a2e72d362d688a.parquet", gaia),
                       ("swe_bench-00000-of-00001-91aa04220f7198b4.parquet", swe)):
        pq.write_table(pa.table(cols, schema=schema), data / name)
        print(data / name, len(cols["trace"]), "rows")


if __name__ == "__main__":
    main()
