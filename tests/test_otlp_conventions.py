"""OTLP output against the conventions it follows, over every fixture.

Generated spans use the attribute names of OpenInference 0.1.41 and the OTel GenAI names and value schemas at revision
9ffc7e00c639be7712e544e3e54331e885dccf84, whose schemas are under ``schemas/genai``. The output is a partial projection:
what the record does not hold stays absent. Spans adopted from a source trace keep what the source gave them, which the
last tests check apart from conformance. "Every fixture" means the format fixtures under ``fixtures`` and the dataset
fixtures under ``corpora/fixtures``; the five Parquet datasets among them load only with pyarrow (the ``hf`` extra),
and the checks that count across all datasets skip without it. The schema tests skip without jsonschema (the ``dev``
extra).
"""

import copy
import importlib.util
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats import otlp

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
CORPORA = HERE / "corpora" / "fixtures"
SCHEMAS = HERE / "schemas" / "genai"
SOURCED = ["otlp/pydantic_ai_2.58.0.json", "otlp/pydantic_ai_2.58.0_delegation.json",
           "otlp/openai_agents_0.24.0_oi_2.5.3.jsonl", "otlp/openai_agents_0.24.0_oi_2.5.3_as_tool.json"]
ARROW = importlib.util.find_spec("pyarrow") is not None
needs_arrow = pytest.mark.skipif(not ARROW, reason="counts over all 13 datasets; the Parquet ones need pyarrow")


@lru_cache(maxsize=None)
def exports():
    """(fixture name, run, resource attributes, spans in document order, attributes by span id) for every fixture
    trajectory. Shared across tests: copy a record before changing it."""
    runs = []
    for p in sorted(FIX.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".jsonl") and p.name not in ("versions.json", "not_a_trace.json") \
                and "grade" not in p.relative_to(FIX).parts:
            runs += [(p.relative_to(FIX).as_posix(), run) for run in ap.load(p)]
    for d in sorted(x for x in CORPORA.iterdir() if x.is_dir()):
        if ARROW or not any(d.rglob("*.parquet")):
            runs += [(d.name, run) for run in ap.load(d.name, root=str(d))]
    out = []
    for name, run in runs:
        if run.kind != "trajectory":
            continue
        doc = otlp.to_document(run)
        spans = spans_of(doc)
        out.append((name, run, otlp.decode_attributes(doc["resourceSpans"][0]["resource"]["attributes"]), spans,
                    {s["spanId"]: otlp.decode_attributes(s["attributes"]) for s in spans}))
    return tuple(out)


def generated():
    """(fixture name, attributes) for every span the writer generated rather than adopted from a source trace."""
    return [(name, a) for name, _, _, _, attrs in exports() for a in attrs.values() if not adopted(a)]


def spans_of(doc):
    return [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]


def attrs_of(span):
    return otlp.decode_attributes(span["attributes"])


def adopted(attrs):
    return "agentpile.span_key" in attrs


def strict_json(text):
    """JSON text as a value, refusing NaN and infinity, which the JSON standard does not have, also where a number
    too large for a float decodes to infinity."""
    def refuse(name):
        raise ValueError(name)
    value = json.loads(text, parse_constant=refuse)
    json.dumps(value, allow_nan=False)
    return value


def trajectories(record):
    yield record
    subs = record.get("subagent_trajectories")
    for sub in subs if isinstance(subs, list) else []:
        if isinstance(sub, dict):
            yield from trajectories(sub)


# -- resource ---------------------------------------------------------------------------------------------------

def test_the_resource_names_the_agent_and_the_dataset():
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)

    def resource():
        doc = otlp.to_document(ap.Run(record))
        return otlp.decode_attributes(doc["resourceSpans"][0]["resource"]["attributes"])

    source = record["extra"]["agentpile"]["source"]
    assert resource()["service.name"] == "synthetic-travel" and "openinference.project.name" not in resource()
    source["dataset"] = "someone/toy"  # its reader ("fixture") is no registered dataset: the source's own id
    assert resource()["service.name"] == "synthetic-travel" and resource()["openinference.project.name"] == "someone/toy"
    source["reader"] = "tau_bench"  # a dataset agentpile loads: the name ap.load takes
    assert resource()["openinference.project.name"] == "tau_bench"
    for name in ("unknown", ""):  # an agent with no usable name: the service is the dataset
        record["agent"]["name"] = name
        assert resource()["service.name"] == "tau_bench"
    del record["agent"]["name"]
    assert resource()["service.name"] == "tau_bench"
    source["dataset"] = None  # and with no dataset either, agentpile
    assert resource()["service.name"] == "agentpile" and "openinference.project.name" not in resource()
    # every dataset fixture becomes a project named as ap.load names it; tau_bench and adp name no agent
    corpora = [(name, res) for name, _, res, _, _ in exports() if (CORPORA / name).is_dir()]
    assert len({name for name, _ in corpora}) == (12 if ARROW else 7)  # catchbench_pre holds configs only
    assert all(res["openinference.project.name"] == name for name, res in corpora)
    assert {res["service.name"] for name, res in corpora if name == "tau_bench"} == {"tau_bench"}


# -- nesting and time --------------------------------------------------------------------------------------------

def nested_record():
    """atif_multiagent with a five-step helper, called from the booking call and holding a one-step subagent of its
    own, none of them timed: their synthetic times must fit inside a call span a few microseconds wide."""
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    inner = {"schema_version": "ATIF-v1.7", "trajectory_id": "inner", "agent": {"name": "inner", "version": "1"},
             "steps": [{"step_id": 1, "source": "agent", "message": "deep"}]}
    helper = {"schema_version": "ATIF-v1.7", "trajectory_id": "helper", "agent": {"name": "helper", "version": "1"},
              "steps": [{"step_id": k + 1, "source": "user" if k == 0 else "agent", "message": "m%d" % k} for k in range(5)],
              "subagent_trajectories": [inner]}
    record["steps"][3]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "helper"}]
    record["subagent_trajectories"] = [helper]
    return record


def misplaced(spans):
    """(span name, problem) for each generated span that leaves its parent's interval, or takes no time although its
    parent does (under a parent that takes no time, it takes none either)."""
    by_id = {s["spanId"]: s for s in spans}
    out = []
    for s in spans:
        if adopted(attrs_of(s)):
            continue
        start, end = int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"])
        parent = by_id.get(s.get("parentSpanId"))
        # a source parent recorded as ending before it starts holds what is under it at its start
        lo, hi = (int(parent["startTimeUnixNano"]), int(parent["endTimeUnixNano"])) if parent is not None else (None, None)
        hi = max(lo, hi) if parent is not None else None
        if end < start or (end == start and (parent is None or hi > lo)):
            out.append((s["name"], "no width"))
        if parent is not None and not lo <= start <= end <= hi:
            out.append((s["name"], "outside its parent"))
    return out


def test_every_generated_span_has_a_width_inside_its_parent():
    docs = [(name, spans) for name, _, _, spans, _ in exports()] + [("nested", spans_of(otlp.to_document(ap.Run(nested_record()))))]
    for name, spans in docs:
        assert misplaced(spans) == [], name
    assert sum(not adopted(attrs_of(s)) for _, spans in docs for s in spans) > (500 if ARROW else 300)
    spans = docs[-1][1]
    helper = next(s for s in spans if s["name"] == "invoke_agent helper")
    assert next(s for s in spans if s["spanId"] == helper["parentSpanId"])["name"] == "execute_tool book_flight"


def with_source_span(record, key, width, address=None):
    """The record with one kept source span of ``width`` ns at the start of synthetic time, which the writer adopts as
    the span of the step at ``address`` or, keyed by a subagent's trajectory id, as that subagent's span."""
    entry = {"name": "source " + key, "kind": "INTERNAL", "start_time_unix_nano": otlp.SYNTHETIC_EPOCH_NS,
             "end_time_unix_nano": otlp.SYNTHETIC_EPOCH_NS + width, "attributes": {}}
    if address is not None:
        entry["address"] = address
    record["extra"]["agentpile"]["spans"] = {key: entry}
    return record


@pytest.mark.parametrize("width", [0, 1, 2, 3, 1000])
def test_calls_fit_inside_a_source_step_span_of_any_width(width):
    # the step calling search_flights and read_calendar has a source span; neither call has one of its own
    record = with_source_span(copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record), "s", width, {"step": 3})
    run = ap.Run(record)
    assert ap.validate(run) == []
    spans = spans_of(otlp.to_document(run))
    calls = [s for s in spans if s["name"].startswith("tool_call ")]
    assert [s["name"] for s in calls] == ["tool_call search_flights", "tool_call read_calendar", "tool_call book_flight"]
    # fewer nanoseconds than calls: the calls overlap, each still taking time; a source span that takes no time keeps
    # its times, so the calls under it take none either
    assert misplaced(spans) == []
    assert [int(s["endTimeUnixNano"]) > int(s["startTimeUnixNano"]) for s in calls[:2]] == [bool(width)] * 2
    assert next(otlp.read(otlp.to_document(run))) == run


@pytest.mark.parametrize("gap", ["no start", "null start", "no end", "null end", "end before start"])
@pytest.mark.parametrize("where", ["step", "subagent"])
def test_a_source_span_without_both_times_still_holds_what_is_written_under_it(gap, where):
    if where == "step":
        record = with_source_span(copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record), "s", 100, {"step": 3})
    else:
        record = with_source_span(nested_record(), "helper", 100)
    entry = next(iter(record["extra"]["agentpile"]["spans"].values()))
    field = "start_time_unix_nano" if "start" in gap.split()[-1] else "end_time_unix_nano"
    if gap.startswith("no "):
        del entry[field]
    elif gap.startswith("null "):
        entry[field] = None
    else:
        entry["start_time_unix_nano"], entry["end_time_unix_nano"] = entry["end_time_unix_nano"], entry["start_time_unix_nano"]
    run = ap.Run(record)
    assert ap.validate(run) == []
    spans = spans_of(otlp.to_document(run))
    (source,) = [s for s in spans if s["name"].startswith("source ")]
    assert misplaced(spans) == []
    if gap != "end before start":  # a time the source lacks is filled in without passing the one it has
        assert int(source["startTimeUnixNano"]) <= int(source["endTimeUnixNano"])
    assert next(otlp.read(otlp.to_document(run))) == run


@pytest.mark.parametrize("times", [{}, {0: -10 ** 9}, {0: 10 ** 9}, {0: 50, 4: 50}, {1: 30, 2: 30}])
def test_steps_fit_inside_a_source_subagent_span_whatever_their_timestamps(times):
    # the helper subagent has a 100 ns source span; its steps carry no timestamp, or ones before, after or inside it,
    # equal ones among them, and their spans are drawn inside it while the record keeps every timestamp
    record = with_source_span(nested_record(), "helper", 100)
    for i, offset in times.items():
        record["subagent_trajectories"][0]["steps"][i]["timestamp"] = otlp.iso_from_ns(otlp.SYNTHETIC_EPOCH_NS + offset)
    run = ap.Run(record)
    assert ap.validate(run) == []
    spans = spans_of(otlp.to_document(run))
    assert misplaced(spans) == []
    helper = [s for s in spans if s["name"] == "source helper"]
    assert len(helper) == 1 and (int(helper[0]["startTimeUnixNano"]), int(helper[0]["endTimeUnixNano"])) == (
        otlp.SYNTHETIC_EPOCH_NS, otlp.SYNTHETIC_EPOCH_NS + 100)
    for i, offset in times.items():
        (step,) = [a for a in map(attrs_of, spans) if a.get("agentpile.kind") == "step"
                   and json.loads(a.get("agentpile.step.message", "null")) == "m%d" % i]
        if not 0 <= offset < 100:
            assert step["agentpile.time_synthetic"] is True  # moved off its source time, which the record keeps
    assert next(otlp.read(otlp.to_document(run))) == run


def test_a_generated_call_span_comes_after_the_step_span_it_sits_under():
    checked = 0
    for name, _, _, spans, attrs in exports():
        position = {s["spanId"]: k for k, s in enumerate(spans)}
        for s in spans:
            a = attrs[s["spanId"]]
            if a.get("agentpile.kind") != "call" or adopted(a):
                continue
            step = attrs[s["parentSpanId"]]  # adopted or generated
            assert step["agentpile.kind"] == "step" and step["agentpile.index"] == a["agentpile.step_index"], (name, s["name"])
            assert position[s["parentSpanId"]] < position[s["spanId"]]
            checked += 1
    assert checked > (90 if ARROW else 50)


def test_the_reader_rebuilds_every_fixture_run():
    for name, run, _, _, _ in exports():
        assert next(otlp.read(otlp.to_document(run))) == run, name
    nested = ap.Run(nested_record())
    assert ap.validate(nested) == [] and next(otlp.read(otlp.to_document(nested))) == nested


def test_a_call_under_a_source_step_span_reads_back_into_the_trajectory_that_owns_the_step():
    # A call with no source span of its own is written under the source span of the step that issued it. That span's
    # source parents need not lead to the step's trajectory, so it names the trajectory span in agentpile.owner, which
    # the reader follows at each hop up from the call.
    cases = Counter()
    for name, run, _, spans, attrs in exports():
        for s in spans:
            a = attrs[s["spanId"]]
            if a.get("agentpile.kind") == "call" and not adopted(a) and adopted(attrs[s["parentSpanId"]]):
                assert attrs[s["parentSpanId"]]["agentpile.owner"] in attrs, (name, s["name"])
                cases[name] += 1
    assert set(cases) == {"otlp/openai_agents_0.24.0_oi_2.5.3.jsonl", "otlp/openai_agents_0.24.0_oi_2.5.3_as_tool.json"} | (
        {"trail"} if ARROW else set())
    # In the as-tool fixture the Boss's model step holds the generated ask_helper call. Give that step a source parent
    # inside the Helper subagent: the call still reads back into the Boss's trajectory, because the step names it.
    run = next(ap.load(FIX / "otlp" / "openai_agents_0.24.0_oi_2.5.3_as_tool.json"))
    doc = otlp.to_document(run)
    spans = {s["spanId"]: s for s in spans_of(doc)}
    (helper,) = [sid for sid, s in spans.items() if attrs_of(s).get("agentpile.kind") == "subagent"]
    (call,) = [s for s in spans.values() if attrs_of(s).get("agentpile.kind") == "call" and not adopted(attrs_of(s))]
    step = spans[call["parentSpanId"]]
    assert adopted(attrs_of(step)) and attrs_of(step)["agentpile.owner"] != helper
    step["parentSpanId"] = helper
    assert next(otlp.read(doc)) == run


# -- model spans -----------------------------------------------------------------------------------------------

def test_a_model_span_shows_what_came_in_since_the_previous_model_step():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    chats = {a["agentpile.index"]: a for a in map(attrs_of, spans_of(otlp.to_document(run)))
             if a.get("gen_ai.operation.name") == "chat"}
    steps = run.steps
    assert json.loads(chats[2]["input.value"]) == [{"role": "system", "content": steps[0]["message"]},
                                                  {"role": "user", "content": steps[1]["message"]}]
    # the results of the previous model step's calls, in record order, with the call each answers
    assert json.loads(chats[3]["input.value"]) == [{"role": "tool", "content": "No events.", "tool_call_id": "c2"},
                                                  {"role": "tool", "content": steps[2]["observation"]["results"][1]["content"],
                                                   "tool_call_id": "c1"}]
    assert json.loads(chats[4]["input.value"]) == [{"role": "tool", "content": "error: malformed arguments", "tool_call_id": "c3"}]
    for a in chats.values():
        assert a["agentpile.input_reconstructed"] is True and a["input.mime_type"] == "application/json"


def test_a_step_that_made_no_model_call_passes_what_it_heard_to_the_next_model_call():
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    record["steps"] = [
        {"step_id": 1, "source": "user", "message": "Find a flight"},
        {"step_id": 2, "source": "agent", "message": "Hello! Let me look.", "llm_call_count": 0,
         "observation": {"results": [{"content": "an unpaired result"}]}, "extra": {"agentpile": {"scripted": True}}},
        {"step_id": 3, "source": "agent", "message": "Found one."}]
    run = ap.Run(record)
    assert ap.validate(run) == []
    spans = list(map(attrs_of, spans_of(otlp.to_document(run))))
    (chat,) = [a for a in spans if a.get("gen_ai.operation.name") == "chat"]
    assert chat["agentpile.index"] == 2  # the scripted step is no model call, so it shows no input of its own
    assert json.loads(chat["input.value"]) == [{"role": "user", "content": "Find a flight"},
                                              {"role": "assistant", "content": "Hello! Let me look."},
                                              {"role": "tool", "content": "an unpaired result"}]


def test_no_generated_span_presents_a_reconstruction_as_the_prompt():
    for name, a in generated():
        assert not any(k.startswith("llm.input_messages") or k in ("gen_ai.input.messages", "gen_ai.system_instructions")
                       for k in a), name
        if a.get("gen_ai.operation.name") == "chat" and "input.value" in a:
            assert a["agentpile.input_reconstructed"] is True
    for name, run, _, spans, attrs in exports():
        # each step's text and each result is shown once at most, so the inputs grow linearly with the run
        shown = sum(len(json.loads(a["input.value"])) for a in attrs.values() if a.get("agentpile.input_reconstructed"))
        held = sum(len(t["steps"]) for t in trajectories(run.record)) + sum(
            len(s["observation"]["results"]) for t in trajectories(run.record) for s in t["steps"]
            if isinstance(s.get("observation"), dict) and isinstance(s["observation"].get("results"), list))
        assert shown <= held, name


# -- tool spans -------------------------------------------------------------------------------------------------

@needs_arrow
def test_tool_spans_leave_out_the_values_the_genai_schemas_do_not_take():
    # Counts from the plan: 6 of 94 argument values and 72 of 78 results have no GenAI attribute, and the nine
    # AgentRewardBench calls that parsed give their parsed object.
    arguments, results = Counter(), Counter()
    for name, a in generated():
        if not (CORPORA / name).is_dir() or a.get("gen_ai.operation.name") != "execute_tool":
            continue
        if "input.value" in a:
            arguments[(name, "gen_ai.tool.call.arguments" in a)] += 1
            if "gen_ai.tool.call.arguments" not in a:
                assert otlp._json_object(a["input.value"]) is None  # left out only for a value that is no object
        if "output.value" in a:
            results[(name, "gen_ai.tool.call.result" in a)] += 1
            if "gen_ai.tool.call.result" not in a:
                assert otlp._json_object(a["output.value"]) is None
    assert sum(arguments.values()) == 94 and sum(n for (_, kept), n in arguments.items() if not kept) == 6
    assert sum(results.values()) == 78 and sum(n for (_, kept), n in results.items() if not kept) == 72
    assert arguments[("agent_reward_bench", True)] == 9 and arguments[("agent_reward_bench", False)] == 3
    # an object holding NaN or infinity is no JSON object either, as text, as a value, or as a number too large for a float
    assert otlp._json_object('{"a": 1}') == {"a": 1} and otlp._json_object('{"a": NaN}') is None
    assert otlp._json_object({"a": float("inf")}) is None and otlp._json_object("[1]") is None
    for text in ('{"a": 1e400}', '{"a": -1e400}', '{"a": {"b": [1, 1e400]}}'):
        assert otlp._json_object(text) is None


@pytest.mark.parametrize("case", ["same id", "no schema", "no id", "own id"])
def test_a_subagent_call_takes_its_tool_schema_from_the_subagent_s_own_tools(case):
    # parent and subagent each define search_flights, with different parameter schemas
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    schema = {"type": "object", "properties": {"child_query": {"type": "string"}}}
    function = {"name": "search_flights", **({} if case == "no schema" else {"parameters": schema})}
    definition = {"type": "function", "function": function}
    tool_id = "t.search" if case in ("same id", "no schema") else "t.child"
    call_ext = {} if case == "no id" else {"tool_id": tool_id}
    record["subagent_trajectories"] = [{
        "schema_version": "ATIF-v1.7", "trajectory_id": "child",
        "agent": {"name": "child", "version": "1", "tool_definitions": [definition]},
        "steps": [{"step_id": 1, "source": "agent", "message": "searching",
                   "tool_calls": [{"tool_call_id": "k1", "function_name": "search_flights", "arguments": {"child_query": "x"}}],
                   **({"extra": {"agentpile": {"calls": {"k1": call_ext}}}} if call_ext else {})}],
        "extra": {"agentpile": {"version": "0", "source": {"format": "synthetic"}, "agents": [], "tools": {
            tool_id: {"name": "search_flights", "agent_or_server": None, "type": None, "definition": definition,
                      "definition_format": "openai", "atif_index": 0, "permission": "read", "permission_from": "heuristic"}}}}}]
    run = ap.Run(record)
    assert ap.validate(run) == []
    by_call = {a["gen_ai.tool.call.id"]: a for a in map(attrs_of, spans_of(otlp.to_document(run)))
               if a.get("gen_ai.operation.name") == "execute_tool"}
    if case == "no schema":
        assert "tool.parameters" not in by_call["k1"]  # the subagent's own tool has none, and the parent's is another tool
    else:
        assert json.loads(by_call["k1"]["tool.parameters"]) == schema
    assert json.loads(by_call["c1"]["tool.parameters"])["required"] == ["origin", "dest"]  # the parent keeps its own
    assert next(otlp.read(otlp.to_document(run))) == run


def test_arguments_too_large_for_a_float_stay_source_text():
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    record["steps"][3]["extra"]["agentpile"]["calls"]["c3"]["arguments_raw"] = '{"offer_id": 1e400}'
    a = tool_spans(record)["book_flight"]
    assert a["input.value"] == '{"offer_id": 1e400}' and "gen_ai.tool.call.arguments" not in a


def tool_spans(record):
    return {a["gen_ai.tool.name"]: a for a in map(attrs_of, spans_of(otlp.to_document(ap.Run(record))))
            if a.get("gen_ai.operation.name") == "execute_tool"}


def test_tool_parameters_come_from_the_recorded_definition():
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    defs = {d["function"]["name"]: d["function"]["parameters"] for d in record["agent"]["tool_definitions"]}
    assert {n: json.loads(a["tool.parameters"]) for n, a in tool_spans(record).items()} == defs
    tools = record["extra"]["agentpile"]["tools"]
    mcp, anthropic = {"type": "object", "properties": {"q": {"type": "string"}}}, {"type": "object", "properties": {}}
    tools["t.search"]["definition"] = {"name": "search_flights", "inputSchema": mcp}
    tools["t.calendar"]["definition"] = {"name": "read_calendar", "input_schema": anthropic}
    spans = tool_spans(record)  # the call's tool id picks its entry, whatever an agent definition of that name says
    assert json.loads(spans["search_flights"]["tool.parameters"]) == mcp
    assert json.loads(spans["read_calendar"]["tool.parameters"]) == anthropic
    del record["steps"][3]["extra"]["agentpile"]["calls"]["c3"]["tool_id"]  # by name, where every definition agrees
    assert json.loads(tool_spans(record)["book_flight"]["tool.parameters"]) == defs["book_flight"]
    tools["t.book"]["definition"] = {"name": "book_flight", "parameters": {"type": "object"}}  # flat OpenAI shape
    assert "tool.parameters" not in tool_spans(record)["book_flight"]  # two definitions disagree: no guess
    record["agent"]["tool_definitions"] = []
    assert json.loads(tool_spans(record)["book_flight"]["tool.parameters"]) == {"type": "object"}
    # a second tool of that name whose schema the record lacks could be the one the call used: no guess either
    tools["t.book2"] = dict(tools["t.book"], definition=None, atif_index=None)
    assert "tool.parameters" not in tool_spans(record)["book_flight"]
    # a call naming its tool takes that tool's schema or none, never another tool's of the same name
    record["steps"][3]["extra"]["agentpile"]["calls"]["c3"]["tool_id"] = "t.book2"
    del tools["t.book"]
    tools["t.other"] = {"name": "book_flight", "definition": {"name": "book_flight", "parameters": {"type": "object"}}}
    assert "tool.parameters" not in tool_spans(record)["book_flight"]


# -- schemas and agreement --------------------------------------------------------------------------------------

SCHEMA_OF = {"gen_ai.input.messages": "gen-ai-input-messages", "gen_ai.output.messages": "gen-ai-output-messages",
             "gen_ai.system_instructions": "gen-ai-system-instructions",
             "gen_ai.tool.call.arguments": "gen-ai-tool-call-arguments", "gen_ai.tool.call.result": "gen-ai-tool-call-result",
             "gen_ai.tool.definitions": "gen-ai-tool-definitions"}


def test_generated_genai_values_fit_the_pinned_schemas():
    jsonschema = pytest.importorskip("jsonschema")
    validators = {key: jsonschema.Draft202012Validator(json.loads((SCHEMAS / (stem + ".json")).read_text(encoding="utf-8")))
                  for key, stem in SCHEMA_OF.items()}
    seen = Counter()
    for name, a in generated():
        for key, validator in validators.items():
            if key in a:
                errors = list(validator.iter_errors(strict_json(a[key])))
                assert not errors, (name, key, errors[0].message)
                seen[key] += 1
    assert seen["gen_ai.output.messages"] > 100 and seen["gen_ai.tool.call.arguments"] >= (88 if ARROW else 50)
    assert seen["gen_ai.tool.call.result"] >= 6 and seen["gen_ai.tool.definitions"] > 0
    assert not seen["gen_ai.input.messages"] and not seen["gen_ai.system_instructions"]


def test_where_both_conventions_hold_a_value_they_agree():
    for name, a in generated():
        for oi, genai in (("llm.model_name", "gen_ai.request.model"), ("tool.name", "gen_ai.tool.name"),
                          ("llm.token_count.prompt", "gen_ai.usage.input_tokens"),
                          ("llm.token_count.completion", "gen_ai.usage.output_tokens"),
                          ("llm.token_count.prompt_details.cache_read", "gen_ai.usage.cache_read.input_tokens")):
            assert (oi in a) == (genai in a) and a.get(oi) == a.get(genai), (name, oi)
        if "agent.name" in a and "gen_ai.agent.name" in a:
            assert a["agent.name"] == a["gen_ai.agent.name"]
        if "gen_ai.output.messages" in a:
            (message,) = json.loads(a["gen_ai.output.messages"])
            assert message["role"] == a["llm.output_messages.0.message.role"] == "assistant"
            texts = [p["content"] for p in message["parts"] if p["type"] == "text"]
            assert texts == [a[k] for k in ("llm.output_messages.0.message.content",) if k in a], name
            calls = [p for p in message["parts"] if p["type"] == "tool_call"]
            for j, part in enumerate(calls):
                base = "llm.output_messages.0.message.tool_calls.%d.tool_call." % j
                assert a.get(base + "id") == part["id"] and a[base + "function.name"] == part["name"]
                if "arguments" not in part:
                    assert base + "function.arguments" not in a
                elif isinstance(part["arguments"], str):  # source text: the same text in both
                    assert a[base + "function.arguments"] == part["arguments"]
                else:
                    assert json.loads(a[base + "function.arguments"]) == part["arguments"]
            assert not any(k.startswith("llm.output_messages.0.message.tool_calls.%d." % len(calls)) for k in a)
        if "gen_ai.tool.call.arguments" in a:  # input.value is the source text: where it is JSON, it is the same object
            assert otlp._json_object(a["input.value"]) in (None, json.loads(a["gen_ai.tool.call.arguments"])), name
        if "gen_ai.tool.call.result" in a:
            assert json.loads(a["output.value"]) == json.loads(a["gen_ai.tool.call.result"]), name


def test_generated_spans_fill_in_nothing_the_record_lacks():
    invented = {"gen_ai.provider.name", "gen_ai.response.finish_reasons", "gen_ai.response.model", "gen_ai.response.id",
                "llm.provider", "llm.system"}
    for name, a in generated():
        assert not invented & set(a), name
        if "gen_ai.output.messages" in a:
            assert all("finish_reason" not in m for m in json.loads(a["gen_ai.output.messages"])), name


# -- source spans -----------------------------------------------------------------------------------------------

def source_spans(path):
    text = path.read_text(encoding="utf-8")
    docs = [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
    return {s["spanId"]: s for d in docs for s in spans_of(d)}


def operations(attribute_dicts):
    return Counter((a.get("gen_ai.operation.name"), a.get("openinference.span.kind")) for a in attribute_dicts
                   if a.get("gen_ai.operation.name") or a.get("openinference.span.kind") not in (None, "CHAIN"))


@pytest.mark.parametrize("name", SOURCED)
def test_source_spans_keep_their_parents_times_attributes_and_operations(name):
    source = source_spans(FIX / name)
    spans = [s for n, _, _, sp, _ in exports() if n == name for s in sp]
    kept = {s["spanId"]: s for s in spans if adopted(attrs_of(s))}
    assert set(kept) == set(source)  # every source span is written, under its own id
    for sid, s in kept.items():
        src = source[sid]
        assert s["name"] == src["name"] and s["kind"] == src.get("kind", s["kind"])
        assert (int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"])) == (int(src["startTimeUnixNano"]), int(src["endTimeUnixNano"]))
        if src.get("parentSpanId"):
            assert s["parentSpanId"] == src["parentSpanId"]
        else:  # a source root sits under the span agentpile writes for its trajectory
            assert not adopted(attrs_of(next(p for p in spans if p["spanId"] == s["parentSpanId"])))
        mine = {k: v for k, v in attrs_of(s).items() if not k.startswith("agentpile.")}
        assert mine == otlp.decode_attributes(src["attributes"]), (name, s["name"])
        assert (s.get("status") or {}).get("code", 0) == (src.get("status") or {}).get("code", 0)
    assert operations(attrs_of(s) for s in spans) == operations(otlp.decode_attributes(s["attributes"]) for s in source.values())
