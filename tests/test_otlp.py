"""OTLP JSON reader and writer: exact round trips, framework spans, parsing, and unambiguous format detection.

Standard library only. The two framework fixtures under ``fixtures/otlp`` were captured from fake-model runs by
``fixtures/otlp/make_fixtures.py`` (Pydantic AI 2.58.0; OpenAI Agents SDK 0.24.0 through
openinference-instrumentation-openai-agents 2.5.3). Tests that check records against Harbor's models skip without Harbor.
"""

import copy
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats import otlp

FIX = Path(__file__).parent / "fixtures"
OTLP = FIX / "otlp"
CORPORA = Path(__file__).parent / "corpora" / "fixtures"
PAI = OTLP / "pydantic_ai_2.58.0.json"
OAI = OTLP / "openai_agents_0.24.0_oi_2.5.3.jsonl"
MISSING = ap.MISSING


def harbor_valid(record):
    trajectories = pytest.importorskip("harbor.models.trajectories")
    trajectories.Trajectory.model_validate(record)


def get(obj, key):
    return obj[key] if isinstance(obj, dict) and key in obj else MISSING


def spans_of(doc):
    return [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]


def attrs_of(span):
    return otlp.decode_attributes(span["attributes"])


def by_name(doc, name):
    return [s for s in spans_of(doc) if s["name"] == name]


def write_read(run, **kwargs):
    """Run -> OTLP JSON text -> Run, through the registered writer and reader."""
    text = ap.writers.get("otlp-json").dumps(run, **kwargs)
    runs = list(ap.readers.get("otlp-json").read(json.loads(text)))
    assert len(runs) == 1
    return runs[0]


# -- the must-keep fields, with missing, null and empty kept apart --------------------------------

def keep_trajectory(traj):
    ext = get(get(traj, "extra"), "agentpile")
    agent = get(traj, "agent")
    out = {
        "top": {k: get(traj, k) for k in ("schema_version", "session_id", "trajectory_id", "notes", "final_metrics",
                                          "continued_trajectory_ref")},
        "agent": {k: get(agent, k) for k in ("name", "version", "model_name", "tool_definitions", "extra")},
        "task": get(ext, "task"), "outcome": get(ext, "outcome"), "source": get(ext, "source"),
        "dataset": get(ext, "dataset"),
        "labels": [{k: get(lab, k) for k in ("name", "target", "value", "label_source", "source_ref")}
                   for lab in (get(ext, "labels") if get(ext, "labels") is not MISSING else [])],
        "agents": [{k: get(a, k) for k in ("id", "name", "instructions", "model", "tools", "handoffs")}
                   for a in (get(ext, "agents") if get(ext, "agents") is not MISSING else [])],
        "tools": {tid: {k: get(t, k) for k in ("name", "agent_or_server", "type", "definition", "definition_format",
                                               "atif_index", "permission", "permission_from")}
                  for tid, t in (get(ext, "tools") if get(ext, "tools") is not MISSING else {}).items()},
        "steps": [],
        "subagents": [keep_trajectory(s) for s in (get(traj, "subagent_trajectories") if get(traj, "subagent_trajectories") is not MISSING else [])],
    }
    for step in traj["steps"]:
        sx = get(get(step, "extra"), "agentpile")
        item = {k: get(step, k) for k in ("step_id", "source", "timestamp", "message", "model_name", "reasoning_effort",
                                          "reasoning_content", "metrics", "llm_call_count", "is_copied_context")}
        item["sx"] = {k: get(sx, k) for k in ("actor", "role", "name", "message_state", "source_ref", "source_fields",
                                              "empty_source", "result_refs", "result_fields")}
        item["calls"] = []
        for call in get(step, "tool_calls") if isinstance(get(step, "tool_calls"), list) else []:
            cx = get(get(sx, "calls"), call["tool_call_id"])
            item["calls"].append({"call": copy.deepcopy(call),
                                  "cx": {k: get(cx, k) for k in ("tool_id", "span_id", "parent", "error", "arguments_raw",
                                                                  "arguments_state", "arguments_stand_in",
                                                                  "id_generated", "source_id", "function_name_raw", "source_ref",
                                                                  "source_fields")}})
        obs = get(step, "observation")
        item["results"] = [{k: get(r, k) for k in ("source_call_id", "content", "subagent_trajectory_ref", "extra")}
                           for r in (get(obs, "results") if isinstance(obs, dict) else [])]
        item["deps"] = [{k: get(d, k) for k in ("src", "dst", "rule", "rule_version", "resource")}
                        for d in (get(sx, "deps") if isinstance(get(sx, "deps"), list) else [])]
        out["steps"].append(item)
    return out


def keep(run):
    return keep_trajectory(run.record)


def with_deps(run, rules=("tool_call_id_v1", "full_context_v1")):
    """Add the derived dependency entries to the step they point at, as a person would before writing."""
    record = copy.deepcopy(run.record)
    for dep in ap.derive_deps(ap.Run(record), rules=list(rules)):
        step = record["steps"][dep["dst"]["step"] - 1]
        step.setdefault("extra", {}).setdefault("agentpile", {}).setdefault("deps", []).append(dep)
    return ap.Run(record)


def all_atif_runs():
    runs = [(f"{path.name}[{i}]", run)
            for path in sorted(FIX.glob("atif_*")) for i, run in enumerate(ap.load(path))]
    for name in ("who_and_when", "mast"):
        runs += [(f"{name}[{i}]", run) for i, run in enumerate(ap.load(name, root=str(CORPORA / name)))]
    return runs


RUNS = all_atif_runs()
RUN_IDS = [name for name, _ in RUNS]


# -- (a) ATIF -> OTLP JSON -> ATIF ------------------------------------------------------------------

@pytest.mark.parametrize("run", [r for _, r in RUNS], ids=RUN_IDS)
def test_round_trip_keeps_every_field(run):
    back = write_read(run)
    assert keep(back) == keep(run)
    assert back == run


@pytest.mark.parametrize("run", [r for _, r in RUNS], ids=RUN_IDS)
def test_round_trip_keeps_derived_dependencies(run):
    wired = with_deps(run)
    back = write_read(wired)
    assert keep(back) == keep(wired)
    assert back == wired
    wanted = sum(len(s["deps"]) for s in keep(wired)["steps"])
    links = [lk for s in spans_of(ap.writers.get("otlp-json").to_document(wired)) for lk in s.get("links", [])]
    assert len(links) == wanted


def test_round_trip_through_a_file_and_the_loader(tmp_path):
    run = next(ap.load(FIX / "atif_multiagent.json"))
    for name in ("trace.json", "trace.jsonl"):
        run.to_otlp(tmp_path / name)
        loaded = list(ap.load(tmp_path / name))
        assert loaded == [run]
    assert json.loads(run.to_otlp()) == otlp.to_document(run)


def test_jsonl_append_writes_one_document_per_line(tmp_path):
    runs = list(ap.load(FIX / "atif_two.jsonl"))
    path = tmp_path / "many.jsonl"
    for i, run in enumerate(runs):
        ap.writers.get("otlp-json").write(run, path, append=i > 0)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    assert list(ap.load(path)) == runs
    with pytest.raises(ValueError, match="append"):
        ap.writers.get("otlp-json").write(runs[0], tmp_path / "x.json", append=True)


def tricky_record():
    """Valid in shape but hostile: nulls, empty strings, absent keys, odd extras, unreferenced extension entries."""
    sub = {"schema_version": "ATIF-v1.7", "session_id": "s-sub", "trajectory_id": "sub-1",
           "agent": {"name": "child", "version": "1"},
           "steps": [{"step_id": 1, "source": "user", "message": "go", "timestamp": "2026-01-01T00:00:05+00:00"},
                     {"step_id": 2, "source": "agent", "message": "done", "model_name": "m-child"}],
           "extra": {"agentpile": {"version": "0", "source": {"format": "synthetic"}, "agents": [], "tools": {}}},
           "subagent_trajectories": [{"schema_version": "ATIF-v1.7", "trajectory_id": "sub-sub",
                                      "agent": {"name": "grandchild", "version": "1"},
                                      "steps": [{"step_id": 1, "source": "agent", "message": "deep"}]}]}
    return {
        "schema_version": "ATIF-v1.7", "session_id": None, "trajectory_id": "t-tricky",
        "agent": {"name": "a", "version": "1", "model_name": None, "tool_definitions": [],
                  "extra": {"k": [1, 2.5, None, "x", {"y": False}], "big": 2 ** 70, "inf": float("inf")}},
        "notes": "", "final_metrics": {"total_prompt_tokens": 10, "total_cost_usd": 0.0, "extra": {}},
        "continued_trajectory_ref": None,
        "extra": {"other": {"deep": [1, 2]}, "agentpile": {
            "version": "0", "source": {"format": "synthetic", "dataset": None, "lost_fields": []},
            "task": {"id": "", "text": None}, "outcome": {"success": None, "score": 0, "raw": ""},
            "labels": [{"name": None, "target": {"kind": "run", "address": {"trajectory": "t-tricky"}}, "value": "",
                        "label_source": "x"},
                       {"name": "n", "target": {"kind": "call", "address": {"trajectory": "t-tricky", "step": 5, "call": "c2"}},
                        "value": None, "label_source": "x", "source_ref": {"nested": [None]}}],
            "agents": [{"id": "x", "name": None, "instructions": "", "model": None, "tools": ["t1"], "handoffs": []}],
            "tools": {"t1": {"name": "f", "agent_or_server": None, "type": None, "definition": None,
                             "definition_format": None, "atif_index": None, "permission": "unknown", "permission_from": "unknown"}},
            "dataset": {"anything": ["goes", {"here": None}]}}},
        "steps": [
            {"step_id": 1, "source": "system", "message": "", "timestamp": "2026-01-01T00:00:00Z"},
            {"step_id": 2, "source": "user", "extra": {},
             "message": [{"type": "text", "text": "hi"}, {"type": "image", "source": {"media_type": "image/png", "path": "a.png"}}]},
            {"step_id": 3, "source": "agent", "message": "", "timestamp": "not a time",
             "extra": {"agentpile": {"message_state": "null"}}},
            {"step_id": 4, "source": "agent", "message": "", "timestamp": None, "model_name": None, "reasoning_effort": 0.5,
             "reasoning_content": "", "llm_call_count": 0, "is_copied_context": False, "tool_calls": [],
             "metrics": {"prompt_tokens": 1, "extra": None}, "extra": {"agentpile": {"message_state": "missing"}}},
            {"step_id": 5, "source": "agent", "message": "x",
             "tool_calls": [{"tool_call_id": "c1", "function_name": "f", "arguments": {}, "extra": {"z": 1}},
                            {"tool_call_id": "c2", "function_name": "g", "arguments": {"a": [1, {"b": None}]}}],
             "observation": {"results": [
                 {"source_call_id": "c2", "content": None},
                 {"source_call_id": "c1", "content": ""},
                 {"source_call_id": None, "content": [{"type": "text", "text": "t"}], "extra": {"e": 1},
                  "subagent_trajectory_ref": [{"trajectory_id": "sub-1", "extra": {"why": "x"}}]}]},
             "extra": {"agentpile": {"actor": "x", "role": None, "name": 0,
                                     "calls": {"c1": {"arguments_raw": None}, "c2": {}, "ghost": {"error": "e"}},
                                     "result_refs": [0, 1, 2], "result_fields": [{"content_state": "null"}, None, {}],
                                     "deps": []}}},
            {"step_id": 6, "source": "agent", "message": "y", "tool_calls": None, "observation": {"results": []},
             "extra": {"agentpile": {"deps": None, "calls": {}}}},
            {"step_id": 7, "source": "agent", "message": "z", "extra": None},
            {"step_id": 8, "source": "agent", "message": "w", "extra": {"agentpile": {}}},
        ],
        "subagent_trajectories": [sub],
    }


def test_round_trip_keeps_missing_null_and_empty_apart():
    run = ap.Run(tricky_record())
    back = write_read(run)
    assert back == run
    assert keep(back) == keep(run)
    kept = keep(back)
    assert kept["top"]["session_id"] is None and kept["top"]["notes"] == ""
    assert kept["top"]["continued_trajectory_ref"] is None
    assert kept["agent"]["model_name"] is None and kept["agent"]["extra"]["big"] == 2 ** 70
    assert kept["steps"][2]["sx"]["message_state"] == "null" and kept["steps"][3]["sx"]["message_state"] == "missing"
    assert kept["steps"][2]["timestamp"] == "not a time" and kept["steps"][3]["timestamp"] is None
    assert kept["steps"][0]["timestamp"] == "2026-01-01T00:00:00Z" and kept["steps"][1]["timestamp"] is MISSING
    assert kept["steps"][4]["results"][0]["content"] is None and kept["steps"][4]["results"][1]["content"] == ""
    assert get(back.record["steps"][3], "tool_calls") == [] and get(back.record["steps"][5], "tool_calls") is None
    assert get(back.record["steps"][1], "extra") == {} and get(back.record["steps"][6], "extra") is None
    assert get(back.record["steps"][7], "extra") == {"agentpile": {}}
    assert get(back.record["steps"][4]["extra"]["agentpile"]["calls"], "ghost") == {"error": "e"}
    assert back.record["steps"][5]["extra"]["agentpile"] == {"deps": None, "calls": {}}
    assert back.record["subagent_trajectories"][0]["subagent_trajectories"][0]["trajectory_id"] == "sub-sub"


def test_mutating_the_view_attributes_changes_nothing_but_mutating_the_record_does():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    doc = otlp.to_document(run)
    for sp in spans_of(doc):
        for kv in sp["attributes"]:
            if not kv["key"].startswith("agentpile."):
                kv["value"] = {"stringValue": "tampered"}
    assert list(otlp.read(doc)) == [run]
    step = [s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "step"][2]
    for kv in step["attributes"]:
        if kv["key"] == "agentpile.step.message":
            kv["value"] = {"stringValue": json.dumps("changed")}
    assert next(otlp.read(doc)).record["steps"][2]["message"] == "changed"


def test_nothing_is_truncated():
    run = ap.Run(tricky_record())
    big = "é" * 200_000 + " end"
    run.record["steps"][0]["message"] = big
    run.record["steps"][4]["observation"]["results"][1]["content"] = big
    doc = otlp.to_document(run)
    call = by_name(doc, "execute_tool f")[0]
    assert attrs_of(call)["gen_ai.tool.call.result"] == big
    system = by_name(doc, "system_message")[0]
    assert attrs_of(system)["input.value"] == big
    assert write_read(run) == run


def test_lone_surrogates_survive_a_file(tmp_path):
    run = ap.Run(tricky_record())
    run.record["steps"][0]["message"] = "bad \ud800 half"
    run.to_otlp(tmp_path / "s.json")
    assert next(ap.load(tmp_path / "s.json")) == run


# -- writer structure -------------------------------------------------------------------------------

def test_span_structure_follows_the_harbor_converter():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    doc = otlp.to_document(run)
    spans = spans_of(doc)
    assert len(spans) == 1 + 5 + 3
    assert len({s["traceId"] for s in spans}) == 1 and len({s["spanId"] for s in spans}) == len(spans)
    root = [s for s in spans if "parentSpanId" not in s]
    assert len(root) == 1 and root[0]["name"] == "invoke_agent synthetic-travel"
    ra = attrs_of(root[0])
    assert ra["openinference.span.kind"] == "AGENT" and ra["gen_ai.operation.name"] == "invoke_agent"
    assert ra["gen_ai.agent.name"] == "synthetic-travel" and ra["gen_ai.request.model"] == "fake-model"
    assert ra["input.value"].startswith("Book me a flight")
    assert all(s["parentSpanId"] == root[0]["spanId"] for s in spans if s is not root[0])
    kinds = [attrs_of(s).get("openinference.span.kind") for s in spans]
    assert kinds.count("TOOL") == 3 and kinds.count("LLM") == 3 and kinds.count("CHAIN") == 2
    user, system = by_name(doc, "user_message")[0], by_name(doc, "system_message")[0]
    for s in (user, system):
        assert attrs_of(s)["openinference.span.kind"] == "CHAIN" and "gen_ai.operation.name" not in attrs_of(s)
    assert attrs_of(user)["input.value"].startswith("Book me")


def test_tool_span_attributes_follow_genai_v1_41_0():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    doc = otlp.to_document(run)
    search = by_name(doc, "execute_tool search_flights")[0]
    a = attrs_of(search)
    assert a["gen_ai.operation.name"] == "execute_tool" and a["gen_ai.tool.name"] == "search_flights"
    assert a["gen_ai.tool.call.id"] == "c1"
    assert json.loads(a["gen_ai.tool.call.arguments"]) == {"origin": "LAX", "dest": "SFO"}
    assert a["gen_ai.tool.call.result"] == '[{"offer_id": 7, "depart": "09:00"}]'
    assert a["tool.name"] == "search_flights" and a["openinference.span.kind"] == "TOOL"
    book = attrs_of(by_name(doc, "execute_tool book_flight")[0])
    assert book["gen_ai.tool.call.arguments"] == "{'offer_id': 7"  # the raw text the source gave
    assert by_name(doc, "execute_tool book_flight")[0]["status"]["code"] == 2
    assert book["error.type"] == "tool_error"
    step = attrs_of([s for s in spans_of(doc) if s["name"] == "chat"][0])
    assert step["gen_ai.operation.name"] == "chat" and step["gen_ai.agent.name"] == "Planner"
    out = json.loads(step["gen_ai.output.messages"])
    assert [p["type"] for p in out[0]["parts"]] == ["text", "tool_call", "tool_call"]
    assert out[0]["parts"][1]["name"] == "search_flights" and out[0]["parts"][1]["id"] == "c1"
    assert step["llm.output_messages.0.message.tool_calls.1.tool_call.function.name"] == "read_calendar"


def test_trace_id_comes_from_the_run_and_output_is_stable():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    assert run.to_otlp() == run.to_otlp()
    renamed = ap.Run(dict(run.record, trajectory_id="other"))
    assert spans_of(otlp.to_document(run))[0]["traceId"] != spans_of(otlp.to_document(renamed))[0]["traceId"]
    hexed = ap.Run(dict(run.record, trajectory_id="0123456789abcdef0123456789abcdef"))
    assert spans_of(otlp.to_document(hexed))[0]["traceId"] == "0123456789abcdef0123456789abcdef"
    assert spans_of(otlp.to_document(run, trace_id="ab" * 16))[0]["traceId"] == "ab" * 16
    with pytest.raises(ValueError):
        otlp.to_document(run, trace_id="short")


def test_a_config_record_has_no_spans_to_write():
    config = next(ap.load(FIX / "config_mcp.json"))
    with pytest.raises(ValueError, match="config"):
        config.to_otlp()


def test_dependency_entries_become_span_links():
    run = with_deps(next(ap.load(FIX / "atif_multiagent.json")), rules=("tool_call_id_v1",))
    run.record["steps"][4]["extra"]["agentpile"]["deps"][0]["resource"] = "file:/tmp/x"
    doc = otlp.to_document(run)
    spans = {s["spanId"]: s for s in spans_of(doc)}
    final = [s for s in spans.values() if attrs_of(s).get("agentpile.index") == 4 and attrs_of(s).get("agentpile.kind") == "step"][0]
    links = final["links"]
    assert len(links) == 1
    la = attrs_of(links[0])
    assert la["agentpile.dep.rule"] == "tool_call_id" and la["agentpile.dep.rule_version"] == "1"
    assert la["agentpile.dep.resource"] == "file:/tmp/x" and la["agentpile.dep.resolved"] is True
    target = spans[links[0]["spanId"]]
    assert target["name"] == "execute_tool book_flight"  # the src end: the call whose result step 5 consumed
    assert json.loads(la["agentpile.dep"])["dst"] == {"trajectory": "fx-1", "step": 5}
    step4 = [s for s in spans.values() if attrs_of(s).get("agentpile.index") == 3 and attrs_of(s).get("agentpile.kind") == "step"][0]
    assert [spans[lk["spanId"]]["name"] for lk in step4["links"]] == ["execute_tool search_flights", "execute_tool read_calendar"]
    assert write_read(run) == run


def test_dependency_ends_that_resolve_to_no_span_still_round_trip():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    dep = {"src": {"trajectory": "fx-1", "agent": "ghost"}, "dst": {"trajectory": "fx-1", "step": 5}, "rule": "r",
           "rule_version": "2", "resource": None}
    odd = {"src": {"step": 99}, "dst": {"trajectory": None, "step": 3, "result": 1}, "rule": "r", "rule_version": "2"}
    run.record["steps"][4].setdefault("extra", {}).setdefault("agentpile", {})["deps"] = [dep, odd, "not a dict"]
    doc = otlp.to_document(run)
    spans = {s["spanId"]: s for s in spans_of(doc)}
    links = [lk for s in spans.values() for lk in s.get("links", [])]
    assert [attrs_of(lk).get("agentpile.dep.resolved") for lk in links] == [False, True, False]
    assert spans[links[0]["spanId"]]["name"].startswith("invoke_agent")  # an unresolved end links to the trajectory span
    assert spans[links[1]["spanId"]]["name"] == "execute_tool search_flights"  # a result resolves to its call
    assert write_read(run) == run


def test_steps_without_timestamps_get_marked_increasing_times():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    doc = otlp.to_document(run)
    steps = sorted((s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "step"), key=lambda s: attrs_of(s)["agentpile.index"])
    starts = [int(s["startTimeUnixNano"]) for s in steps]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert all(attrs_of(s)["agentpile.time_synthetic"] is True for s in steps)
    back = next(otlp.read(doc))
    assert all("timestamp" not in s for s in back.steps)


def test_source_timestamps_are_kept_and_only_gaps_are_synthetic():
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    record["steps"][1]["timestamp"] = "2026-03-01T10:00:00.123456789Z"
    record["steps"][3]["timestamp"] = "2026-03-01T12:00:00+02:00"
    run = ap.Run(record)
    doc = otlp.to_document(run)
    steps = sorted((s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "step"), key=lambda s: attrs_of(s)["agentpile.index"])
    assert int(steps[1]["startTimeUnixNano"]) == otlp.parse_iso_ns("2026-03-01T10:00:00.123456789Z") == 1772359200123456789
    assert int(steps[3]["startTimeUnixNano"]) == otlp.parse_iso_ns("2026-03-01T10:00:00Z")
    flags = [attrs_of(s).get("agentpile.time_synthetic", False) for s in steps]
    assert flags == [True, False, True, False, True]
    assert int(steps[0]["startTimeUnixNano"]) < int(steps[1]["startTimeUnixNano"])
    back = write_read(run)
    assert [get(s, "timestamp") for s in back.steps] == [MISSING, "2026-03-01T10:00:00.123456789Z", MISSING,
                                                         "2026-03-01T12:00:00+02:00", MISSING]
    assert back == run


def test_iso_helpers():
    assert otlp.parse_iso_ns("1970-01-01T00:00:01Z") == 10 ** 9
    assert otlp.parse_iso_ns("1970-01-01 01:00:00+01:00") == 0
    assert otlp.parse_iso_ns("1970-01-01T00:00:00.5") == 500_000_000
    assert otlp.parse_iso_ns("2026-13-01T00:00:00Z") is None and otlp.parse_iso_ns(5) is None
    assert otlp.iso_from_ns(1792487964564038000) == "2026-10-20T09:19:24.564038Z"


def trail_like_record():
    """A run that kept source spans: a CHAIN span with no kind, and a child, with labels aimed at them."""
    run = next(ap.load(FIX / "atif_multiagent.json"))
    record = copy.deepcopy(run.record)
    ext = record["extra"]["agentpile"]
    ext["spans"] = {
        "a1b2c3d4e5f60718": {"name": "Root chain", "kind": None, "parent": None, "trace_id": "f" * 32,
                             "start_time_unix_nano": 1_700_000_000_000_000_000, "end_time_unix_nano": 1_700_000_000_500_000_000,
                             "status": {"code": "OK"}, "attributes": {"openinference.span.kind": "CHAIN", "n": 3, "f": 0.5,
                                                                      "flag": True, "nested": {"a": [1, None]}, "empty": ""},
                             "events": [{"name": "exception", "time_unix_nano": 1_700_000_000_100_000_000,
                                         "attributes": {"exception.message": "boom"}}],
                             "links": [{"trace_id": "e" * 32, "span_id": "d" * 16, "attributes": {"k": "v"}}], "address": None},
        "not-hex": {"name": None, "kind": "CLIENT", "parent": "a1b2c3d4e5f60718", "trace_id": None,
                    "start_time_unix_nano": None, "end_time_unix_nano": None, "status": None, "attributes": {},
                    "events": [], "links": [], "address": {"trajectory": "fx-1", "step": 3}},
    }
    ext["labels"].append({"name": "root_cause", "target": {"kind": "span", "address": {"trajectory": "fx-1", "span": "a1b2c3d4e5f60718"}},
                          "value": True, "label_source": "fixture"})
    ext["labels"].append({"name": "x", "target": {"kind": "span", "address": {"trajectory": "fx-1", "span": "not-hex"}},
                          "value": None, "label_source": "fixture"})
    return ap.Run(record)


def test_kept_source_spans_are_written_back_as_ordinary_spans():
    run = trail_like_record()
    doc = otlp.to_document(run)
    root_chain = [s for s in spans_of(doc) if s["spanId"] == "a1b2c3d4e5f60718"]
    assert len(root_chain) == 1
    s = root_chain[0]
    assert s["name"] == "Root chain" and s["kind"] == 0 and s["status"]["code"] == 1
    a = attrs_of(s)
    assert a["openinference.span.kind"] == "CHAIN" and a["n"] == 3 and a["f"] == 0.5 and a["flag"] is True
    assert a["nested"] == {"a": [1, None]} and a["empty"] == ""
    assert s["events"][0]["name"] == "exception" and attrs_of(s["events"][0])["exception.message"] == "boom"
    assert s["links"][0]["spanId"] == "d" * 16
    child = [x for x in spans_of(doc) if attrs_of(x).get("agentpile.span_key") == "not-hex"][0]
    assert child["name"] == "" and child["kind"] == 3 and child["parentSpanId"] == "a1b2c3d4e5f60718"
    assert child["spanId"] != "not-hex" and len(child["spanId"]) == 16
    assert len({x["spanId"] for x in spans_of(doc)}) == len(spans_of(doc))
    back = write_read(run)
    assert back == run
    assert list(back.spans) == ["a1b2c3d4e5f60718", "not-hex"]
    for label in back.labels[-2:]:
        assert label["target"]["address"]["span"] in back.spans


# -- (b) framework spans -----------------------------------------------------------------------------

def count_spans(source):
    return len(otlp._parse_spans(source))


def test_pydantic_ai_spans_become_valid_atif():
    runs = list(ap.load(PAI))
    assert len(runs) == 2  # one run per trace id, in first-seen order
    run = runs[0]
    assert ap.validate(run) == []
    harbor_valid(run.record)
    assert run.id == run.source["record_key"] and len(run.id) == 32
    steps = run.steps
    assert [s["source"] for s in steps] == ["user", "agent", "agent"]
    assert steps[0]["message"] == "What is the weather in Paris and what time is it?"
    first = steps[1]
    assert first["message"] == "Let me check both." and first["model_name"] == "fake-fn"
    assert [(c["tool_call_id"], c["function_name"], c["arguments"]) for c in first["tool_calls"]] == [
        ("call_weather_1", "get_weather", {"city": "Paris"}), ("call_time_1", "get_time", {"zone": "UTC"})]
    results = first["observation"]["results"]
    assert [(r["source_call_id"], r["content"]) for r in results] == [
        ("call_weather_1", "sunny in Paris"), ("call_time_1", '{"zone":"UTC","time":"12:00"}')]
    assert steps[2]["message"] == "It is sunny in Paris and 12:00 UTC." and "tool_calls" not in steps[2]
    assert first["metrics"] == {"prompt_tokens": 61, "completion_tokens": 15}
    assert first["timestamp"].endswith("Z") and "T" in first["timestamp"] and "timestamp" not in steps[0]
    assert [a["id"] for a in run.agents] == ["weather_agent"]
    agent = run.agents[0]
    assert agent["instructions"] == "You answer questions about weather and time." and agent["model"] == "fake-fn"
    assert [run.tools[t]["name"] for t in agent["tools"]] == ["get_weather", "get_time"]
    assert run.tools["weather_agent/get_time"]["permission"] == "unknown"
    assert [t["function"]["name"] for t in run.record["agent"]["tool_definitions"]] == ["get_weather", "get_time"]
    assert first["extra"]["agentpile"]["calls"]["call_time_1"]["tool_id"] == "weather_agent/get_time"
    second = runs[1]
    assert ap.validate(second) == []
    assert [s["message"] for s in second.steps] == ["Thanks.", "Anytime."] and "tool_calls" not in second.steps[1]
    assert second.id != run.id


def test_pydantic_ai_keeps_every_span_with_the_address_it_became():
    doc = json.loads(PAI.read_text(encoding="utf-8"))
    run = next(ap.load(PAI))
    ids = {s["spanId"] for s in spans_of(doc) if s["traceId"] == run.id}
    assert set(run.spans) == ids and len(ids) == 5
    by_op = {}
    for key, entry in run.spans.items():
        by_op.setdefault(entry["attributes"]["gen_ai.operation.name"], []).append(entry)
    assert sorted(by_op) == ["chat", "execute_tool", "invoke_agent"]
    assert [e["address"] for e in sorted(by_op["chat"], key=lambda e: e["start_time_unix_nano"])] == [
        {"trajectory": run.id, "step": 2}, {"trajectory": run.id, "step": 3}]
    assert sorted(e["address"]["call"] for e in by_op["execute_tool"]) == ["call_time_1", "call_weather_1"]
    assert all(e["address"]["step"] == 2 for e in by_op["execute_tool"])
    assert by_op["invoke_agent"][0]["address"] is None
    chat = by_op["chat"][0]
    assert chat["kind"] == "CLIENT" and chat["trace_id"] == run.id and isinstance(chat["start_time_unix_nano"], int)
    assert chat["attributes"]["gen_ai.input.messages"].startswith('[{"role":"user"')  # attributes are kept verbatim
    assert chat["resource"]["service.name"] == "fixture-pydantic-ai" and chat["scope"]["version"] == "2.58.0"


def test_pydantic_ai_source_names_the_dialect_and_its_versions():
    run = next(ap.load(PAI))
    src = run.source
    assert src["dialect"] == "pydantic-ai/genai" and src["dialect_tested"] is True
    assert src["scopes"] == [{"name": "pydantic-ai", "version": "2.58.0"}]
    assert src["format"] == "otlp-json" and src["reader"] == "otlp-json" and "semconv_version" not in src  # the spans declare none
    assert src["lost_fields"] == []


def test_a_declared_schema_url_gives_the_semantic_convention_version():
    doc = json.loads(PAI.read_text(encoding="utf-8"))
    doc["resourceSpans"][0]["scopeSpans"][0]["schemaUrl"] = "https://opentelemetry.io/schemas/1.37.0"
    run = next(otlp.read(doc))
    assert run.source["semconv_version"] == "1.37.0"
    assert run.source["schema_urls"] == ["https://opentelemetry.io/schemas/1.37.0"]
    assert run.spans[next(iter(run.spans))]["schema_url"] == "https://opentelemetry.io/schemas/1.37.0"


def with_scope_version(path, version):
    """The framework fixture's spans with the instrumentation scope of every span set to another version."""
    docs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
    for doc in docs:
        for rs in doc["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                if version is None:
                    ss["scope"].pop("version", None)
                else:
                    ss["scope"]["version"] = version
    return docs


@pytest.mark.parametrize("path, scope, tested", [(PAI, "pydantic-ai", "2.58.0"),
                                                 (OAI, "openinference.instrumentation.openai_agents", "2.5.3")],
                         ids=["pydantic-ai", "openai-agents"])
def test_an_untested_version_of_a_known_dialect_is_not_interpreted(path, scope, tested):
    for version in ("9.9.9", None, "2.58.01"):
        docs = with_scope_version(path, version)
        with pytest.raises(ap.UnsupportedVersion) as err:
            list(otlp.read(docs))
        assert scope in str(err.value) and f"tested: {tested}" in str(err.value) and "allow_untested_version" in str(err.value)
        with pytest.raises(ap.UnsupportedVersion):  # the loader passes the same refusal on
            list(ap.load(docs, reader="otlp-json"))
    control = list(otlp.read(with_scope_version(path, tested)))  # the tested version reads, and nothing says otherwise
    assert control and all(r.source["dialect_tested"] is True for r in control)
    assert not any("not tested" in f for r in control for f in r.source["lost_fields"])


def test_one_untested_span_among_tested_ones_refuses_the_whole_trace():
    doc = json.loads(PAI.read_text(encoding="utf-8"))
    extra = copy.deepcopy(doc["resourceSpans"][0]["scopeSpans"][0])
    extra["scope"]["version"] = "3.0.0"
    for sp in extra["spans"]:
        sp["spanId"] = sp["spanId"][::-1]
    doc["resourceSpans"][0]["scopeSpans"].append(extra)
    with pytest.raises(ap.UnsupportedVersion, match="3.0.0"):
        list(otlp.read(doc))


def test_an_untested_version_is_read_only_on_request_and_the_run_says_so():
    docs = with_scope_version(PAI, "9.9.9")
    run = next(otlp.read(docs, allow_untested_version=True))
    assert run.source["dialect"] == "pydantic-ai/genai" and run.source["dialect_tested"] is False
    assert run.source["scopes"] == [{"name": "pydantic-ai", "version": "9.9.9"}]
    assert any("9.9.9" in f and "not tested" in f and "2.58.0" in f for f in run.source["lost_fields"])
    assert [s["source"] for s in run.steps] == ["user", "agent", "agent"]
    assert run == next(ap.load(docs, reader="otlp-json", allow_untested_version=True))
    assert write_read(run) == run  # the marking is part of the record and survives the writer
    tested = next(ap.load(PAI))
    assert tested.source["dialect_tested"] is True and not any("not tested" in f for f in tested.source["lost_fields"])


def test_an_unknown_scope_or_an_unmarked_one_is_not_a_dialect_and_needs_no_opt_in():
    for scope, attrs in ((("some-other-instrumentation", "9.9.9"), {"gen_ai.operation.name": "chat"}),
                         (("pydantic-ai", "9.9.9"), {"not.a.genai.span": "x"})):
        run = next(otlp.read(make_doc([("0000000000000001", None, "s", attrs, 1)], scope=scope)))
        assert run.source["dialect"] == "generic"  # kept as spans and nothing interpreted, so there is no layout to get wrong


def test_openai_agents_spans_become_valid_atif():
    runs = list(ap.load(OAI))  # one trace spread over three JSONL lines
    assert len(runs) == 1
    run = runs[0]
    assert ap.validate(run) == []
    harbor_valid(run.record)
    steps = run.steps
    assert [(s["source"], s["message"]) for s in steps] == [
        ("system", "Route the customer."), ("user", "Where is order A1? I want a refund."), ("agent", "Checking."),
        ("agent", ""), ("system", "Handle refunds."), ("agent", "Refund started for order A1.")]
    lookup, handoff, final = steps[2], steps[3], steps[5]
    assert [(c["tool_call_id"], c["function_name"], c["arguments"]) for c in lookup["tool_calls"]] == [
        ("call_lookup_1", "lookup_order", {"order_id": "A1"})]
    assert lookup["observation"]["results"] == [{"source_call_id": "call_lookup_1", "content": "order A1: shipped"}]
    assert [(c["tool_call_id"], c["function_name"], c["arguments"]) for c in handoff["tool_calls"]] == [
        ("call_handoff_1", "transfer_to_refunds", {})]  # the required ATIF object; the call extension says it is rebuilt
    cx = handoff["extra"]["agentpile"]["calls"]["call_handoff_1"]
    assert "openinference" in cx["arguments_stand_in"] and "'{}'" in cx["arguments_stand_in"] and "arguments_state" not in cx
    assert "arguments_stand_in" not in lookup["extra"]["agentpile"]["calls"]["call_lookup_1"]  # recorded arguments are never marked
    assert handoff["observation"]["results"] == [{"source_call_id": "call_handoff_1", "content": '{"assistant": "Refunds"}'}]
    assert handoff["extra"]["agentpile"]["message_state"] == "missing" and "tool_calls" not in final
    assert [s["extra"]["agentpile"].get("actor") for s in steps] == [None, None, "Triage", "Triage", None, "Refunds"]
    assert [a["id"] for a in run.agents] == ["Triage", "Refunds"]
    triage, refunds = run.agents
    assert triage["handoffs"] == ["Refunds"] and refunds["handoffs"] == []
    assert triage["instructions"] == "Route the customer." and refunds["instructions"] == "Handle refunds."
    assert [run.tools[t]["name"] for t in triage["tools"]] == ["lookup_order", "transfer_to_refunds"]
    assert run.record["agent"]["name"] == "Triage" and len(run.record["agent"]["tool_definitions"]) == 2
    assert lookup["metrics"] == {"prompt_tokens": 30, "completion_tokens": 10, "cached_tokens": 0}
    src = run.source
    assert src["dialect"] == "openinference/openai-agents" and src["dialect_tested"] is True
    assert src["scopes"] == [{"name": "openinference.instrumentation.openai_agents", "version": "2.5.3"}]
    assert any("TOOL span" in f and "tool call id" in f for f in src["lost_fields"])


def test_openai_agents_keeps_every_span_and_pairs_by_call_id_only():
    run = next(ap.load(OAI))
    docs = [json.loads(line) for line in OAI.read_text(encoding="utf-8").splitlines()]
    assert len(docs) == 3 and len(run.spans) == 12 == count_spans(docs)
    assert {s["traceId"] for d in docs for s in spans_of(d)} == {run.id}
    kinds = {}
    for entry in run.spans.values():
        kinds.setdefault(entry["attributes"]["openinference.span.kind"], []).append(entry)
    assert {k: len(v) for k, v in kinds.items()} == {"LLM": 3, "TOOL": 2, "AGENT": 3, "CHAIN": 4}
    llm_addresses = sorted(e["address"]["step"] for e in kinds["LLM"])
    assert llm_addresses == [3, 4, 6]
    # a TOOL span names no call id, so it is neither a result nor matched by name or position
    assert all(e["address"] is None for e in kinds["TOOL"]) and all(e["address"] is None for e in kinds["AGENT"])
    assert {e["attributes"].get("tool.name", e["name"]) for e in kinds["TOOL"]} == {"lookup_order", "handoff to Refunds"}
    refunds_agent = [e for e in kinds["AGENT"] if e["attributes"].get("agent.name") == "Refunds"][0]
    assert refunds_agent["attributes"]["graph.node.parent_id"] == "Triage"


@pytest.mark.parametrize("path", [PAI, OAI], ids=["pydantic-ai", "openai-agents"])
def test_framework_runs_round_trip_through_the_writer(path):
    for run in ap.load(path):
        back = write_read(run)
        assert back == run
        assert keep(back) == keep(run)
        assert ap.validate(back) == []
        harbor_valid(back.record)
        wired = with_deps(run, rules=("tool_call_id_v1",))
        assert write_read(wired) == wired


@pytest.mark.parametrize("path", [PAI, OAI], ids=["pydantic-ai", "openai-agents"])
def test_written_framework_runs_keep_their_original_spans_resolvable(path):
    run = next(ap.load(path))
    run.record["extra"]["agentpile"]["labels"] = [
        {"name": "suspect", "target": {"kind": "span", "address": {"trajectory": run.id, "span": key}}, "value": True,
         "label_source": "t"} for key in list(run.spans)[:2]]
    doc = otlp.to_document(run)
    written = {s["spanId"] for s in spans_of(doc)}
    assert set(run.spans) <= written  # the source spans keep their own ids
    assert len({s["traceId"] for s in spans_of(doc)}) == 1 and spans_of(doc)[0]["traceId"] == run.id
    back = next(otlp.read(doc))
    assert back == run
    for label in back.labels:
        assert label["target"]["address"]["span"] in back.spans


def test_pydantic_ai_attributes_the_spec_names_map_as_documented():
    run = next(ap.load(PAI))
    execute = [e for e in run.spans.values() if e["attributes"]["gen_ai.operation.name"] == "execute_tool"]
    assert {e["attributes"]["gen_ai.tool.call.id"] for e in execute} == {"call_weather_1", "call_time_1"}
    cx = run.steps[1]["extra"]["agentpile"]["calls"]
    tool_span_keys = {k for k, e in run.spans.items() if e["attributes"]["gen_ai.operation.name"] == "execute_tool"}
    assert {c["span_id"] for c in cx.values()} == tool_span_keys
    assert run.steps[1]["extra"]["agentpile"]["result_refs"] == [cx["call_weather_1"]["span_id"], cx["call_time_1"]["span_id"]]
    assert run.steps[1]["extra"]["agentpile"]["result_fields"] == [{"name": "get_weather"}, {"name": "get_time"}]
    assert [e.kind for e in run.events() if e.kind == "result"] == ["result", "result"]


# -- synthetic framework documents ------------------------------------------------------------------------

def make_doc(spans, scope=("pydantic-ai", "2.58.0"), trace="aa" * 16):
    """A minimal OTLP document; each span is (id, parent, name, attrs, start) with times in microseconds."""
    return {"resourceSpans": [{"resource": {"attributes": otlp.encode_attributes({"service.name": "t"})},
                               "scopeSpans": [{"scope": {"name": scope[0], "version": scope[1]}, "spans": [
                                   {"traceId": trace, "spanId": sid, **({"parentSpanId": parent} if parent else {}), "name": name,
                                    "kind": 1, "startTimeUnixNano": str(start * 1000), "endTimeUnixNano": str(start * 1000 + 500),
                                    "attributes": otlp.encode_attributes(attrs)}
                                   for sid, parent, name, attrs, start in spans]}]}]}


def msgs(*messages):
    return json.dumps(list(messages))


def part_text(role, text):
    return {"role": role, "parts": [{"type": "text", "content": text}]}


def delegation_doc():
    call = {"type": "tool_call", "id": "d1", "name": "ask_helper", "arguments": {"q": "2+2"}}
    first = [part_text("user", "Compute")]
    return make_doc([
        ("0000000000000001", None, "invoke_agent boss", {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "boss"}, 0),
        ("0000000000000002", "0000000000000001", "chat m",
         {"gen_ai.operation.name": "chat", "gen_ai.agent.name": "boss", "gen_ai.request.model": "m",
          "gen_ai.input.messages": msgs(*first), "gen_ai.output.messages": msgs({"role": "assistant", "parts": [call]})}, 10),
        ("0000000000000003", "0000000000000001", "execute_tool ask_helper",
         {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "ask_helper", "gen_ai.tool.call.id": "d1",
          "gen_ai.tool.call.result": "4"}, 20),
        ("0000000000000004", "0000000000000003", "invoke_agent helper",
         {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "helper"}, 21),
        ("0000000000000005", "0000000000000004", "chat m",
         {"gen_ai.operation.name": "chat", "gen_ai.agent.name": "helper", "gen_ai.request.model": "m",
          "gen_ai.input.messages": msgs(part_text("user", "2+2?")), "gen_ai.output.messages": msgs(part_text("assistant", "4"))}, 22),
        ("0000000000000006", "0000000000000001", "chat m",
         {"gen_ai.operation.name": "chat", "gen_ai.agent.name": "boss", "gen_ai.request.model": "m",
          "gen_ai.input.messages": msgs(*first, {"role": "assistant", "parts": [call]},
                                        {"role": "user", "parts": [{"type": "tool_call_response", "id": "d1", "result": "4"}]}),
          "gen_ai.output.messages": msgs(part_text("assistant", "It is 4"))}, 30),
    ])


def test_an_agent_run_nested_under_a_tool_is_a_subagent_trajectory():
    run = next(otlp.read(delegation_doc()))
    assert ap.validate(run) == []
    harbor_valid(run.record)
    assert [s["message"] for s in run.steps] == ["Compute", "", "It is 4"]
    call_step = run.steps[1]
    assert call_step["observation"]["results"] == [
        {"source_call_id": "d1", "content": "4", "subagent_trajectory_ref": [{"trajectory_id": "0000000000000004"}]}]
    sub = run.record["subagent_trajectories"][0]
    assert sub["trajectory_id"] == "0000000000000004" and sub["agent"]["name"] == "helper"
    assert [s["message"] for s in sub["steps"]] == ["2+2?", "4"]
    assert run.spans["0000000000000005"]["address"] == {"trajectory": "0000000000000004", "step": 2}
    assert run.spans["0000000000000003"]["address"] == {"trajectory": run.id, "step": 2, "call": "d1"}
    assert run.spans["0000000000000006"]["address"] == {"trajectory": run.id, "step": 3}
    assert write_read(run) == run
    doc = otlp.to_document(run)
    sub_span = [s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "subagent"][0]
    call_span = [s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "call"][0]
    assert sub_span["parentSpanId"] == call_span["spanId"]  # nested under the call that referenced it


def test_a_result_echoed_by_a_later_input_pairs_by_id_when_no_tool_span_ran():
    doc = delegation_doc()
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    doc["resourceSpans"][0]["scopeSpans"][0]["spans"] = [s for s in spans if s["spanId"] not in ("0000000000000003", "0000000000000004", "0000000000000005")]
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert run.steps[1]["observation"]["results"] == [{"source_call_id": "d1", "content": "4"}]
    assert run.steps[1]["extra"]["agentpile"]["result_refs"] == ["0000000000000006"]


def test_a_tool_span_with_an_unknown_call_id_is_never_matched_by_name_or_position():
    doc = delegation_doc()
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        for kv in s["attributes"]:
            if kv["key"] == "gen_ai.tool.call.id":
                kv["value"] = {"stringValue": "other"}  # the tool span now names no call of this run
            if kv["key"] == "gen_ai.input.messages" and "tool_call_response" in kv["value"]["stringValue"]:
                messages = json.loads(kv["value"]["stringValue"])
                kv["value"]["stringValue"] = json.dumps(messages[:2])  # and no later input echoes d1's result
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert "observation" not in run.steps[1]  # call d1 has the same name as the tool span, yet gets no result
    stray = run.steps[2]["observation"]["results"]
    assert stray == [{"source_call_id": None, "content": "4"}]  # kept unpaired on the latest agent step
    assert run.steps[2]["extra"]["agentpile"]["result_fields"] == [{"name": "ask_helper", "tool_call_id": "other"}]


def test_an_unsupported_dialect_is_read_generically_and_says_what_it_lost():
    doc = make_doc([
        ("0000000000000001", None, "workflow", {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "x"}, 0),
        ("0000000000000002", "0000000000000001", "chat", {"gen_ai.operation.name": "chat", "gen_ai.input.messages": "[]"}, 5),
    ], scope=("some-other-instrumentation", "1.0"))
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert len(run.steps) == 1 and run.steps[0]["extra"]["agentpile"] == {"message_state": "missing", "empty_source": True}
    assert set(run.spans) == {"0000000000000001", "0000000000000002"} and all(e["address"] is None for e in run.spans.values())
    assert run.source["dialect"] == "generic"
    lost = run.source["lost_fields"]
    assert any(f.startswith("steps:") for f in lost) and any(f.startswith("tool_calls:") for f in lost)
    assert any(f.startswith("observation:") for f in lost)
    assert write_read(run) == run


def test_two_supported_dialects_in_one_trace_are_read_generically():
    oi = make_doc([("0000000000000009", None, "a", {"openinference.span.kind": "LLM"}, 1)],
                  scope=("openinference.instrumentation.openai_agents", "2.5.3"))
    both = delegation_doc()
    both["resourceSpans"] += oi["resourceSpans"]
    both["resourceSpans"][1]["scopeSpans"][0]["spans"][0]["traceId"] = "aa" * 16
    run = next(otlp.read(both))
    assert run.source["dialect"] == "generic" and "two supported dialects" in run.source["lost_fields"][0]


def test_unmapped_message_parts_are_listed():
    doc = delegation_doc()
    for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        for kv in s["attributes"]:
            if kv["key"] == "gen_ai.output.messages" and "It is 4" in kv["value"]["stringValue"]:
                kv["value"]["stringValue"] = json.dumps([{"role": "assistant", "parts": [
                    {"type": "text", "content": "It is 4"}, {"type": "blob", "content": "AAAA", "modality": "image"}]}])
    run = next(otlp.read(doc))
    assert any("blob" in f for f in run.source["lost_fields"])
    assert run.steps[2]["message"] == "It is 4"


def test_repeated_history_is_not_repeated_in_steps_and_new_user_turns_are_kept():
    first = [part_text("user", "hello")]
    reply = part_text("assistant", "hi")
    doc = make_doc([
        ("0000000000000001", None, "invoke_agent a", {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "a"}, 0),
        ("0000000000000002", "0000000000000001", "chat",
         {"gen_ai.operation.name": "chat", "gen_ai.input.messages": msgs(*first), "gen_ai.output.messages": msgs(reply)}, 1),
        ("0000000000000003", "0000000000000001", "chat",
         {"gen_ai.operation.name": "chat", "gen_ai.input.messages": msgs(*first, reply, part_text("user", "hello"), part_text("user", "again")),
          "gen_ai.output.messages": msgs(part_text("assistant", "yes"))}, 2),
    ])
    run = next(otlp.read(doc))
    assert [(s["source"], s["message"]) for s in run.steps] == [
        ("user", "hello"), ("agent", "hi"), ("user", "hello"), ("user", "again"), ("agent", "yes")]


def test_a_two_part_text_message_keeps_both_parts_and_gen_ai_semconv_response_key_is_accepted():
    doc = make_doc([
        ("0000000000000001", None, "chat",
         {"gen_ai.operation.name": "chat",
          "gen_ai.input.messages": msgs({"role": "user", "parts": [{"type": "text", "content": "a"}, {"type": "text", "content": "b"}]}),
          "gen_ai.output.messages": msgs({"role": "assistant", "parts": [
              {"type": "reasoning", "content": "think"}, {"type": "tool_call", "id": "k", "name": "f", "arguments": "{bad"}]})}, 1),
        ("0000000000000002", None, "chat",
         {"gen_ai.operation.name": "chat",
          "gen_ai.input.messages": msgs({"role": "user", "parts": [{"type": "text", "content": "a"}, {"type": "text", "content": "b"}]},
                                         {"role": "assistant", "parts": [{"type": "reasoning", "content": "think"},
                                                                         {"type": "tool_call", "id": "k", "name": "f", "arguments": "{bad"}]},
                                         {"role": "tool", "parts": [{"type": "tool_call_response", "id": "k", "response": {"ok": 1}}]}),
          "gen_ai.output.messages": msgs(part_text("assistant", "done"))}, 2),
    ])
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert run.steps[0]["message"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    agent = run.steps[1]
    assert agent["reasoning_content"] == "think" and agent["tool_calls"][0]["arguments"] == {}
    assert agent["extra"]["agentpile"]["calls"]["k"]["arguments_raw"] == "{bad"
    assert agent["observation"]["results"] == [{"source_call_id": "k", "content": '{"ok": 1}'}]
    assert agent["extra"]["agentpile"]["result_fields"] == [{"content_raw": {"ok": 1}}]
    assert write_read(run) == run


def test_history_no_span_produced_becomes_an_agent_step_with_its_result():
    call = {"type": "tool_call", "id": "k1", "name": "lookup", "arguments": {"x": 1}}
    history = [part_text("user", "earlier question"), {"role": "assistant", "parts": [{"type": "text", "content": "looking"}, call]},
               {"role": "user", "parts": [{"type": "tool_call_response", "id": "k1", "result": "found"}]},
               part_text("user", "next question")]
    doc = make_doc([("0000000000000001", None, "chat",
                     {"gen_ai.operation.name": "chat", "gen_ai.input.messages": msgs(*history),
                      "gen_ai.output.messages": msgs(part_text("assistant", "answer"))}, 1)])
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert [(s["source"], s["message"]) for s in run.steps] == [
        ("user", "earlier question"), ("agent", "looking"), ("user", "next question"), ("agent", "answer")]
    past = run.steps[1]
    assert "timestamp" not in past and "actor" not in past["extra"]["agentpile"]  # no span of this trace produced it
    assert past["tool_calls"][0]["arguments"] == {"x": 1}
    assert past["observation"]["results"] == [{"source_call_id": "k1", "content": "found"}]
    assert run.spans["0000000000000001"]["address"] == {"trajectory": run.id, "step": 4}
    assert write_read(run) == run


def test_a_supported_trace_with_no_model_call_gets_one_placeholder_step():
    doc = make_doc([("0000000000000001", None, "invoke_agent a", {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "a"}, 0)])
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert len(run.steps) == 1 and run.steps[0]["extra"]["agentpile"]["empty_source"] is True
    assert any("no model call span" in f for f in run.source["lost_fields"])
    assert [a["id"] for a in run.agents] == ["a"]


def oi_messages(prefix, messages):
    attrs = {}
    for i, m in enumerate(messages):
        base = f"{prefix}.{i}.message."
        attrs[base + "role"] = m["role"]
        if "content" in m:
            attrs[base + "content"] = m["content"]
        for j, (kind, text) in enumerate(m.get("contents", [])):
            attrs[f"{base}contents.{j}.message_content.type"] = kind
            attrs[f"{base}contents.{j}.message_content.text"] = text
        if "tool_call_id" in m:
            attrs[base + "tool_call_id"] = m["tool_call_id"]
        for j, c in enumerate(m.get("tool_calls", [])):
            attrs[f"{base}tool_calls.{j}.tool_call.function.name"] = c["name"]
            if "id" in c:
                attrs[f"{base}tool_calls.{j}.tool_call.id"] = c["id"]
            if "args" in c:
                attrs[f"{base}tool_calls.{j}.tool_call.function.arguments"] = c["args"]
    return attrs


def test_openinference_message_shapes_are_mapped_by_id_and_odd_ones_are_kept_raw():
    OI = ("openinference.instrumentation.openai_agents", "2.5.3")
    out_calls = [{"id": "x1", "name": "f", "args": '{"k": 1}'}, {"name": "g", "args": "[1]"}]
    first_out = [{"role": "assistant", "contents": [("text", "a"), ("text", "b"), ("reasoning", "hmm")], "tool_calls": out_calls}]
    first_in = [{"role": "developer", "content": "Be brief."}, {"role": "user", "content": "go"}]
    attrs1 = {"openinference.span.kind": "LLM", "llm.model_name": "m1", **oi_messages("llm.input_messages", first_in),
              **oi_messages("llm.output_messages", first_out), "llm.input_messages.0.message.function_call_name": "legacy"}
    again_in = first_in + [{"role": "assistant", "contents": [("text", "a"), ("text", "b")]},
                           {"role": "assistant", "tool_calls": [out_calls[0]]}, {"role": "assistant", "tool_calls": [out_calls[1]]},
                           {"role": "tool", "tool_call_id": "x1", "content": "R1"}]
    attrs2 = {"openinference.span.kind": "LLM", "llm.model_name": "m1", **oi_messages("llm.input_messages", again_in),
              **oi_messages("llm.output_messages", [{"role": "assistant", "content": "done"}])}
    doc = make_doc([
        ("0000000000000001", None, "bot", {"openinference.span.kind": "AGENT", "agent.name": "bot", "session.id": "sess-1"}, 0),
        ("0000000000000002", "0000000000000001", "response", attrs1, 1),
        ("0000000000000003", "0000000000000001", "response", attrs2, 2),
    ], scope=OI)
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    harbor_valid(run.record)
    assert run.record["session_id"] == "sess-1"
    assert [(s["source"], s["extra"]["agentpile"].get("role")) for s in run.steps[:2]] == [("system", "developer"), ("user", None)]
    agent = run.steps[2]
    assert agent["message"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert agent["reasoning_content"] == "hmm"
    first, second = agent["tool_calls"]
    assert (first["tool_call_id"], first["arguments"]) == ("x1", {"k": 1})
    assert second["tool_call_id"] == "ap-3-1" and second["arguments"] == {}  # no id in the source: one is generated, never guessed
    cx = agent["extra"]["agentpile"]["calls"]
    assert cx["ap-3-1"] == {"id_generated": True, "arguments_raw": "[1]", "arguments_state": "unparsed",
                            "source_ref": "0000000000000002"}
    assert agent["observation"]["results"] == [{"source_call_id": "x1", "content": "R1"}]
    assert run.steps[3]["message"] == "done" and len(run.steps) == 4
    assert any("function_call_name" in f for f in run.source["lost_fields"])
    assert write_read(run) == run


def test_synthetic_call_times_stay_between_their_step_and_the_next():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    spans = spans_of(otlp.to_document(run))
    steps = sorted((s for s in spans if attrs_of(s)["agentpile.kind"] == "step"), key=lambda s: attrs_of(s)["agentpile.index"])
    for call in (s for s in spans if attrs_of(s)["agentpile.kind"] == "call"):
        i = attrs_of(call)["agentpile.step_index"]
        assert int(steps[i]["startTimeUnixNano"]) < int(call["startTimeUnixNano"])
        if i + 1 < len(steps):
            assert int(call["startTimeUnixNano"]) < int(steps[i + 1]["startTimeUnixNano"])
        assert attrs_of(call)["agentpile.time_synthetic"] is True


def subagent_record():
    """A valid ATIF record with an embedded subagent that a tool result references, and one nobody references."""
    parent_ext = {"version": "0", "source": {"format": "synthetic"}, "agents": [], "tools": {}}
    return {
        "schema_version": "ATIF-v1.7", "session_id": "p", "trajectory_id": "p", "agent": {"name": "parent", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "delegate"},
            {"step_id": 2, "source": "agent", "message": "asking",
             "tool_calls": [{"tool_call_id": "t1", "function_name": "ask", "arguments": {}}],
             "observation": {"results": [{"source_call_id": "t1", "content": "answered",
                                          "subagent_trajectory_ref": [{"trajectory_id": "child"}]}]}},
            {"step_id": 3, "source": "agent", "message": "thanks"}],
        "subagent_trajectories": [
            {"schema_version": "ATIF-v1.7", "session_id": "p", "trajectory_id": "child", "agent": {"name": "child", "version": "1"},
             "steps": [{"step_id": 1, "source": "agent", "message": "42", "timestamp": "2026-02-02T02:02:02Z"}]},
            {"schema_version": "ATIF-v1.7", "trajectory_id": "orphan", "agent": {"name": "orphan", "version": "1"},
             "steps": [{"step_id": 1, "source": "agent", "message": "unused"}]}],
        "extra": {"agentpile": parent_ext}}


def test_subagent_trajectories_nest_under_the_call_that_references_them():
    run = ap.Run(subagent_record())
    assert ap.validate(run) == []
    harbor_valid(run.record)
    doc = otlp.to_document(run)
    spans = {s["spanId"]: s for s in spans_of(doc)}
    subs = [s for s in spans.values() if attrs_of(s)["agentpile.kind"] == "subagent"]
    assert [s["name"] for s in subs] == ["invoke_agent child", "invoke_agent orphan"]
    assert spans[subs[0]["parentSpanId"]]["name"] == "execute_tool ask"
    assert spans[subs[1]["parentSpanId"]]["name"].startswith("invoke_agent parent")  # unreferenced: under the trajectory span
    kinds = {s["name"]: attrs_of(s)["openinference.span.kind"] for s in spans.values()}
    assert kinds["invoke_agent child"] == "AGENT" and kinds["execute_tool ask"] == "TOOL"
    child_step = [s for s in spans.values() if s.get("parentSpanId") == subs[0]["spanId"]][0]
    assert int(child_step["startTimeUnixNano"]) == otlp.parse_iso_ns("2026-02-02T02:02:02Z")
    assert write_read(run) == run


def random_json(rng, depth=0):
    kinds = ["str", "int", "float", "bool", "none", "list", "dict"] if depth < 3 else ["str", "int", "float", "bool", "none"]
    kind = rng.choice(kinds)
    if kind == "str":
        pool = ["", " ", "plain", "quote\" and \\ backslash", "line\nbreak\ttab", "\u2028\u2029", "emoji \U0001F600", "\u4e2d\u6587",
                "a.b.c", "agentpile.step.message", "null", "{}", "[1]", "\x00nul", "caf\u00e9"]
        return rng.choice(pool)
    if kind == "int":
        return rng.choice([0, -1, 1, 2 ** 31, -2 ** 63, 2 ** 63 - 1, 2 ** 64, 10 ** 30])
    if kind == "float":
        return rng.choice([0.0, -0.0, 1.5, -2.25e-10, 1e300, 0.1])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "none":
        return None
    if kind == "list":
        return [random_json(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(["k", "a.b", "", "agentpile.ext", "x y", "\u00e9"]) + str(i): random_json(rng, depth + 1)
            for i in range(rng.randint(0, 4))}


def test_random_json_values_survive_in_every_free_form_field():
    import random

    rng = random.Random(7)
    for _ in range(40):
        record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
        ext = record["extra"]["agentpile"]
        ext["dataset"] = random_json(rng)
        ext["outcome"]["raw"] = random_json(rng)
        ext["labels"][0]["value"] = random_json(rng)
        ext["labels"][0]["source_ref"] = random_json(rng)
        record["agent"]["extra"] = {"free": random_json(rng)}
        record["final_metrics"] = {"extra": {"free": random_json(rng)}}
        step = record["steps"][2]
        step["extra"]["agentpile"]["source_fields"] = {"free": random_json(rng)}
        step["extra"]["agentpile"]["source_ref"] = random_json(rng)
        step["extra"]["other"] = random_json(rng)
        step["observation"]["results"][0]["extra"] = {"free": random_json(rng)}
        step["tool_calls"][0]["arguments"] = {"free": random_json(rng)}
        step["tool_calls"][0]["extra"] = {"free": random_json(rng)}
        step["metrics"] = {"prompt_tokens": 1, "extra": {"free": random_json(rng)}}
        step["extra"]["agentpile"]["calls"]["c1"]["error"] = random_json(rng)
        record["steps"][1]["message"] = "text with \u0000 and a lone \ud800 half"
        run = ap.Run(record)
        assert write_read(run) == run
        assert next(ap.load(json.loads(run.to_otlp()))) == run


def test_a_model_call_span_without_content_says_so():
    doc = make_doc([("0000000000000001", None, "chat", {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m"}, 1)])
    run = next(otlp.read(doc))
    assert ap.validate(run) == []
    assert run.steps[0]["extra"]["agentpile"]["message_state"] == "missing" and run.steps[0]["model_name"] == "m"
    assert any("absent on a model call span" in f for f in run.source["lost_fields"])
    oi = make_doc([("0000000000000001", None, "response",
                    {"openinference.span.kind": "LLM", "llm.input_messages.0.message.contents.0.message_content.type": "image",
                     "llm.input_messages.1.message.contents.0.message_content.type": "image"}, 1)],
                  scope=("openinference.instrumentation.openai_agents", "2.5.3"))
    notes = next(otlp.read(oi)).source["lost_fields"]
    assert len([f for f in notes if "message_content.type='image'" in f]) == 1  # one note, not one per message index


def test_plain_atif_without_any_agentpile_fields_round_trips_without_gaining_any():
    record = subagent_record()
    del record["extra"]
    for sub in record["subagent_trajectories"]:
        sub["steps"][0]["extra"] = {"harbor_note": 1}
    run = ap.Run(record)
    back = write_read(run)
    assert back == run and "extra" not in back.record
    assert back.record["subagent_trajectories"][0]["steps"][0]["extra"] == {"harbor_note": 1}
    harbor_valid(back.record)


PAI_DELEGATION = OTLP / "pydantic_ai_2.58.0_delegation.json"
OAI_AS_TOOL = OTLP / "openai_agents_0.24.0_oi_2.5.3_as_tool.json"


def test_real_pydantic_ai_delegation_is_a_referenced_subagent():
    run = next(ap.load(PAI_DELEGATION))
    assert ap.validate(run) == []
    harbor_valid(run.record)
    assert len(run.spans) == 6
    sub = run.record["subagent_trajectories"][0]
    assert sub["agent"]["name"] == "helper_agent" and [s["message"] for s in sub["steps"]] == ["answer?", "42"]
    result = run.steps[1]["observation"]["results"][0]
    assert result["source_call_id"] == "call_ask_1" and result["content"] == "42"
    assert result["subagent_trajectory_ref"] == [{"trajectory_id": sub["trajectory_id"]}]
    assert [a["id"] for a in run.agents] == ["boss_agent"] and [a["id"] for a in run.subagents[0].agents] == ["helper_agent"]
    helper_span = run.spans[sub["trajectory_id"]]
    assert helper_span["attributes"]["gen_ai.operation.name"] == "invoke_agent" and helper_span["address"] is None
    helper_chat = run.spans[sub["steps"][1]["extra"]["agentpile"]["source_ref"]]
    assert helper_chat["address"] == {"trajectory": sub["trajectory_id"], "step": 2}
    assert write_read(run) == run


def test_real_openai_agents_as_tool_is_an_unreferenced_subagent():
    run = next(ap.load(OAI_AS_TOOL))
    assert ap.validate(run) == []
    harbor_valid(run.record)
    assert [s["message"] for s in run.steps] == ["Delegate.", "What is the answer?", "", "The answer is 42."]
    assert run.steps[2]["observation"]["results"] == [{"source_call_id": "call_ask_1", "content": "42"}]  # paired by id, no ref
    sub = run.record["subagent_trajectories"][0]
    assert sub["agent"]["name"] == "Helper" and [s["message"] for s in sub["steps"]] == ["Help.", "answer?", "42"]
    # the TOOL span names no call id, so nothing ties the subagent to the call
    assert all("subagent_trajectory_ref" not in r for st in run.steps for r in (st.get("observation") or {}).get("results", []))
    assert write_read(run) == run


# -- a trace read from OTLP is written without repeating its operations -----------------------------------------

FRAMEWORK_FILES = [PAI, PAI_DELEGATION, OAI, OAI_AS_TOOL]
FRAMEWORK_IDS = ["pydantic-ai", "pydantic-ai-delegation", "openai-agents", "openai-agents-as-tool"]
CHAT_OPS = ("chat", "generate_content", "text_completion")
TOKEN_KEYS = ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "llm.token_count.prompt",
              "llm.token_count.completion", "llm.token_count.total")


def source_spans(path, trace_id):
    """The spans of one trace exactly as the framework exported them."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.suffix == ".jsonl" else [path.read_text(encoding="utf-8")]
    return [s for line in lines if line.strip() for s in spans_of(json.loads(line)) if s["traceId"] == trace_id]


def operation_counts(spans):
    """Operations the way a telemetry backend counts them: by GenAI operation name, else by OpenInference span kind."""
    counts = Counter()
    for s in spans:
        a = attrs_of(s)
        op, kind = a.get("gen_ai.operation.name"), a.get("openinference.span.kind")
        if op in CHAT_OPS or kind == "LLM":
            counts["chat"] += 1
        elif op == "execute_tool" or kind == "TOOL":
            counts["tool"] += 1
        elif op == "invoke_agent" or kind == "AGENT":
            counts["agent"] += 1
    return counts


def token_totals(spans):
    totals = Counter()
    for s in spans:
        a = attrs_of(s)
        for key in TOKEN_KEYS:
            if isinstance(a.get(key), int):
                totals[key] += a[key]
    return totals


@pytest.mark.parametrize("path", FRAMEWORK_FILES, ids=FRAMEWORK_IDS)
def test_exporting_a_framework_trace_repeats_none_of_its_operations(path):
    for run in ap.load(path):
        source = source_spans(path, run.id)
        doc = otlp.to_document(run)
        written = spans_of(doc)
        ids = [s["spanId"] for s in written]
        assert len(set(ids)) == len(ids)  # every span has its own identity
        counts = operation_counts(source)
        assert counts["chat"] >= 1 and operation_counts(written) == counts  # chat, tool and agent operations as in the source
        assert token_totals(source) and token_totals(written) == token_totals(source)
        by_id = {s["spanId"]: s for s in written}
        assert all(s["parentSpanId"] in by_id for s in written if "parentSpanId" in s)
        roots = [s for s in written if "parentSpanId" not in s]
        assert len(roots) == 1 and attrs_of(roots[0])["agentpile.kind"] == "trajectory"
        assert {s["spanId"] for s in source} <= set(ids)
        assert sum(1 for s in written if "agentpile.span_key" in attrs_of(s)) == len(source)  # each source span, once
        for s in source:  # unchanged: its own id, name, kind, times, attributes and parent (the trajectory span for a root)
            w = by_id[s["spanId"]]
            assert w["name"] == s["name"] and otlp._kind_int(w["kind"]) == otlp._kind_int(s["kind"])
            assert (int(w["startTimeUnixNano"]), int(w["endTimeUnixNano"])) == (int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"]))
            assert {k: v for k, v in attrs_of(w).items() if not k.startswith("agentpile.")} == attrs_of(s)
            assert w.get("parentSpanId") == s.get("parentSpanId", roots[0]["spanId"])
        back = write_read(run)
        assert back == run and keep(back) == keep(run)  # the agentpile round trip is still exact
        assert list(back.spans) == list(run.spans)  # in the order the source gave the spans
        assert otlp.to_document(back) == doc  # and writing what was read back changes nothing


def test_a_source_span_is_written_as_the_step_or_call_it_became():
    run = next(ap.load(PAI))
    written = {s["spanId"]: s for s in spans_of(otlp.to_document(run))}
    seen = Counter()
    for key, entry in run.spans.items():
        a = attrs_of(written[key])
        address = entry["address"]
        if address is None:
            assert a["agentpile.kind"] == "span"  # the invoke_agent span: neither a step nor a call
        elif "call" in address:
            seen["call"] += 1
            assert a["agentpile.kind"] == "call" and a["gen_ai.tool.call.id"] == address["call"]
            assert a["agentpile.step_index"] == address["step"] - 1
        else:
            seen["step"] += 1
            assert a["agentpile.kind"] == "step" and a["agentpile.index"] == address["step"] - 1
    assert seen == {"step": 2, "call": 2}


def test_a_subagent_trajectory_is_written_as_the_agent_span_it_came_from():
    run = next(ap.load(PAI_DELEGATION))
    sub_id = run.record["subagent_trajectories"][0]["trajectory_id"]
    written = {s["spanId"]: s for s in spans_of(otlp.to_document(run))}
    helper = written[sub_id]
    assert attrs_of(helper)["agentpile.kind"] == "subagent" and attrs_of(helper)["gen_ai.agent.name"] == "helper_agent"
    assert written[helper["parentSpanId"]]["name"] == "execute_tool ask_helper"  # still below the call that delegated
    chats = [s for s in written.values() if s["name"] == "chat fake-boss"]
    tool = written[helper["parentSpanId"]]
    assert len({s["parentSpanId"] for s in chats} | {tool["parentSpanId"]}) == 1  # chat and tool spans share the agent span
    assert written[tool["parentSpanId"]]["name"] == "invoke_agent boss_agent"


@pytest.mark.parametrize("path", FRAMEWORK_FILES, ids=FRAMEWORK_IDS)
def test_what_no_source_span_stands_behind_is_carried_by_spans_that_are_not_operations(path):
    for run in ap.load(path):
        added = [s for s in spans_of(otlp.to_document(run)) if "agentpile.span_key" not in attrs_of(s)]
        assert added
        for s in added:
            a = attrs_of(s)
            assert a["openinference.span.kind"] == "CHAIN" and "gen_ai.operation.name" not in a
            assert not set(TOKEN_KEYS) & set(a) and not any(k.startswith(("llm.", "gen_ai.tool.")) for k in a)
            assert s["name"] in ("user_message", "system_message") or s["name"].startswith(("trajectory", "tool_call", "agent_step"))


def test_a_call_no_source_span_ran_is_not_exported_as_a_tool_operation():
    run = next(ap.load(OAI))  # its two TOOL spans name no call id, so neither is paired with a call
    written = spans_of(otlp.to_document(run))
    assert sorted(s["name"] for s in written if s["name"].startswith("tool_call ")) == [
        "tool_call lookup_order", "tool_call transfer_to_refunds"]
    assert operation_counts(written)["tool"] == 2  # the TOOL spans of the source
    run = next(ap.load(PAI))  # here the tool spans carry the call ids
    assert not [s for s in spans_of(otlp.to_document(run)) if s["name"].startswith("tool_call ")]


def test_history_a_model_call_re_sent_is_not_exported_as_a_second_model_call():
    call = {"type": "tool_call", "id": "k1", "name": "lookup", "arguments": {"x": 1}}
    history = [part_text("user", "earlier question"), {"role": "assistant", "parts": [{"type": "text", "content": "looking"}, call]},
               {"role": "user", "parts": [{"type": "tool_call_response", "id": "k1", "result": "found"}]},
               part_text("user", "next question")]
    doc = make_doc([("0000000000000001", None, "chat",
                     {"gen_ai.operation.name": "chat", "gen_ai.input.messages": msgs(*history),
                      "gen_ai.output.messages": msgs(part_text("assistant", "answer"))}, 1)])
    run = next(otlp.read(doc))
    out = otlp.to_document(run)
    assert operation_counts(spans_of(out)) == operation_counts(spans_of(doc)) == Counter(chat=1)
    assert [s["name"] for s in sorted(spans_of(out), key=lambda s: attrs_of(s).get("agentpile.index", 0))
            if attrs_of(s).get("agentpile.kind") == "step"] == ["user_message", "agent_step", "user_message", "chat"]
    assert len(by_name(out, "tool_call lookup")) == 1 and not by_name(out, "execute_tool lookup")
    assert write_read(run) == run


def test_a_kept_span_addressed_to_an_agent_step_is_that_steps_span_and_one_addressed_to_a_user_step_is_not():
    run = trail_like_record()
    doc = otlp.to_document(run)
    mine = [s for s in spans_of(doc) if attrs_of(s).get("agentpile.span_key") == "not-hex"]
    assert len(mine) == 1 and attrs_of(mine[0])["agentpile.kind"] == "step" and attrs_of(mine[0])["agentpile.index"] == 2
    assert mine[0]["name"] == "" and mine[0]["kind"] == 3  # the source span, not a new "chat" span
    assert not by_name(doc, "chat") and not by_name(doc, "execute_tool search_flights") and by_name(doc, "tool_call search_flights")
    assert len({s["spanId"] for s in spans_of(doc)}) == len(spans_of(doc))
    assert write_read(run) == run
    run.record["extra"]["agentpile"]["spans"]["not-hex"]["address"] = {"trajectory": "fx-1", "step": 2}  # step 2 is a user step
    mine = [s for s in spans_of(otlp.to_document(run)) if attrs_of(s).get("agentpile.span_key") == "not-hex"]
    assert len(mine) == 1 and attrs_of(mine[0])["agentpile.kind"] == "span"
    assert write_read(run) == run


def test_a_step_id_that_two_steps_share_is_never_claimed_by_a_source_span():
    record = copy.deepcopy(next(ap.load(PAI)).record)
    record["steps"][2]["step_id"] = 2  # steps 2 and 3 are both step 2 now
    run = ap.Run(record)
    doc = otlp.to_document(run)
    written = spans_of(doc)
    assert len({s["spanId"] for s in written}) == len(written)
    assert Counter(attrs_of(s)["agentpile.kind"] for s in written) == Counter(trajectory=1, step=3, call=2, span=5)
    assert operation_counts(written) == operation_counts(source_spans(PAI, run.id)) == Counter(chat=2, tool=2, agent=1)
    assert write_read(run) == run


def test_a_call_id_that_two_calls_share_is_never_claimed_by_a_source_span():
    record = copy.deepcopy(next(ap.load(PAI)).record)
    record["steps"][1]["tool_calls"][1]["tool_call_id"] = record["steps"][1]["tool_calls"][0]["tool_call_id"]
    run = ap.Run(record)
    written = spans_of(otlp.to_document(run))
    assert Counter(attrs_of(s)["agentpile.kind"] for s in written)["call"] == 2
    assert operation_counts(written) == Counter(chat=2, tool=2, agent=1)  # the tool spans are kept, the calls carry the record
    assert write_read(run) == run


def test_a_kept_entry_that_is_not_an_object_survives_the_round_trip():
    run = trail_like_record()
    spans = run.record["extra"]["agentpile"]["spans"]
    spans.update({"null-entry": None, "number-entry": 5, "list-entry": [1, {"a": None}]})
    doc = otlp.to_document(run)
    assert len({s["spanId"] for s in spans_of(doc)}) == len(spans_of(doc))
    back = write_read(run)
    assert back == run and list(back.spans) == list(spans)


def test_hostile_edits_of_a_run_with_source_spans_still_round_trip_exactly():
    import random

    rng = random.Random(5)
    bases = [r.record for path in FRAMEWORK_FILES for r in ap.load(path)]

    def walk(rec):
        yield rec
        for sub in rec.get("subagent_trajectories") or []:
            yield from walk(sub)

    for _ in range(120):
        record = copy.deepcopy(rng.choice(bases))
        trajs = list(walk(record))
        for _ in range(rng.randint(1, 5)):
            traj = rng.choice(trajs)
            spans = traj.get("extra", {}).get("agentpile", {}).get("spans") or {}
            keys = [k for k in spans if isinstance(spans[k], dict)]
            kind = rng.choice(["parent", "address", "time", "step_id", "drop_step", "call_id", "sub_id", "cycle", "self_parent"])
            if kind == "parent" and keys:
                spans[rng.choice(keys)]["parent"] = rng.choice(keys + [None, "dangling", ""])
            elif kind == "address" and keys:
                tid = rng.choice([t.get("trajectory_id") for t in trajs] + [None, "other"])
                spans[rng.choice(keys)]["address"] = rng.choice([
                    {"trajectory": tid, "step": rng.choice([1, 2, 3, 99, None, "x"])},
                    {"trajectory": tid, "step": rng.choice([2, 3]), "call": rng.choice(["call_time_1", "call_ask_1", "none"])},
                    None, {"trajectory": tid}, {"span": "x"}, {"step": 2, "result": 0}])
            elif kind == "time" and keys:
                entry = spans[rng.choice(keys)]
                entry["start_time_unix_nano"], entry["end_time_unix_nano"] = rng.choice([(None, None), (5, 1), (None, 7)])
            elif kind == "step_id" and traj.get("steps"):
                rng.choice(traj["steps"])["step_id"] = rng.choice([1, 2, 3, "2", None])
            elif kind == "drop_step" and len(traj.get("steps", [])) > 1:
                del traj["steps"][rng.randrange(len(traj["steps"]))]
            elif kind == "call_id":
                calls = [c for s in traj.get("steps", []) for c in s.get("tool_calls", [])]
                if calls:
                    rng.choice(calls)["tool_call_id"] = rng.choice(["x", "call_time_1", None, 3])
            elif kind == "sub_id" and traj.get("subagent_trajectories"):
                rng.choice(traj["subagent_trajectories"])["trajectory_id"] = rng.choice([None, "x", record.get("trajectory_id")])
            elif kind == "cycle" and len(keys) > 1:
                spans[keys[0]]["parent"], spans[keys[1]]["parent"] = keys[1], keys[0]
            elif kind == "self_parent" and keys:
                spans[keys[0]]["parent"] = keys[0]
        run = ap.Run(record)
        doc = otlp.to_document(run)
        ids = [s["spanId"] for s in spans_of(doc)]
        assert len(set(ids)) == len(ids)
        assert write_read(run) == run


# -- tool call arguments the source did not record ------------------------------------------------------------------

def genai_call_run(parts):
    attrs = {"gen_ai.operation.name": "chat", "gen_ai.input.messages": msgs(part_text("user", "go")),
             "gen_ai.output.messages": msgs({"role": "assistant", "parts": parts})}
    return next(otlp.read(make_doc([("0000000000000001", None, "chat", attrs, 1)])))


def test_absent_null_and_empty_arguments_stay_apart_in_the_genai_dialect():
    run = genai_call_run([
        {"type": "tool_call", "id": "absent", "name": "f"},
        {"type": "tool_call", "id": "empty_object", "name": "f", "arguments": {}},
        {"type": "tool_call", "id": "null", "name": "f", "arguments": None},
        {"type": "tool_call", "id": "empty_text", "name": "f", "arguments": "{}"},
        {"type": "tool_call", "id": "given", "name": "f", "arguments": {"k": 1}}])
    assert ap.validate(run) == []
    step = run.steps[1]
    assert [c["arguments"] for c in step["tool_calls"]] == [{}, {}, {}, {}, {"k": 1}]  # ATIF needs an object
    cxs = step["extra"]["agentpile"]["calls"]
    assert cxs["absent"] == {"arguments_state": "missing", "source_ref": "0000000000000001"}  # no rule: absent stays absent
    assert cxs["null"] == {"arguments_raw": None, "source_ref": "0000000000000001"}
    for name in ("empty_object", "empty_text", "given"):  # recorded values carry no mark
        assert not {"arguments_state", "arguments_stand_in", "arguments_raw"} & set(cxs[name])
    assert "pydantic-ai/genai" not in otlp.OMITS_EMPTY_ARGUMENTS
    assert write_read(run) == run and keep(write_read(run)) == keep(run)
    harbor_valid(run.record)  # last: it skips when Harbor is not installed


def test_openinference_arguments_it_omits_are_rebuilt_as_an_empty_object_and_say_which_rule():
    oi = ("openinference.instrumentation.openai_agents", "2.5.3")
    calls = [{"id": "x1", "name": "f"}, {"id": "x2", "name": "g", "args": "{}"}, {"id": "x3", "name": "h", "args": '{"k": 1}'}]
    attrs = {"openinference.span.kind": "LLM", **oi_messages("llm.input_messages", [{"role": "user", "content": "go"}]),
             **oi_messages("llm.output_messages", [{"role": "assistant", "tool_calls": calls}])}
    run = next(otlp.read(make_doc([("0000000000000001", None, "response", attrs, 1)], scope=oi)))
    assert ap.validate(run) == []
    cxs = run.steps[1]["extra"]["agentpile"]["calls"]
    assert [c["arguments"] for c in run.steps[1]["tool_calls"]] == [{}, {}, {"k": 1}]
    rule = otlp.OMITS_EMPTY_ARGUMENTS["openinference/openai-agents"]
    assert cxs["x1"]["arguments_stand_in"] == rule and "'{}'" in rule and "openinference" in rule and "arguments_state" not in cxs["x1"]
    for recorded in ("x2", "x3"):  # arguments the source did record carry no mark, even the empty object
        assert not {"arguments_state", "arguments_stand_in", "arguments_raw"} & set(cxs.get(recorded, {}))
    assert write_read(run) == run


def test_the_view_shows_no_arguments_the_source_never_recorded():
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    step = record["steps"][2]  # calls c1 (search_flights) and c2 (read_calendar)
    for call in step["tool_calls"]:
        call["arguments"] = {}
    calls = step["extra"]["agentpile"].setdefault("calls", {})
    calls.setdefault("c1", {})["arguments_state"] = "missing"
    calls.setdefault("c2", {})["arguments_raw"] = None  # how every reader stores an explicit null
    assert ap.validate(record) == []
    run = ap.Run(record)
    doc = otlp.to_document(run)
    first, second = (attrs_of(by_name(doc, f"execute_tool {n}")[0]) for n in ("search_flights", "read_calendar"))
    assert "gen_ai.tool.call.arguments" not in first and "input.value" not in first  # absent in the source: absent in the view
    assert second["gen_ai.tool.call.arguments"] == "null"
    other = attrs_of(by_name(doc, "execute_tool book_flight")[0])
    assert other["gen_ai.tool.call.arguments"] == "{'offer_id': 7"  # recorded arguments still show
    chat = attrs_of([s for s in spans_of(doc) if s["name"] == "chat" and attrs_of(s)["agentpile.index"] == 2][0])
    parts = json.loads(chat["gen_ai.output.messages"])[0]["parts"]
    assert "arguments" not in parts[1] and parts[2]["arguments"] is None
    assert not any(k.endswith("tool_calls.0.tool_call.function.arguments") for k in chat)
    assert chat["llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments"] == "null"
    assert write_read(run) == run


# One call per kind of argument: (call id, the arguments ATIF holds, the call extension).
ARGUMENT_CASES = [
    ("parsed", {"k": 1}, {}),                                                   # recorded as an object
    ("empty", {}, {}),                                                          # a genuine empty object
    ("text", {}, {"arguments_raw": "{bad", "arguments_state": "unparsed"}),     # text that is no JSON object: {} stands in
    ("list", {}, {"arguments_raw": ["x", 1], "arguments_state": "unparsed"}),   # a JSON value that is no object
    ("browser", {"arg0": "7"}, {"arguments_raw": "fill('7')"}),                 # raw kept beside a faithful parse
    ("null", {}, {"arguments_raw": None}),                                      # an explicit null
    ("absent", {}, {"arguments_state": "missing"}),                             # no arguments member at all
    ("legacy", {}, {"arguments_raw": "{bad"}),                                  # written before the "unparsed" mark existed
]


def arguments_run(cases=ARGUMENT_CASES):
    """A run whose one agent step makes the calls of ``cases``."""
    record = copy.deepcopy(next(ap.load(FIX / "atif_multiagent.json")).record)
    step = record["steps"][2]
    step["tool_calls"] = [{"tool_call_id": cid, "function_name": "f_" + cid, "arguments": args} for cid, args, _ in cases]
    del step["observation"]
    sx = step["extra"]["agentpile"]
    del sx["result_refs"]
    sx["calls"] = {cid: cx for cid, _, cx in cases if cx}
    return ap.Run(record)


def argument_views(run, cases=ARGUMENT_CASES):
    """call id -> (model span in gen_ai.output.messages, model span in llm.output_messages, tool span); MISSING when absent."""
    doc = otlp.to_document(run)
    chat = attrs_of([s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "step" and attrs_of(s)["agentpile.index"] == 2][0])
    parts = {p["id"]: p for p in json.loads(chat["gen_ai.output.messages"])[0]["parts"] if p["type"] == "tool_call"}
    out = {}
    for j, (cid, _, _) in enumerate(cases):
        tool = attrs_of(by_name(doc, "execute_tool f_" + cid)[0])
        key = "llm.output_messages.0.message.tool_calls.%d.tool_call.function.arguments" % j
        out[cid] = (parts[cid].get("arguments", MISSING), chat.get(key, MISSING), tool.get("gen_ai.tool.call.arguments", MISSING))
    return out


def test_a_model_span_shows_the_source_arguments_exactly_when_atif_holds_a_stand_in():
    run = arguments_run()
    assert ap.validate(run) == []
    views = argument_views(run)
    # an object the source recorded: the model span shows it, the tool span shows its text
    assert views["parsed"][0] == {"k": 1} and json.loads(views["parsed"][1]) == {"k": 1} and json.loads(views["parsed"][2]) == {"k": 1}
    assert views["empty"] == ({}, "{}", "{}")  # a genuine empty object stays an empty object
    # a stand-in {}: both spans show the source value, and the llm.* text is the source text itself, not a quoted copy
    assert views["text"] == ("{bad", "{bad", "{bad")
    assert views["list"][0] == ["x", 1] and json.loads(views["list"][1]) == ["x", 1] and json.loads(views["list"][2]) == ["x", 1]
    # raw kept beside a faithful parse (AgentRewardBench): the model span keeps the parsed object, only the tool span shows the raw text
    assert views["browser"][0] == {"arg0": "7"} and json.loads(views["browser"][1]) == {"arg0": "7"} and views["browser"][2] == "fill('7')"
    assert views["null"] == (None, "null", "null")
    assert views["absent"] == (MISSING, MISSING, MISSING)
    assert views["legacy"] == ({}, "{}", "{bad")  # a record written before the mark exports as it did
    assert write_read(run) == run and keep(write_read(run)) == keep(run)
    harbor_valid(run.record)  # last: it skips when Harbor is not installed


def test_the_fixture_call_with_malformed_arguments_shows_its_text_on_both_spans():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    doc = otlp.to_document(run)
    chat = attrs_of([s for s in spans_of(doc) if attrs_of(s).get("agentpile.kind") == "step" and attrs_of(s)["agentpile.index"] == 3][0])
    (part,) = [p for p in json.loads(chat["gen_ai.output.messages"])[0]["parts"] if p["type"] == "tool_call"]
    assert part["arguments"] == "{'offer_id': 7"
    assert chat["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"] == "{'offer_id': 7"
    assert attrs_of(by_name(doc, "execute_tool book_flight")[0])["gen_ai.tool.call.arguments"] == "{'offer_id': 7"
    assert [c["arguments"] for c in run.steps[3]["tool_calls"]] == [{}]  # the record still holds the stand-in
    legacy = copy.deepcopy(run.record)
    del legacy["steps"][3]["extra"]["agentpile"]["calls"]["c3"]["arguments_state"]
    (old,) = [p for p in json.loads(attrs_of([s for s in spans_of(otlp.to_document(ap.Run(legacy)))
                                              if attrs_of(s).get("agentpile.kind") == "step"
                                              and attrs_of(s)["agentpile.index"] == 3][0])["gen_ai.output.messages"])[0]["parts"]
              if p["type"] == "tool_call"]
    assert old["arguments"] == {}  # without the mark the model span shows what ATIF holds, as before


def test_the_model_span_form_of_each_call_reads_back_as_the_same_call_in_both_dialects():
    run = arguments_run()
    chat = attrs_of([s for s in spans_of(otlp.to_document(run)) if attrs_of(s).get("agentpile.kind") == "step"
                     and attrs_of(s)["agentpile.index"] == 2][0])
    parts = json.loads(chat["gen_ai.output.messages"])[0]["parts"]
    back = genai_call_run([p for p in parts if p["type"] == "tool_call"])  # the view alone, as a collector would keep it
    assert ap.validate(back) == []
    step = back.steps[1]
    cxs = step["extra"]["agentpile"]["calls"]
    held = {c["tool_call_id"]: c["arguments"] for c in step["tool_calls"]}
    assert held == {"parsed": {"k": 1}, "empty": {}, "text": {}, "list": {}, "browser": {"arg0": "7"}, "null": {},
                    "absent": {}, "legacy": {}}
    assert {k: cxs.get(k, {}).get("arguments_state") for k in held} == {
        "parsed": None, "empty": None, "text": "unparsed", "list": "unparsed", "browser": None, "null": None,
        "absent": "missing", "legacy": None}
    assert cxs["text"]["arguments_raw"] == "{bad" and cxs["list"]["arguments_raw"] == ["x", 1] and cxs["null"]["arguments_raw"] is None
    assert write_read(back) == back
    # the OpenInference form: arguments are text, so a source string comes back as the same string
    oi = ("openinference.instrumentation.openai_agents", "2.5.3")
    calls = [{"id": cid, "name": "f_" + cid, "args": chat["llm.output_messages.0.message.tool_calls.%d.tool_call.function.arguments" % j]}
             for j, (cid, _, _) in enumerate(ARGUMENT_CASES) if cid in ("parsed", "text", "browser")]
    attrs = {"openinference.span.kind": "LLM", **oi_messages("llm.input_messages", [{"role": "user", "content": "go"}]),
             **oi_messages("llm.output_messages", [{"role": "assistant", "tool_calls": calls}])}
    oi_run = next(otlp.read(make_doc([("0000000000000001", None, "response", attrs, 1)], scope=oi)))
    oi_cx = oi_run.steps[1]["extra"]["agentpile"]["calls"]
    assert oi_cx["text"] == {"arguments_raw": "{bad", "arguments_state": "unparsed", "source_ref": "0000000000000001"}
    assert [c["arguments"] for c in oi_run.steps[1]["tool_calls"]] == [{"k": 1}, {}, {"arg0": "7"}]
    assert "arguments_state" not in oi_cx.get("parsed", {}) and "arguments_state" not in oi_cx.get("browser", {})


# -- AnyValue, ids and document shapes -------------------------------------------------------------------

def test_anyvalue_forms_and_numbers():
    decode = otlp.decode_value
    assert decode({"stringValue": "x"}) == "x" and decode({"stringValue": ""}) == ""
    assert decode({"boolValue": False}) is False
    assert decode({"intValue": "42"}) == 42 and decode({"intValue": 7}) == 7 and decode({"intValue": "-9223372036854775808"}) == -2 ** 63
    assert decode({"doubleValue": 1.5}) == 1.5 and decode({"doubleValue": "Infinity"}) == float("inf")
    assert decode({"doubleValue": "NaN"}) != decode({"doubleValue": "NaN"})
    assert decode({"arrayValue": {"values": [{"intValue": "1"}, {"stringValue": "a"}, {}]}}) == [1, "a", None]
    assert decode({"arrayValue": {}}) == []
    assert decode({"kvlistValue": {"values": [{"key": "k", "value": {"boolValue": True}}]}}) == {"k": True}
    assert decode({"bytesValue": "AAEC"}) == {"$bytes": "AAEC"}
    assert decode({}) is None and decode({"string_value": "snake"}) == "snake"
    for value in ("", "x", 0, -1, 2 ** 62, 1.25, True, False, None, [], {}, [1, [2, {"a": None}]], {"a": {"b": [True]}},
                  {"$bytes": "AAEC"}):
        assert decode(json.loads(json.dumps(otlp.encode_value(value)))) == value
    assert otlp.encode_value(2 ** 70) == {"stringValue": str(2 ** 70)}
    assert otlp.encode_value(1) == {"intValue": "1"} and otlp.encode_value(True) == {"boolValue": True}


def spans_doc(**span_extra):
    span = {"traceId": "0102030405060708090a0b0c0d0e0f10", "spanId": "1112131415161718", "name": "s", "kind": 1,
            "startTimeUnixNano": "1000", "endTimeUnixNano": 2000,
            "attributes": [{"key": "k", "value": {"intValue": "3"}}]}
    span.update(span_extra)
    return {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"scope": {"name": "x"}, "spans": [span]}]}]}


def test_ids_enums_numbers_events_and_links_are_normalized():
    doc = spans_doc(parentSpanId="AQIDBAUGBwg=", status={"code": "STATUS_CODE_ERROR", "message": "bad"}, kind="SPAN_KIND_CLIENT",
                    events=[{"timeUnixNano": "1500", "name": "e", "attributes": [{"key": "a", "value": {"stringValue": "b"}}],
                             "droppedAttributesCount": 2}],
                    links=[{"traceId": "AQIDBAUGBwgJCgsMDQ4PEA==", "spanId": "1112131415161718", "attributes": [], "flags": 257}])
    run = next(otlp.read(doc))
    (key, entry), = run.spans.items()
    assert key == "1112131415161718" and entry["trace_id"] == "0102030405060708090a0b0c0d0e0f10"
    assert entry["parent"] == "0102030405060708" and entry["kind"] == "CLIENT"
    assert entry["status"] == {"code": "ERROR", "message": "bad"}
    assert entry["start_time_unix_nano"] == 1000 and entry["end_time_unix_nano"] == 2000 and entry["attributes"] == {"k": 3}
    assert entry["events"] == [{"name": "e", "time_unix_nano": 1500, "attributes": {"a": "b"}, "dropped_attributes_count": 2}]
    assert entry["links"][0]["trace_id"] == "0102030405060708090a0b0c0d0e0f10" and entry["links"][0]["flags"] == 257
    assert write_read(run) == run


def test_snake_case_and_legacy_spellings_and_missing_fields():
    doc = {"resource_spans": [{"resource": {"attributes": [{"key": "service.name", "value": {"string_value": "svc"}}]},
                               "instrumentation_library_spans": [{"instrumentation_library": {"name": "lib", "version": "1"},
                                                                   "spans": [{"trace_id": "aa" * 16, "span_id": "bb" * 8, "name": "n",
                                                                              "start_time_unix_nano": "5"}]}]}]}
    assert otlp.sniff(doc)
    run = next(otlp.read(doc))
    entry = run.spans["bb" * 8]
    assert entry["kind"] == "UNSPECIFIED" and entry["status"] is None and entry["end_time_unix_nano"] is None
    assert entry["resource"] == {"service.name": "svc"} and entry["scope"] == {"name": "lib", "version": "1"}
    assert entry["parent"] is None and entry["events"] == [] and entry["links"] == []


def test_runs_come_one_per_trace_id_in_first_seen_order_across_documents():
    def doc(trace, sid):
        return spans_doc(traceId=trace, spanId=sid)

    docs = [doc("11" * 16, "01" * 8), doc("22" * 16, "02" * 8), doc("11" * 16, "03" * 8)]
    runs = list(otlp.read(docs))
    assert [r.id for r in runs] == ["11" * 16, "22" * 16]
    assert list(runs[0].spans) == ["01" * 8, "03" * 8] and list(runs[1].spans) == ["02" * 8]
    assert list(otlp.read({"resourceSpans": []})) == []


def test_a_span_id_seen_twice_keeps_both_spans():
    docs = [spans_doc(), spans_doc(name="again")]
    run = next(otlp.read(docs))
    assert list(run.spans) == ["1112131415161718", "1112131415161718#2"]


def test_file_shapes(tmp_path):
    single = spans_doc()
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(single, indent=2), encoding="utf-8")
    lines = tmp_path / "lines.jsonl"
    lines.write_text(json.dumps(single) + "\n\n" + json.dumps(spans_doc(spanId="2" * 16)) + "\n", encoding="utf-8")
    jsonl_in_json = tmp_path / "lines_as.json"
    jsonl_in_json.write_text(lines.read_text(encoding="utf-8"), encoding="utf-8")
    multi_line_doc = tmp_path / "pretty.jsonl"
    multi_line_doc.write_text(json.dumps(single, indent=2), encoding="utf-8")
    zipped = tmp_path / "z.jsonl.gz"
    zipped.write_bytes(gzip.compress(lines.read_bytes()))
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([single, spans_doc(spanId="2" * 16)]), encoding="utf-8")
    for path, n in ((pretty, 1), (lines, 2), (jsonl_in_json, 2), (multi_line_doc, 1), (zipped, 2), (listed, 2)):
        assert otlp.sniff(path) and len(next(ap.load(path)).spans) == n, path.name
    junk = tmp_path / "junk.json"
    junk.write_text("{not json", encoding="utf-8")
    assert not otlp.sniff(junk) and not otlp.sniff(tmp_path / "missing.json") and not otlp.sniff({"messages": []})
    with pytest.raises(ValueError):
        list(otlp.read(junk))


def test_reader_options_are_not_swallowed():
    with pytest.raises(TypeError, match="takes no options except allow_untested_version"):
        list(ap.load(PAI, typo_option=1))
    assert list(ap.load(PAI, allow_untested_version=False)) == list(ap.load(PAI))
    assert list(ap.load(PAI, allow_untested_version=True)) == list(ap.load(PAI))  # a tested trace is unmarked either way


# -- (d) the format sniffers stay unambiguous --------------------------------------------------------------------

def fixture_files():
    # tests/fixtures/grade holds raw source records for the GRADE compatibility tests, which no format reader claims
    return sorted(p for p in FIX.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl") and p.name != "versions.json"
                  and "grade" not in p.relative_to(FIX).parts)


@pytest.mark.parametrize("path", fixture_files(), ids=lambda p: str(p.relative_to(FIX)))
def test_every_fixture_file_is_recognized_by_exactly_one_reader(path):
    if path.name == "not_a_trace.json":
        with pytest.raises(ap.UnrecognizedSource):
            ap.load(path)
        return
    matches = [n for n in ap.readers.names() if ap.readers.get(n).sniff(path)]
    assert len(matches) == 1, matches
    assert matches == (["otlp-json"] if path.parent == OTLP else matches)
    assert list(ap.load(path))  # and the loader accepts it without naming a reader


def test_other_shapes_are_not_taken_for_otlp():
    atif = json.loads((FIX / "atif_multiagent.json").read_text())
    assert not otlp.sniff(atif) and not otlp.sniff([atif]) and not otlp.sniff({"messages": [{"role": "user", "content": "x"}]})
    assert not otlp.sniff({"resourceSpans": {}}) and not otlp.sniff(None) and not otlp.sniff("text")
    assert otlp.sniff({"resourceSpans": []}) and otlp.sniff([{"resourceSpans": []}])
    assert [n for n in ap.readers.names() if ap.readers.get(n).sniff(spans_doc())] == ["otlp-json"]
    assert next(ap.load(spans_doc())).source["format"] == "otlp-json"


def test_reading_and_writing_otlp_needs_only_the_standard_library():
    script = f"""
import json
import sys
import agentpile as ap
for path in (r"{PAI}", r"{OAI}"):
    for run in ap.load(path):
        assert ap.validate(run) == []
        assert next(ap.load(json.loads(run.to_otlp()))) == run
third_party = sorted(m for m in ("opentelemetry", "google", "pydantic", "yaml", "numpy", "pandas", "pyarrow", "huggingface_hub")
                     if m in sys.modules)
print(",".join(third_party))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_the_writer_and_reader_are_registered_built_ins():
    assert "otlp-json" in ap.readers and "otlp-json" in ap.writers
    assert ap.readers.get("otlp-json") is otlp and ap.writers.get("otlp-json") is otlp
    run = next(ap.load(FIX / "atif_multiagent.json"))
    assert run.to_otlp() == otlp.dumps(run)
