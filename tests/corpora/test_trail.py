"""TRAIL loader on a synthetic copy of the pinned file layout (no dataset content).

The fixture is built by fixtures/trail/make_fixture.py: five Parquet rows (three
GAIA, two SWE Bench) written by hand in the real trace and annotation schema.
"""

import calendar
import os
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import agentpile as ap  # noqa: E402
from agentpile.corpora.trail import _duration_ns, _iso_ns, _loads_tolerant  # noqa: E402

ROOT = Path(__file__).parent / "fixtures" / "trail"
G1, G2, G3 = ("gaia/0000000000000000000000000000aaa1", "gaia/0000000000000000000000000000aaa2", "gaia/row-2")
S1, S2 = "swe_bench/0000000000000000000000000000bbb1", "swe_bench/0000000000000000000000000000bbb2"
T0 = calendar.timegm((2025, 1, 1, 12, 0, 0)) * 10**9  # BASE of the fixture generator


def _runs(**kw):
    return {r.id: r for r in ap.load("trail", root=str(ROOT), **kw)}


def _step(run, span_id, part="output_message"):
    """The step read from the given span (the output step by default)."""
    for step in run.steps:
        sx = step["extra"]["agentpile"]
        if sx.get("span_id") == span_id and sx["source_ref"]["part"] == part:
            return step
    raise AssertionError(f"no {part} step from span {span_id}")


def test_rows_become_runs_in_split_order():
    runs = _runs()
    assert list(runs) == [G1, G2, G3, S1, S2]
    assert all(ap.validate(r) == [] for r in runs.values())
    assert list(_runs(split="swe_bench")) == [S1, S2]
    with pytest.raises(ValueError, match="split must be one of"):
        _runs(split="train")
    with pytest.raises(TypeError, match="split= only"):
        _runs(subset="gaia")
    src = runs[G1].source
    assert src["dataset"] == "PatronusAI/TRAIL" and src["revision"] == ap.datasets.get("trail").revision
    assert src["file"] == "data/gaia-00000-of-00001-33a2e72d362d688a.parquet" and src["split"] == "gaia"
    assert src["config"] == "default" and src["record_key"] == G1 and src["reader"] == "trail"
    assert runs[G2].dataset["row"] == 1 and runs[S2].source["file"].startswith("data/swe_bench-")


def test_unparseable_row_is_an_empty_run():
    run = _runs()[G3]
    assert len(run.steps) == 1 and run.steps[0]["extra"]["agentpile"]["empty_source"] is True
    assert run.dataset["trace_text"] == "this row is not JSON" and run.dataset["trace_id"] is None
    assert run.dataset["labels_parse"] == "missing" and run.labels == [] and run.spans == {}


def test_every_span_goes_into_the_span_map():
    run = _runs()[G1]
    spans = run.spans
    assert len(spans) == 21
    kinds = {}
    for s in spans.values():
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    assert kinds == {None: 3, "AGENT": 2, "LLM": 9, "CHAIN": 5, "TOOL": 2}
    s = spans["a000000000000007"]
    assert s["name"] == "LiteLLMModel.__call__" and s["kind"] == "LLM" and s["parent"] == "a000000000000006"
    assert s["trace_id"] == "0000000000000000000000000000aaa1"
    assert s["start_time_unix_nano"] == T0 + 1_000_000_000 and s["end_time_unix_nano"] == T0 + 1_500_000_000
    assert s["status"] == {"code": "Ok", "message": ""} and s["events"] == [] and s["links"] == []
    assert s["attributes"]["llm.model_name"] == "synth-model" and "llm.input_messages.3.message.content" in s["attributes"]
    assert s["source_fields"]["duration"] == "PT0.5S" and s["source_fields"]["timestamp"].endswith("01.000000Z")
    assert s["source_fields"]["scope_name"] == "openinference.instrumentation.smolagents"
    assert s["address"] == {"trajectory": G1, "step": 7}
    wrapper = spans["a000000000000002"]
    assert wrapper["kind"] is None and wrapper["address"] is None and wrapper["name"] == "answer_single_question"
    example = wrapper["source_fields"]["logs"][0]["body"]["function.arguments"]["example"]
    assert example["task_id"] == "synthetic-task-1"


def test_failed_llm_span_keeps_status_and_events_but_makes_no_step():
    run = _runs()[G1]
    failed = run.spans["a000000000000018"]
    assert failed["kind"] == "LLM" and failed["address"] is None
    assert failed["status"] == {"code": "Error", "message": "BadRequestError: synthetic failure"}
    (event,) = failed["events"]
    assert event["name"] == "exception" and event["attributes"]["exception.type"] == "BadRequestError"
    assert event["timestamp"] == "2025-01-01T12:00:08.200000"  # no zone in the source, read as UTC
    assert failed["start_time_unix_nano"] <= event["time_unix_nano"] <= failed["end_time_unix_nano"]
    assert not any(s["extra"]["agentpile"].get("span_id") == "a000000000000018" for s in run.steps)
    chain = run.spans["a000000000000019"]
    assert chain["kind"] == "CHAIN" and chain["address"] is None and chain["status"]["code"] == "Error"


def test_only_llm_spans_with_output_have_an_address_and_it_points_back():
    run = _runs()[G1]
    addressed = {sid: s["address"] for sid, s in run.spans.items() if s["address"]}
    assert set(addressed) == {"a000000000000004", "a000000000000005", "a000000000000007", "a000000000000009",
                              "a000000000000011", "a000000000000014", "a000000000000016", "a000000000000020"}
    for sid, address in addressed.items():
        step = run.steps[address["step"] - 1]
        sx = step["extra"]["agentpile"]
        assert address["trajectory"] == G1 and sx["span_id"] == sid and sx["source_ref"]["part"] == "output_message"
        assert run.spans[sid]["kind"] == "LLM"
    for step in run.steps:
        sx = step["extra"]["agentpile"]
        assert sx["span_id"] in run.spans
        for cid, cx in (sx.get("calls") or {}).items():
            assert cx["span_id"] in run.spans


def test_repeated_span_orphan_parent_and_unreadable_time():
    run = _runs()[G2]
    assert len(run.spans) == 6
    assert run.dataset["duplicate_spans"] == {"b000000000000004": {"count": 2, "identical": True}}
    assert run.dataset["spans_with_missing_parent"] == {"b000000000000006": "ffffffffffffffff"}
    assert run.spans["b000000000000006"]["parent"] == "ffffffffffffffff"
    late = run.spans["b000000000000005"]
    assert late["start_time_unix_nano"] is None and late["end_time_unix_nano"] is None
    assert "timestamp" not in _step(run, "b000000000000005")
    assert run.spans["b000000000000001"]["end_time_unix_nano"] - run.spans["b000000000000001"]["start_time_unix_nano"] \
        == (62 * 10**9 + 250_000_000)  # PT1M2.25S
    assert run.spans["b000000000000003"]["end_time_unix_nano"] - run.spans["b000000000000003"]["start_time_unix_nano"] \
        == 3600 * 10**9 + 500_000_000  # PT1H0M0.5S


def test_each_input_message_is_read_once_per_agent_run():
    run = _runs()[G1]
    shape = [(s["source"], s["extra"]["agentpile"].get("actor") or s["extra"]["agentpile"].get("agent"),
              s["message"][:12]) for s in run.steps]
    manager, helper = "a000000000000003", "a000000000000008"
    assert shape == [
        ("user", manager, "Below I will"), ("agent", manager, "### 1. Facts"),
        ("user", manager, "Now make a p"), ("agent", manager, "1. Ask the h"),
        ("system", manager, "You are a sy"), ("user", manager, "New task:\nSy"), ("agent", manager, "Thought: del"),
        ("user", helper, "Below I will"), ("agent", helper, "- helper fac"),
        ("system", helper, "You are the "), ("user", helper, "New task:\nYo"), ("agent", helper, ""),
        ("agent", helper, ""), ("agent", manager, "Thought: fin"),
        ("system", None, "Earlier you "), ("user", None, "Read the con"), ("agent", None, "FINAL ANSWER"),
    ]
    # the final-answer call outside every AGENT span repeats the memory with every role set to user; only its new texts are read
    final = [s for s in run.steps if s["extra"]["agentpile"]["span_id"] == "a000000000000020"]
    assert [s["source"] for s in final] == ["system", "user", "agent"]
    texts = [s["message"] for s in run.steps if s["source"] in ("system", "user")]
    assert len(texts) == len(set(texts))


def test_renderings_of_earlier_outputs_are_copies_not_steps():
    run = _runs()[G1]
    assert not any(s["message"].startswith(("[FACTS LIST]", "[PLAN]")) for s in run.steps)
    facts = _step(run, "a000000000000004")["extra"]["agentpile"]["history_copy"]
    plan = _step(run, "a000000000000005")["extra"]["agentpile"]["history_copy"]  # the output ends with <end_plan>
    assert facts == [{"span_id": "a000000000000007", "index": 2}] and plan == [{"span_id": "a000000000000007", "index": 3}]
    delegating = _step(run, "a000000000000007")["extra"]["agentpile"]["history_copy"]
    assert delegating == [{"span_id": "a000000000000016", "index": 4}]  # the same text plus <end_code>
    assert not any(s["extra"]["agentpile"].get("history_only") for s in run.steps)


def test_history_tool_call_becomes_a_flagged_call_on_the_latest_agent_step():
    run = _runs()[G1]
    step = _step(run, "a000000000000007")
    (call,) = step["tool_calls"]
    assert call == {"tool_call_id": "call_1", "function_name": "python_interpreter", "arguments": {}}
    cx = step["extra"]["agentpile"]["calls"]["call_1"]
    assert cx["arguments_raw"] == "search_agent(task='q')" and cx["arguments_raw"] in step["message"]
    assert cx["call_from"] == "history_message" and cx["paired_by"] == "latest_agent_step"
    assert cx["span_id"] == "a000000000000007" and cx["parent"] == "a000000000000006"
    assert cx["source_ref"] == {"span_id": "a000000000000016", "index": 5, "part": "input_message"}
    assert "tool_id" not in cx  # python_interpreter has no definition in the trace
    ok, unpaired = step["observation"]["results"]
    assert ok["source_call_id"] == "call_1" and ok["content"].startswith("Call id: call_1\nObservation:")
    assert unpaired == {"source_call_id": None, "content": "Error:\nError in code parsing:\nsynthetic parse failure"}
    sx = step["extra"]["agentpile"]
    assert sx["result_refs"] == [{"span_id": "a000000000000016", "index": 6}, {"span_id": "a000000000000016", "index": 7}]
    assert sx["result_fields"] == [{"role": "tool-response"}, {"role": "tool-response"}]


def test_structured_call_keeps_its_id_and_the_history_message_only_refers_to_it():
    run = _runs()[G1]
    step = _step(run, "a000000000000011")
    assert step["message"] == "" and step["extra"]["agentpile"]["message_state"] == "missing"
    assert step["tool_calls"] == [{"tool_call_id": "call_sub1", "function_name": "web_search",
                                   "arguments": {"query": "synthetic fact"}}]
    cx = step["extra"]["agentpile"]["calls"]["call_sub1"]
    assert cx["tool_id"] == "web_search" and cx["span_id"] == "a000000000000011" and "call_from" not in cx
    assert cx["source_ref"] == {"span_id": "a000000000000011", "part": "output_message", "index": 0, "call": 0}
    assert cx["history_refs"] == [{"span_id": "a000000000000014", "index": 3}]
    (result,) = step["observation"]["results"]
    assert result["source_call_id"] == "call_sub1" and result["content"].endswith("synthetic fact is A.")
    last = _step(run, "a000000000000014")
    assert last["tool_calls"][0]["tool_call_id"] == "call_sub2" and "observation" not in last
    assert len([c for s in run.steps for c in s.get("tool_calls") or []]) == 3  # no call is guessed from message text


def test_odd_messages_keep_what_the_source_says():
    run = _runs()[G2]
    by = {s["extra"]["agentpile"]["source_ref"]["index"]: s for s in run.steps
          if s["extra"]["agentpile"]["span_id"] == "b000000000000002"
          and s["extra"]["agentpile"]["source_ref"]["part"] == "input_message"}
    assert by[2]["source"] == "agent" and by[2]["extra"]["agentpile"]["history_only"] is True  # no output carries this text
    assert by[3]["source"] == "user" and by[3]["extra"]["agentpile"]["role"] == "function"
    assert by[4]["extra"]["agentpile"] == {**by[4]["extra"]["agentpile"], "role": None, "role_state": "missing"}
    assert by[5]["extra"]["agentpile"]["role"] == "tool-call" and by[5]["message"] == "Calling tools: not a list"
    # a response with an id no call carries goes unpaired on the latest agent step and keeps the id
    (unpaired,) = by[2]["observation"]["results"]
    assert unpaired["source_call_id"] is None
    assert by[2]["extra"]["agentpile"]["result_fields"] == [{"role": "tool-response", "tool_call_id": "call_x"}]
    out = _step(run, "b000000000000002")
    (call,) = out["tool_calls"]
    assert call == {"tool_call_id": "ap-7-0", "function_name": "mystery", "arguments": {}}
    cx = out["extra"]["agentpile"]["calls"]["ap-7-0"]
    assert cx["id_generated"] is True and cx["arguments_raw"] == "not json" and "source_id" not in cx
    assert "metrics" not in out and "model_name" not in out
    late = [s for s in run.steps if s["extra"]["agentpile"]["span_id"] == "b000000000000005" and s["source"] == "agent"]
    first = late[0]
    assert first["extra"]["agentpile"]["history_only"] is True and first["message"] == "unrelated role-ful text"
    (call9,) = first["tool_calls"]  # attached to the latest agent step of the run, here the history-only step
    assert call9["tool_call_id"] == "call_9" and first["extra"]["agentpile"]["calls"]["call_9"]["arguments_raw"] == "print(1)"
    (null_result,) = first["observation"]["results"]
    assert null_result == {"source_call_id": None, "content": ""}
    assert first["extra"]["agentpile"]["result_fields"] == [{"role": "tool-response", "content_state": "null"}]
    # the role-less repeat of an earlier assistant text is the same message and is not read again
    assert [s["message"] for s in run.steps].count("A1 matches no output") == 1
    assert len(run.steps) == 11


def test_trimmed_window_with_a_roleless_first_message_adds_nothing_new():
    run = _runs()[S1]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "agent"]
    assert not any(s["extra"]["agentpile"].get("role_state") for s in run.steps)
    first, second, third = run.steps[2:]
    assert first["tool_calls"][0]["tool_call_id"] == "call_1" and second["tool_calls"][0]["tool_call_id"] == "call_2"
    assert first["observation"]["results"][0]["source_call_id"] == "call_1"
    assert second["observation"]["results"][0]["source_call_id"] == "call_2"
    assert second["extra"]["agentpile"]["history_copy"] == [{"span_id": "c000000000000005", "index": 3}]
    assert "tool_calls" not in third


def test_group_without_an_agent_span_still_reads_each_message_once():
    run = _runs()[S2]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent"]
    assert run.agents == [] and "actor" not in run.steps[2]["extra"]["agentpile"]
    assert run.dataset["spans_with_missing_parent"] == {"d000000000000001": "eeeeeeeeeeeeeeee",
                                                        "d000000000000002": "eeeeeeeeeeeeeeee"}
    assert run.record["agent"] == {"name": "smolagents", "version": "unknown", "model_name": "synth-model"}
    assert run.labels == [] and run.task == {"id": None, "text": None}


def test_agents_tools_task_and_agent_block():
    runs = _runs()
    g1 = runs[G1]
    assert g1.agents == [
        {"id": "a000000000000003", "name": "CodeAgent.run", "instructions": None, "model": None,
         "tools": ["final_answer", "helper_tool"], "handoffs": []},
        {"id": "a000000000000008", "name": "ToolCallingAgent.run", "instructions": None, "model": None,
         "tools": ["web_search", "final_answer"], "handoffs": []}]
    assert set(g1.tools) == {"web_search", "final_answer", "helper_tool"}
    web = g1.tools["web_search"]
    assert web["definition_format"] == "openai-function-json-schema" and web["definition"]["function"]["name"] == "web_search"
    assert web["permission"] == "unknown" and web["permission_from"] == "unknown" and web["atif_index"] is None
    assert g1.tools["helper_tool"]["definition"] is None and g1.tools["helper_tool"]["definition_format"] is None
    s1 = runs[S1]
    assert s1.tools["final_answer"]["definition_format"] == "smolagents-tool-span"
    assert s1.tools["final_answer"]["definition"]["description"] == "Synthetic final_answer."
    assert g1.record["agent"] == {"name": "smolagents", "version": "unknown", "model_name": "synth-model"}
    assert s1.record["agent"]["model_name"] == "synth-claude"
    assert runs[G2].record["agent"]["model_name"] == "synth-model"  # one LLM span has no model name; the others agree
    assert g1.task == {"id": "synthetic-task-1", "text": "Synthetic task text for the manager."}
    assert runs[G2].task == {"id": None, "text": "Fallback task text."}
    assert s1.task == {"id": "synth__repo-1", "text": "Fix the synthetic issue."}
    assert g1.outcome == {"success": None, "score": None, "raw": None}
    assert [i for i in g1.dataset["instrumentation"]] == [
        {"scope_name": "openinference.instrumentation.smolagents", "scope_version": "0.1.6"}]


def test_metrics_come_from_token_counts():
    runs = _runs()
    assert _step(runs[G1], "a000000000000007")["metrics"] == {"prompt_tokens": 10, "completion_tokens": 5,
                                                              "extra": {"total_tokens": 15}}
    third = runs[S1].steps[-1]
    assert third["metrics"] == {"completion_tokens": 3}  # "many" is not a count and stays in the extension
    assert third["extra"]["agentpile"]["source_fields"] == {"llm.token_count.prompt": "many"}
    assert third["model_name"] == "synth-claude" and third["timestamp"].startswith("2025-01-01T12:00:00.5")


def test_error_labels_sit_on_spans_that_exist():
    runs = _runs()
    for run in runs.values():
        for label in run.labels:
            if label["target"]["kind"] == "span":
                assert label["target"]["address"]["trajectory"] == run.id
                assert label["target"]["address"]["span"] in run.spans
            assert label["label_source"] == "trail"
    labels = runs[G1].labels
    errors = [l for l in labels if l["name"] == "error"]
    spans = [l["target"]["address"]["span"] for l in errors if l["target"]["kind"] == "span"]
    assert spans == ["a000000000000007", "a000000000000012", "a000000000000011", "a000000000000011"]
    first = errors[0]
    assert first["value"] == {"category": "Formatting Errors", "location": "a000000000000007",
                              "evidence": "search_agent(task='q')",
                              "description": "The code block has no final answer call.", "impact": "LOW"}
    assert first["source_ref"] == {"field": "errors", "index": 0, "location": "a000000000000007", "resolved": True}
    # a label on a TOOL span targets that span; the span is not a step
    assert errors[1]["target"] == {"kind": "span", "address": {"trajectory": G1, "span": "a000000000000012"}}
    assert runs[G1].spans["a000000000000012"]["address"] is None
    # the sentinel location names no span: the annotation stays, at run level
    sentinel = errors[4]
    assert sentinel["target"] == {"kind": "run", "address": {"trajectory": G1}}
    assert sentinel["source_ref"]["resolved"] is False and sentinel["value"]["category"] == "Resource Abuse"
    scores = {l["name"]: l["value"] for l in labels if l["name"] != "error"}
    assert scores["overall"] == 3.5 and scores["reliability_score"] == 4 and len(scores) == 9
    assert scores["plan_opt_reasoning"] == "Plan was loose."
    assert all(l["target"] == {"kind": "run", "address": {"trajectory": G1}} for l in labels if l["name"] != "error")


def test_tolerant_label_parse_and_malformed_error_object():
    run = _runs()[G2]
    assert run.dataset["labels_parse"] == "tolerant" and _runs()[G1].dataset["labels_parse"] == "strict"
    first, stray, overall = run.labels
    assert first["target"]["address"]["span"] == "b000000000000002" and first["value"]["evidence"] == "x, y"
    assert stray["target"]["address"]["span"] == "b000000000000003"  # a CHAIN span
    assert stray["value"]["Observation: ..."] == "stray key" and "description" not in stray["value"]
    assert stray["value"]["within the same output."] == "stray text"
    assert overall["name"] == "overall" and overall["value"] == 1
    assert "labels_trace_id" not in run.dataset  # this label file carries no trace_id


def test_events_walk_steps_calls_and_results():
    run = _runs()[G1]
    events = list(run.events())
    kinds = [e.kind for e in events]
    assert kinds.count("call") == 3 and kinds.count("result") == 3 and kinds.count("message") == 15
    call = next(e for e in events if e.call_id == "call_1" and e.kind == "call")
    assert call.content == "search_agent(task='q')" and call.name == "python_interpreter"
    assert call.address == {"trajectory": G1, "step": 7, "call": "call_1"}
    result = next(e for e in events if e.kind == "result" and e.call_id == "call_sub1")
    assert result.address["step"] == 12 and result.source_ref == {"span_id": "a000000000000014", "index": 4}
    assert not any(e.address.get("step") == 12 and e.kind == "message" for e in events)  # the output has no content


def test_helpers_read_times_and_loose_json():
    assert _iso_ns("2025-01-01T12:00:00.000001Z") == T0 + 1000
    assert _iso_ns("2025-01-01T12:00:00") == T0 and _iso_ns("2025-01-01T13:00:00+01:00") == T0
    assert _iso_ns("2025-01-01T12:00:00.123456789123Z") == T0 + 123_456_789
    assert _iso_ns("unknown") is None and _iso_ns(None) is None
    assert _duration_ns("PT9.9S") == 9_900_000_000 and _duration_ns("PT1M2.25S") == 62_250_000_000
    assert _duration_ns("PT1H0M0.5S") == 3_600_500_000_000 and _duration_ns("PT") is None and _duration_ns(5) is None
    text = '{"a": [1, 2,], "b": "x,]", "c": {"d": "q\\",}",},}'
    value, mode = _loads_tolerant(text)
    assert mode == "tolerant" and value == {"a": [1, 2], "b": "x,]", "c": {"d": 'q",}'}}
    assert _loads_tolerant('{"a": 1}') == ({"a": 1}, "strict")
    with pytest.raises(ValueError):
        _loads_tolerant('{"a": }')


def test_atif_part_validates_against_harbor():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs().values():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    local = Path(data) / "trail" if data else None
    root = str(local) if local is not None and local.is_dir() else None  # else the library download
    runs = list(ap.load("trail", root=root, verify=True))
    assert len(runs) == 148 and sum(r.source["split"] == "gaia" for r in runs) == 117
    assert all(ap.validate(r) == [] for r in runs)
    assert sum(len(r.spans) for r in runs) == 4625  # 4,626 spans in the files, one repeated in a SWE Bench trace
    assert sum(len(r.steps) for r in runs) == 3492
    errors = [l for r in runs for l in r.labels if l["name"] == "error"]
    on_spans = [l for l in errors if l["target"]["kind"] == "span"]
    assert len(errors) == 841 and len(on_spans) == 839
    for run in runs:
        assert all(l["target"]["address"]["span"] in run.spans for l in run.labels if l["target"]["kind"] == "span")
    assert sum(r.dataset["labels_parse"] == "tolerant" for r in runs) == 1
    steps = [s for r in runs for s in r.steps]
    calls = [c for s in steps for c in s.get("tool_calls") or []]
    results = [x for s in steps for x in (s.get("observation") or {}).get("results", [])]
    assert len(calls) == 1112 and len(results) == 1092 and sum(x["source_call_id"] is None for x in results) == 8
    from_history = [1 for s in steps for cx in (s["extra"]["agentpile"].get("calls") or {}).values()
                    if cx.get("call_from") == "history_message"]
    assert len(from_history) == 564  # CodeAgent calls; the 548 structured calls come from ToolCallingAgent outputs
    flags = [s["extra"]["agentpile"] for s in steps]
    assert not any(f.get("history_only") or f.get("role_state") for f in flags)  # every assistant message found its output
    addressed = sum(1 for r in runs for s in r.spans.values() if s["address"])
    assert addressed == 1967  # the LLM spans with an output; the 6 failed ones have none
