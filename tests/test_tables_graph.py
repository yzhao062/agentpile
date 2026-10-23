"""Flat tables (``ap.table``, ``ap.to_pandas``) and the trajectory graph (``ap.to_networkx``).

``table`` runs on the standard library alone and is tested on the Quick Start conversation, five dataset
fixtures and a config file. The pandas and networkx functions use ``pytest.importorskip``.
"""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

import agentpile as ap

FIX = Path(__file__).parent / "fixtures"
CORPORA = Path(__file__).parent / "corpora" / "fixtures"

QUICK_START = [
    {"role": "user", "content": "Is it raining in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "light rain, 14 C"},
    {"role": "assistant", "content": "Yes, light rain."},
]

# Every stand-in the record marks: a part ATIF cannot hold, a step with no message, a call with unparsed,
# absent and null arguments, a call with no id, and a result with no content and one with null content.
STAND_INS = [
    {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "u"}}]},
    {"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "not json"}},
        {"type": "function", "function": {"name": "b"}},
        {"id": "c3", "type": "function", "function": {"name": "c", "arguments": None}},
    ]},
    {"role": "tool", "tool_call_id": "c1"},
    {"role": "tool", "tool_call_id": "c3", "content": None},
    {"role": "assistant", "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "d", "arguments": "{}"}}]},
]

# Two edits of one file, and an id from the first result that the second call repeats: every kind of rule has an edge.
EDITING = [
    {"role": "user", "content": "Fix the order."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "e1", "type": "function", "function": {
        "name": "str_replace_editor", "arguments": json.dumps({"command": "create", "path": "/w/a.py"})}}]},
    {"role": "tool", "tool_call_id": "e1", "content": json.dumps({"order_id": "ord-1234"})},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "e2", "type": "function", "function": {
        "name": "str_replace_editor", "arguments": json.dumps({"command": "view", "path": "/w/a.py", "note": "ord-1234"})}}]},
    {"role": "tool", "tool_call_id": "e2", "content": "print(1)"},
    {"role": "assistant", "content": "Done."},
]

SCALARS = (type(None), bool, int, float, str)


def quick_start():
    return next(ap.load(QUICK_START))


def stand_ins():
    return next(ap.load(STAND_INS))


def editing():
    return next(ap.load(EDITING))


def corpus(name, **kw):
    return list(ap.load(name, root=str(CORPORA / name), **kw))


def who_and_when():
    return corpus("who_and_when")


def trail():
    pytest.importorskip("pyarrow")
    return corpus("trail")


def key_of(address):
    """The node key of a record address, written out apart from the code under test."""
    key = f"step:{address['step']}"
    if address.get("call") is not None:
        key += f"/call:{address['call']}"
    if address.get("result") is not None:
        key += f"/result:{address['result']}"
    return key


def every_fixture():
    """Runs from fixtures that need no optional library."""
    runs = [quick_start(), stand_ins(), editing()] + list(ap.load(FIX / "config_mcp.json"))
    for name in ("who_and_when", "adp", "catchbench_pre", "tau_bench", "agentdojo", "mast"):
        runs += corpus(name)
    return runs


# -- table ---------------------------------------------------------------------------------------------------

def test_quick_start_events_in_source_order():
    rows = ap.table(quick_start())
    assert [(r["index"], r["kind"], r["role"], r["step"], r["call"], r["result"], r["source_call_id"], r["name"],
             r["content"], r["content_state"]) for r in rows] == [
        (0, "message", "user", 1, None, None, None, None, "Is it raining in Paris?", None),
        (1, "message", "agent", 2, None, None, None, None, None, "null"),
        (2, "call", "agent", 2, "call_1", None, None, "get_weather", '{"city": "Paris"}', None),
        (3, "result", "tool", 2, None, 0, "call_1", None, "light rain, 14 C", None),
        (4, "message", "agent", 3, None, None, None, None, "Yes, light rain.", None),
    ]
    assert {r["run_id"] for r in rows} == {"conversation-0"}
    assert all(r["error"] is None and not r["id_generated"] and r["paired_by"] is None for r in rows)
    assert list(rows[0]) == ap._tables.EVENT_COLUMNS


def test_quick_start_run_row():
    assert ap.table(quick_start(), "runs") == [{
        "run_id": "conversation-0", "kind": "trajectory", "dataset": None, "config": None, "revision": None,
        "file": None, "format": "openai-messages", "task_id": None, "n_steps": 3, "n_messages": 3, "n_calls": 1,
        "n_results": 1, "n_agents": 0, "n_tools": 0, "n_labels": 0}]
    assert ap.table(quick_start(), "labels") == []


def test_one_run_and_an_iterable_give_the_same_rows():
    run = quick_start()
    assert ap.table(run) == ap.table([run]) == ap.table(iter([run]))
    assert ap.table([run, run], "runs") == ap.table(run, "runs") * 2
    assert ap.table([]) == []


def test_a_generator_from_load_is_consumed_once():
    gen = ap.load(QUICK_START)
    assert len(ap.table(gen, "runs")) == 1
    assert ap.table(gen, "runs") == []


def test_unknown_level_raises_before_the_runs_are_read():
    gen = ap.load(QUICK_START)
    with pytest.raises(ValueError, match="level must be one of 'runs', 'events', 'labels'"):
        ap.table(gen, "steps")
    assert len(ap.table(gen, "runs")) == 1  # the generator was left alone


def test_something_that_is_not_a_run_is_a_type_error():
    with pytest.raises(TypeError, match="Run"):
        ap.table(quick_start().to_dict())
    with pytest.raises(TypeError, match="Run"):
        ap.table([quick_start().to_dict()])
    with pytest.raises(TypeError, match="Run"):
        ap.table(None)


def test_who_and_when_events_carry_the_actor():
    runs = who_and_when()
    rows = ap.table(runs)
    assert len(rows) == sum(len(r.steps) for r in runs) == 15
    first = [r for r in rows if r["run_id"] == "Algorithm-Generated/1"]
    assert [(r["index"], r["step"], r["role"], r["actor"], r["kind"]) for r in first] == [
        (0, 1, "agent", "Math_Expert", "message"), (1, 2, "agent", "Coder_Expert", "message"),
        (2, 3, "agent", "Computer_terminal", "message"), (3, 4, "agent", "Math_Expert", "message"),
        (4, 5, "agent", "Math_Expert", "message")]
    hand = [r for r in rows if r["run_id"] == "Hand-Crafted/1"]
    assert [(r["role"], r["actor"]) for r in hand][:3] == [("user", None), ("agent", "Orchestrator"), ("agent", "Orchestrator")]


def test_who_and_when_run_rows_flatten_the_outcome():
    rows = ap.table(who_and_when(), "runs")
    assert [r["run_id"] for r in rows] == ["Algorithm-Generated/1", "Algorithm-Generated/2", "Hand-Crafted/1"]
    first = rows[0]
    assert first["dataset"] == "Kevin355/Who_and_When" and first["revision"] == ap.datasets.get("who_and_when").revision
    assert first["task_id"] == "q-synth-1" and first["n_steps"] == 5 and first["n_agents"] == 3 and first["n_labels"] == 3
    assert first["n_messages"] == 5 and first["n_calls"] == 0 and first["n_results"] == 0
    assert first["outcome.success"] is False and first["outcome.score"] is None
    assert first["outcome.raw"] == '{"is_correct": false}'
    assert json.loads(rows[2]["outcome.raw"]) == {"is_corrected": False}
    assert [c for c in first if c.startswith("outcome.")] == ["outcome.success", "outcome.score", "outcome.raw"]


def test_who_and_when_labels_say_where_they_point():
    rows = ap.table(who_and_when(), "labels")
    first = [r for r in rows if r["run_id"] == "Algorithm-Generated/1"]
    by_name = {r["name"]: r for r in first}
    assert by_name["mistake_step"]["target_kind"] == "step" and by_name["mistake_step"]["step"] == 5
    assert (by_name["mistake_step"]["value"], by_name["mistake_step"]["value_type"]) == ("true", "bool")
    assert by_name["mistake_step"]["label_source"] == "who_and_when"
    assert by_name["mistake_agent"]["target_kind"] == "agent" and by_name["mistake_agent"]["agent"] == "Math_Expert"
    assert by_name["mistake_agent"]["step"] is None
    hand = {r["name"]: r for r in rows if r["run_id"] == "Hand-Crafted/1"}
    assert hand["mistake_type"]["target_kind"] == "run" and hand["mistake_type"]["value"] == "Tool failure"
    assert hand["mistake_type"]["value_type"] == "str"
    assert hand["mistake_type"]["step"] is None and hand["mistake_type"]["agent"] is None
    assert list(rows[0]) == ap._tables.LABEL_COLUMNS


def test_catchbench_pre_is_config_rows_with_tool_labels_and_no_events():
    runs = corpus("catchbench_pre")
    assert ap.table(runs, "events") == []
    run_rows = ap.table(runs, "runs")
    assert {r["kind"] for r in run_rows} == {"config"}
    assert all(r["n_steps"] == r["n_messages"] == r["n_calls"] == r["n_results"] == 0 for r in run_rows)
    assert run_rows[0]["n_tools"] == len(runs[0].tools) and run_rows[0]["n_tools"] > 0
    assert run_rows[0]["dataset"] == "yzhao062/catchbench" and run_rows[0]["config"] == runs[0].source["config"]
    labels = ap.table(runs, "labels")
    assert len(labels) == sum(len(r.labels) for r in runs)
    tool_labels = [r for r in labels if r["target_kind"] == "tool"]
    assert tool_labels and all(r["tool"] and r["name"] == "excess_capability" and (r["value"], r["value_type"]) == ("true", "bool")
                               for r in tool_labels)
    minimal = next(r for r in labels if r["run_id"] == "crewai-x1" and r["name"] == "minimal_reference")
    assert (minimal["value"], minimal["value_type"]) == ('["WebSearch"]', "json")
    assert minimal["target_kind"] == "run" and minimal["label_source"] == "catchbench"


def test_config_file_has_a_run_row_and_no_events():
    run = next(ap.load(FIX / "config_mcp.json"))
    assert ap.table(run, "events") == []
    (row,) = ap.table(run, "runs")
    assert row["run_id"] == "fx-config-1" and row["kind"] == "config" and row["format"] == "mcp-tools-list"
    assert row["n_steps"] == 0 and row["n_agents"] == 1 and row["n_tools"] == len(run.tools) == 2
    assert not [c for c in row if c.startswith("outcome.")]
    assert ap.table(run, "labels") == [{
        "run_id": "fx-config-1", "name": "excess_permission", "value": "true", "value_type": "bool", "target_kind": "agent",
        "trajectory": "fx-config-1", "step": None, "call": None, "result": None, "span": None, "agent": "assistant",
        "tool": None, "label_source": "fixture"}]


def test_labels_aimed_at_different_trajectories_stay_distinct():
    record = next(ap.load(FIX / "atif_multiagent.json")).to_dict()
    record["subagent_trajectories"] = [{
        "schema_version": "ATIF-v1.7", "session_id": "sub-1", "trajectory_id": "sub-1",
        "agent": {"name": "helper", "version": "0"},
        "steps": [{"step_id": 1, "source": "agent", "message": "sub says hi"}]}]
    record["extra"]["agentpile"]["labels"] = [
        {"name": "blame", "target": {"kind": "step", "address": {"trajectory": tid, "step": 1}}, "value": True,
         "label_source": "fixture"} for tid in ("fx-1", "sub-1")]
    run = ap.Run(record)
    assert ap.validate(run.record) == []
    rows = ap.table(run, "labels")
    assert [(r["run_id"], r["trajectory"], r["step"]) for r in rows] == [("fx-1", "fx-1", 1), ("fx-1", "sub-1", 1)]
    pd = pytest.importorskip("pandas")
    frame = ap.to_pandas(run, "labels")
    assert isinstance(frame, pd.DataFrame) and list(frame.trajectory) == ["fx-1", "sub-1"]


def test_run_rows_of_mixed_kinds_share_their_keys():
    rows = ap.table(who_and_when() + list(ap.load(FIX / "config_mcp.json")) + [quick_start()], "runs")
    assert len({tuple(r) for r in rows}) == 1
    config = rows[3]
    assert config["kind"] == "config" and config["outcome.success"] is None and config["outcome.raw"] is None
    assert rows[4]["outcome.raw"] is None


def test_adp_marks_inferred_pairings_and_generated_ids():
    runs = corpus("adp")
    rows = ap.table(runs)
    paired = [r for r in rows if r["paired_by"]]
    assert paired and {r["paired_by"] for r in paired} == {"adp_sequence"} and {r["kind"] for r in paired} == {"result"}
    assert all(r["source_call_id"] for r in paired)
    generated = [r for r in rows if r["id_generated"]]
    assert generated and all(r["kind"] == "call" and r["call"] for r in generated)
    states = {r["content_state"] for r in rows}
    assert {None, "null", "missing", "unparsed", "raw"} <= states
    assert len(rows) == sum(len(list(r.events())) for r in runs)
    assert sum(r["n_results"] for r in ap.table(runs, "runs")) == sum(r["kind"] == "result" for r in rows)
    assert ap.table(runs, "labels") == []  # ADP has no source labels
    assert {r["config"] for r in ap.table(runs, "runs")} >= {"agenttuning_alfworld", "agenttuning_kg"}


def test_stand_ins_are_marked_in_columns():
    run = stand_ins()
    assert ap.validate(run) == []
    rows = ap.table(run)
    assert [(r["kind"], r["step"], r["call"], r["result"]) for r in rows] == [
        ("message", 1, None, None),  # step 2 and step 3 have no message, so they yield no message event
        ("call", 2, "c1", None), ("call", 2, "ap-2-1", None), ("call", 2, "c3", None),
        ("result", 2, None, 0), ("result", 2, None, 1), ("call", 3, "c4", None)]
    by = {(r["kind"], r["call"] or r["result"]): r for r in rows if r["kind"] != "message"}
    assert rows[0]["content_state"] == "raw" and json.loads(rows[0]["content"])[1]["type"] == "image_url"
    assert by[("call", "c1")]["content_state"] == "unparsed" and by[("call", "c1")]["content"] == "not json"
    assert by[("call", "ap-2-1")]["content_state"] == "missing" and by[("call", "ap-2-1")]["content"] is None
    assert by[("call", "ap-2-1")]["id_generated"] is True and by[("call", "c1")]["id_generated"] is False
    assert by[("call", "c3")]["content_state"] == "null" and by[("call", "c3")]["content"] is None
    assert by[("call", "c4")]["content_state"] is None and by[("call", "c4")]["content"] == "{}"
    assert by[("result", 0)]["content_state"] == "missing" and by[("result", 1)]["content_state"] == "null"
    assert by[("result", 0)]["source_call_id"] == "c1" and by[("result", 1)]["source_call_id"] == "c3"
    (row,) = ap.table(run, "runs")
    assert (row["n_steps"], row["n_messages"], row["n_calls"], row["n_results"]) == (3, 1, 4, 2)


def test_trail_span_labels_and_inferred_pairings():
    runs = trail()
    rows = ap.table(runs)
    assert {r["paired_by"] for r in rows if r["paired_by"]} == {"latest_agent_step", "unique_tool_span"}
    unparsed = [r for r in rows if r["content_state"] == "unparsed"]
    assert unparsed and all(r["kind"] == "call" for r in unparsed)
    labels = ap.table(runs, "labels")
    spans = [r for r in labels if r["target_kind"] == "span"]
    assert spans and all(r["span"] and r["step"] is None for r in spans)
    errors = [r for r in labels if r["name"] == "error"]
    assert errors and all(r["value_type"] == "json" and json.loads(r["value"])["category"] for r in errors)
    scores = [r for r in labels if r["name"] == "reliability_score"]
    assert scores and all(r["value_type"] == "int" and isinstance(json.loads(r["value"]), int) and r["target_kind"] == "run"
                          for r in scores)
    assert len(labels) == sum(len(r.labels) for r in runs)


def test_every_cell_is_a_scalar_and_the_rows_go_to_csv_and_json():
    runs = every_fixture()
    for level in ("runs", "events", "labels"):
        rows = ap.table(runs, level)
        assert rows, level
        assert all(type(v) in SCALARS for row in rows for v in row.values()), level
        assert len({tuple(row) for row in rows}) == 1, level  # one set of keys per level
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        assert len(list(csv.DictReader(io.StringIO(out.getvalue())))) == len(rows)
        assert json.loads(json.dumps(rows)) == rows


def test_table_counts_agree_with_the_runs():
    runs = every_fixture()
    runs_rows = ap.table(runs, "runs")
    assert len(runs_rows) == len(runs)
    for run, row in zip(runs, runs_rows):
        kinds = [e.kind for e in run.events()]
        assert (row["run_id"], row["kind"], row["n_steps"]) == (run.id, run.kind, len(run.steps))
        assert (row["n_messages"], row["n_calls"], row["n_results"]) == (
            kinds.count("message"), kinds.count("call"), kinds.count("result"))
        assert (row["n_agents"], row["n_tools"], row["n_labels"]) == (len(run.agents), len(run.tools), len(run.labels))
    assert len(ap.table(runs, "events")) == sum(r["n_messages"] + r["n_calls"] + r["n_results"] for r in runs_rows)
    assert len(ap.table(runs, "labels")) == sum(r["n_labels"] for r in runs_rows)


def test_event_rows_follow_run_events_exactly():
    run = next(r for r in corpus("adp") if r.id == "agenttuning_alfworld/synth_alf_0")
    rows = ap.table(run)
    for row, event in zip(rows, run.events()):
        assert (row["kind"], row["role"], row["actor"], row["name"]) == (event.kind, event.role, event.actor, event.name)
        assert (row["step"], row["call"], row["result"]) == (
            event.address["step"], event.address.get("call"), event.address.get("result"))
    assert [r["index"] for r in rows] == list(range(len(rows)))


def test_table_imports_no_third_party_module():
    script = (
        "import sys\n"
        "import agentpile as ap\n"
        f"run = next(ap.load({QUICK_START!r}))\n"
        "for level in ('runs', 'events', 'labels'):\n"
        "    ap.table(run, level)\n"
        "print(','.join(m for m in ('pandas', 'networkx', 'numpy') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# -- to_pandas -----------------------------------------------------------------------------------------------

def test_to_pandas_names_the_extra_when_pandas_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match=r"pip install 'agentpile\[pandas\]'"):
        ap.to_pandas(quick_start())


def test_to_pandas_events():
    pd = pytest.importorskip("pandas")
    frame = ap.to_pandas(quick_start())
    assert isinstance(frame, pd.DataFrame) and frame.shape == (5, 16)
    assert list(frame.columns) == ap._tables.EVENT_COLUMNS
    assert list(frame["kind"]) == ["message", "message", "call", "result", "message"]
    assert str(frame["step"].dtype) == "Int64" and str(frame["result"].dtype) == "Int64"
    assert list(frame["step"]) == [1, 2, 2, 2, 3]
    assert frame["result"].isna().tolist() == [True, True, True, False, True]
    assert frame.loc[3, "result"] == 0 and frame.loc[3, "source_call_id"] == "call_1"
    assert frame.loc[2, "content"] == '{"city": "Paris"}'
    assert frame["id_generated"].dtype == bool


def test_to_pandas_levels_match_table():
    pd = pytest.importorskip("pandas")
    runs = who_and_when() + [quick_start()]
    for level in ("runs", "events", "labels"):
        rows = ap.table(runs, level)
        frame = ap.to_pandas(runs, level)
        assert len(frame) == len(rows) and list(frame.columns) == list(rows[0])
        assert frame.astype(object).where(frame.notna(), None).to_dict("records") == rows
    runs_frame = ap.to_pandas(runs, "runs")
    assert str(runs_frame["n_steps"].dtype) == "Int64"
    assert list(runs_frame["outcome.success"].dropna()) == [False, False, False]
    assert runs_frame.groupby("dataset")["n_steps"].sum().to_dict() == {"Kevin355/Who_and_When": 15}


def test_to_pandas_with_no_rows_keeps_the_columns():
    pytest.importorskip("pandas")
    config = list(ap.load(FIX / "config_mcp.json"))
    events = ap.to_pandas(config, "events")
    assert len(events) == 0 and list(events.columns) == ap._tables.EVENT_COLUMNS
    assert str(events["step"].dtype) == "Int64"
    labels = ap.to_pandas([], "labels")
    assert len(labels) == 0 and list(labels.columns) == ap._tables.LABEL_COLUMNS
    runs = ap.to_pandas(config, "runs")
    assert len(runs) == 1 and runs.loc[0, "kind"] == "config" and runs.loc[0, "n_tools"] == 2


def test_every_level_of_every_dataset_reaches_arrow_and_parquet(tmp_path):
    # Datasets label with booleans, numbers, strings and objects alike; one mixed column used to break Arrow here.
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("pandas")
    by_dataset = {d.name: corpus(d.name) for d in sorted(CORPORA.iterdir()) if d.is_dir()}
    assert len(by_dataset) == 13
    everything = [run for runs in by_dataset.values() for run in runs]
    for name, runs in [*by_dataset.items(), ("all", everything)]:
        for level in ap._tables.LEVELS:
            rows = ap.table(runs, level)
            pa.Table.from_pylist(rows)
            pq.write_table(pa.Table.from_pandas(ap.to_pandas(runs, level)), tmp_path / "t.parquet")
            assert pq.read_table(tmp_path / "t.parquet").num_rows == len(rows), (name, level)
    rows = ap.table(everything, "labels")
    values = [label.get("value") for run in everything for label in run.labels]
    assert len(rows) == len(values) and {r["value_type"] for r in rows} == {"bool", "int", "float", "str", "json", None}
    tags = {type(None): None, bool: "bool", int: "int", float: "float", str: "str"}
    frame = ap.to_pandas(everything, "labels")
    cells = list(zip(frame["value"], frame["value_type"]))
    assert len(cells) == len(rows)
    for row, cell, value in zip(rows, cells, values):  # the value comes back with its own type: True, not 1
        assert row["value_type"] == tags.get(type(value), "json"), row
        for text, kind in ((row["value"], row["value_type"]), cell):  # the same rule reads a table row and a DataFrame cell
            back = text if kind in ("str", None) else json.loads(text)
            assert type(back) is type(value) and back == value, row


def test_to_pandas_rejects_an_unknown_level():
    pytest.importorskip("pandas")
    with pytest.raises(ValueError, match="level must be one of"):
        ap.to_pandas(quick_start(), "steps")


# -- to_networkx ---------------------------------------------------------------------------------------------

def test_to_networkx_rejects_a_config_record_with_or_without_networkx():
    run = next(ap.load(FIX / "config_mcp.json"))
    with pytest.raises(ValueError, match="config record has no steps"):
        ap.to_networkx(run)


def test_to_networkx_names_the_extra_when_networkx_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "networkx", None)
    with pytest.raises(ImportError, match=r"pip install 'agentpile\[graph\]'"):
        ap.to_networkx(quick_start())


def test_to_networkx_needs_a_run():
    with pytest.raises(TypeError, match="Run"):
        ap.to_networkx(quick_start().to_dict())


def test_quick_start_graph_nodes_and_edges():
    nx = pytest.importorskip("networkx")
    graph = ap.to_networkx(quick_start())
    assert isinstance(graph, nx.MultiDiGraph)
    assert list(graph.nodes) == ["step:1", "step:2", "step:2/call:call_1", "step:2/result:0", "step:3"]
    assert graph.graph == {"run_id": "conversation-0", "labels": []}
    node = graph.nodes["step:2/call:call_1"]
    assert (node["kind"], node["role"], node["actor"], node["name"], node["step"]) == ("call", "agent", None, "get_weather", 2)
    assert node["content"] == '{"city": "Paris"}' and node["labels"] == []
    assert graph.nodes["step:2"]["content"] is None and graph.nodes["step:3"]["content"] == "Yes, light rain."
    assert graph.nodes["step:2/result:0"]["role"] == "tool"
    edges = sorted(graph.edges(keys=True, data=True))
    assert [(u, v, k, d["kind"]) for u, v, k, d in edges] == sorted([
        ("step:1", "step:2", "next", "next"), ("step:2", "step:2/call:call_1", "next", "next"),
        ("step:2/call:call_1", "step:2/result:0", "next", "next"), ("step:2/result:0", "step:3", "next", "next"),
        ("step:2/result:0", "step:2/call:call_1", "result_of", "result_of")])
    assert all("paired_by" not in d for _, _, _, d in edges)  # the source gave the call id: nothing is inferred


def test_quick_start_dependencies_are_edges_only_when_asked():
    pytest.importorskip("networkx")
    run = quick_start()
    assert not [d for _, _, d in ap.to_networkx(run).edges(data=True) if d["kind"] == "dep"]
    graph = ap.to_networkx(run, rules=["tool_call_id"])
    deps = [(u, v, k, d) for u, v, k, d in graph.edges(keys=True, data=True) if d["kind"] == "dep"]
    assert len(deps) == 1
    u, v, k, d = deps[0]
    assert (u, v, k) == ("step:2/call:call_1", "step:3", "tool_call_id_v1")
    assert (d["rule"], d["rule_version"], d["resource"]) == ("tool_call_id", "1", None)


def test_rules_can_be_one_name_one_function_or_a_list():
    pytest.importorskip("networkx")
    run = quick_start()

    def first_to_last(r):
        return [{"src": {"trajectory": r.id, "step": 1}, "dst": {"trajectory": r.id, "step": 3},
                 "rule": "first_to_last", "rule_version": "9"}]

    for rules in ("tool_call_id", ["tool_call_id_v1"]):
        graph = ap.to_networkx(run, rules=rules)
        assert sum(d["kind"] == "dep" for _, _, d in graph.edges(data=True)) == 1
    graph = ap.to_networkx(run, rules=first_to_last)
    assert graph.has_edge("step:1", "step:3", "first_to_last_v9")
    both = ap.to_networkx(run, rules=["tool_call_id", first_to_last, first_to_last])
    keys = sorted(k for _, _, k, d in both.edges(keys=True, data=True) if d["kind"] == "dep")
    assert keys == ["first_to_last_v9", "first_to_last_v9#2", "tool_call_id_v1"]  # one edge per entry


def test_a_dependency_that_names_no_node_is_an_error():
    pytest.importorskip("networkx")

    def stray(r):
        return [{"src": {"trajectory": r.id, "step": 1}, "dst": {"trajectory": r.id, "step": 99},
                 "rule": "stray", "rule_version": "1"}]

    with pytest.raises(ValueError, match="names an address this run has no node for"):
        ap.to_networkx(quick_start(), rules=[stray])


def test_content_is_cut_to_max_content():
    pytest.importorskip("networkx")
    run = quick_start()
    graph = ap.to_networkx(run, max_content=10)
    assert graph.nodes["step:1"]["content"] == "Is it r..." and len(graph.nodes["step:1"]["content"]) == 10
    assert graph.nodes["step:3"]["content"] == "Yes, li..."
    assert graph.nodes["step:2/result:0"]["content"] == "light r..."
    assert ap.to_networkx(run, max_content=None).nodes["step:1"]["content"] == "Is it raining in Paris?"
    assert ap.to_networkx(run, max_content=len("Is it raining in Paris?")).nodes["step:1"]["content"] == "Is it raining in Paris?"


def test_a_step_without_a_message_still_has_a_node():
    pytest.importorskip("networkx")
    run = stand_ins()
    graph = ap.to_networkx(run, rules=["tool_call_id", "full_context"])
    assert list(graph.nodes) == ["step:1", "step:2", "step:2/call:c1", "step:2/call:ap-2-1", "step:2/call:c3",
                                 "step:2/result:0", "step:2/result:1", "step:3", "step:3/call:c4"]
    assert graph.nodes["step:2"]["kind"] == "step" and graph.nodes["step:2"]["role"] == "agent"
    assert graph.nodes["step:2"]["content"] is None and graph.nodes["step:1"]["kind"] == "message"
    deps = {(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "dep" and d["rule"] == "tool_call_id"}
    assert deps == {("step:2/call:c1", "step:3"), ("step:2/call:c3", "step:3")}  # the next agent step has no message
    assert all("kind" in d for _, d in graph.nodes(data=True))  # no edge created a node of its own
    assert sum(d["rule"] == "full_context" for _, _, d in graph.edges(data=True) if d["kind"] == "dep") == 3


def test_result_of_runs_from_a_result_to_its_call_and_names_inferred_pairings():
    pytest.importorskip("networkx")
    run = next(r for r in corpus("adp") if r.id == "agenttuning_alfworld/synth_alf_0")
    graph = ap.to_networkx(run)
    answered = [(u, v, d) for u, v, k, d in graph.edges(keys=True, data=True) if k == "result_of"]
    rows = [r for r in ap.table(run) if r["kind"] == "result" and r["source_call_id"]]
    assert answered and len(answered) == len(rows)
    for (u, v, d), row in zip(sorted(answered), sorted(rows, key=lambda r: (r["step"], r["result"]))):
        assert u == f"step:{row['step']}/result:{row['result']}" and v == f"step:{row['step']}/call:{row['source_call_id']}"
        assert d["paired_by"] == row["paired_by"] == "adp_sequence"
        assert graph.nodes[u]["kind"] == "result" and graph.nodes[v]["kind"] == "call"


def test_who_and_when_graph_places_labels_and_dependencies():
    pytest.importorskip("networkx")
    for run in who_and_when():
        rules = ["tool_call_id", "full_context"]
        deps = ap.derive_deps(run, rules)
        graph = ap.to_networkx(run, rules=rules)
        assert graph.number_of_nodes() == len(run.steps) == len(list(run.events()))
        assert all("kind" in d for _, d in graph.nodes(data=True))
        dep_edges = [(u, v, k, d) for u, v, k, d in graph.edges(keys=True, data=True) if d["kind"] == "dep"]
        assert len(dep_edges) == len(deps) == len(run.steps) * (len(run.steps) - 1) // 2
        for dep in deps:
            u, v = key_of(dep["src"]), key_of(dep["dst"])
            assert graph.has_edge(u, v, f"{dep['rule']}_v{dep['rule_version']}")
            assert graph.edges[u, v, f"{dep['rule']}_v{dep['rule_version']}"]["rule"] == dep["rule"]
        placed = sum(len(d["labels"]) for _, d in graph.nodes(data=True))
        assert placed + len(graph.graph["labels"]) == len(run.labels)
        for label in run.labels:
            target = label["target"]
            if target["kind"] == "step":
                assert label in graph.nodes[key_of(target["address"])]["labels"]
            else:
                assert label in graph.graph["labels"]
        assert graph.graph["run_id"] == run.id
    first = ap.to_networkx(who_and_when()[0])
    assert [l["name"] for l in first.nodes["step:5"]["labels"]] == ["mistake_step", "mistake_reason"]
    assert [l["name"] for l in first.graph["labels"]] == ["mistake_agent"]


def test_resource_and_value_flow_dependencies_join_call_nodes():
    pytest.importorskip("networkx")
    run = editing()
    graph = ap.to_networkx(run, rules=["tool_call_id", "resource_chain", "last_writer", "value_flow"])
    deps = {(u, v, k): d for u, v, k, d in graph.edges(keys=True, data=True) if d["kind"] == "dep"}
    first, second = "step:2/call:e1", "step:3/call:e2"
    assert set(deps) == {
        (first, "step:3", "tool_call_id_v1"), (second, "step:4", "tool_call_id_v1"),
        (first, second, "resource_chain_v1"), (first, second, "last_writer_v1"), (first, second, "value_flow_v1")}
    chain = deps[(first, second, "resource_chain_v1")]
    assert chain["resource"] == "file:/w/a.py" and chain["extractor"] == "file_editor_paths_v1"
    assert deps[(first, second, "value_flow_v1")]["resource"] == ["value:ord-1234"]
    assert deps[(first, second, "value_flow_v1")]["extractor"] == "id_like_values_v1"
    assert all("kind" in d for _, d in graph.nodes(data=True))


def test_dependencies_of_every_rule_land_on_nodes_in_every_fixture():
    pytest.importorskip("networkx")
    rules = ["tool_call_id", "full_context", "value_flow", "resource_chain", "last_writer", "prior_reads"]
    checked = 0
    for run in every_fixture():
        if run.kind != "trajectory":
            continue
        deps = ap.derive_deps(run, rules)
        graph = ap.to_networkx(run, rules=rules)
        events = sum(1 for _ in run.events())
        steps_without_message = len(run.steps) - sum(1 for e in run.events() if e.kind == "message")
        assert graph.number_of_nodes() == events + steps_without_message
        assert all("kind" in d for _, d in graph.nodes(data=True))
        assert sum(d["kind"] == "dep" for _, _, d in graph.edges(data=True)) == len(deps)
        for dep in deps:
            assert graph.has_edge(key_of(dep["src"]), key_of(dep["dst"]))
        placed = sum(len(d["labels"]) for _, d in graph.nodes(data=True))
        assert placed + len(graph.graph["labels"]) == len(run.labels)
        checked += len(deps)
    assert checked > 100


def test_trail_span_labels_land_on_the_node_of_the_span():
    pytest.importorskip("networkx")
    runs = trail()
    on_nodes = off_nodes = 0
    for run in runs:
        graph = ap.to_networkx(run, rules=["tool_call_id", "value_flow"])
        assert all("kind" in d for _, d in graph.nodes(data=True))
        for label in run.labels:
            target = label["target"]
            if target["kind"] != "span":
                assert label in graph.graph["labels"]
                continue
            address = run.spans[target["address"]["span"]]["address"]
            if address:
                assert label in graph.nodes[key_of(address)]["labels"]
                on_nodes += 1
            else:
                assert label in graph.graph["labels"]
                off_nodes += 1
        assert sum(len(d["labels"]) for _, d in graph.nodes(data=True)) + len(graph.graph["labels"]) == len(run.labels)
    assert on_nodes >= 4 and off_nodes >= 1  # a CHAIN span that became no event keeps its label on the graph
    assert any(d["kind"] == "step" for run in runs for _, d in ap.to_networkx(run).nodes(data=True))


def test_graph_labels_are_copies():
    pytest.importorskip("networkx")
    run = who_and_when()[0]
    before = run.to_dict()
    graph = ap.to_networkx(run)
    graph.nodes["step:5"]["labels"][0]["value"] = "changed"
    graph.graph["labels"].clear()
    assert run.to_dict() == before
