"""SWE-Gym OpenHands loader on a synthetic copy of the pinned file layout (no dataset content)."""

import os
from pathlib import Path

import pytest

import agentpile as ap

pytest.importorskip("pyarrow")

ROOT = Path(__file__).parent / "fixtures" / "swe_gym_openhands"


def _runs(**kw):
    return list(ap.load("swe_gym_openhands", root=str(ROOT), **kw))


def _ext(step):
    return step.get("extra", {}).get("agentpile", {})


def test_rows_become_runs_keyed_on_global_row_index():
    runs = _runs()
    assert [r.id for r in runs] == ["0", "1", "2", "3"]  # two rows in shard 0, one in shard 1, one in shard 2
    assert all(ap.validate(r) == [] for r in runs)
    assert [(r.source["file"], r.source["row_in_file"]) for r in runs] == [
        ("data/train.raw-00000-of-00003.parquet", 0), ("data/train.raw-00000-of-00003.parquet", 1),
        ("data/train.raw-00001-of-00003.parquet", 0), ("data/train.raw-00002-of-00003.parquet", 0)]
    assert [r.source["instance_id"] for r in runs] == ["synth__repo-1", "synth__repo-2", "synth__repo-3", "synth__repo-4"]
    assert runs[0].source["record_key"] == "0" and runs[0].source["split"] == "train.raw"
    assert runs[0].source["revision"] == ap.datasets.get("swe_gym_openhands").revision
    assert [r.id for r in _runs(limit=2)] == ["0", "1"]


def test_turns_become_steps_and_results_stay_on_the_step_that_issued_the_call():
    run = _runs()[0]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "agent", "agent", "user", "agent"]
    assert run.steps[2]["observation"]["results"][0]["content"] == "OBSERVATION:\nlisting"  # kept verbatim
    assert run.steps[2]["observation"]["results"][0]["source_call_id"] == "call_A1"
    assert _ext(run.steps[2])["result_fields"] == [{"name": "str_replace_editor"}]
    assert run.record["agent"]["name"] == "OpenHands" and run.record["agent"]["version"] == "unknown"
    assert all("actor" not in _ext(s) for s in run.steps)  # no message names an agent


def test_multi_call_turn_pairs_every_result_by_id_in_call_order():
    step = _runs()[0].steps[3]
    ids = [c["tool_call_id"] for c in step["tool_calls"]]
    assert ids == ["call_B1", "call_B2", "call_B3"]
    assert [c["arguments"] for c in step["tool_calls"]][1] == {"command": "pwd"}
    results = step["observation"]["results"]
    assert [r["source_call_id"] for r in results] == ids
    assert [r["content"] for r in results] == ["OBSERVATION:\nfirst", "OBSERVATION:\nsecond", "OBSERVATION:\nthird"]
    assert _ext(step)["result_refs"] == [5, 6, 7]


def test_a_reused_call_id_binds_to_the_latest_call_only():
    run = _runs()[0]
    first, reused = run.steps[2], run.steps[4]
    assert first["tool_calls"][0]["tool_call_id"] == reused["tool_calls"][0]["tool_call_id"] == "call_A1"
    assert len(first["observation"]["results"]) == 1 and len(reused["observation"]["results"]) == 1
    assert reused["observation"]["results"][0]["content"] == "OBSERVATION:\nreused"
    assert first["observation"]["results"][0]["content"] == "OBSERVATION:\nlisting"
    assert ap.validate(run) == []


def test_empty_and_null_assistant_content_are_told_apart():
    steps = _runs()[0].steps
    assert steps[2]["message"] == "" and "message_state" not in _ext(steps[2])  # empty string is a message
    assert steps[4]["message"] == "" and _ext(steps[4])["message_state"] == "null"
    assert steps[5]["message"] == "Thinking aloud without a tool call." and "tool_calls" not in steps[5]
    assert steps[6] == {"step_id": 7, "source": "user", "message": "Please continue working on the task.",
                        "extra": {"agentpile": {"source_ref": 11}}}


def test_the_final_unanswered_call_leaves_the_observation_empty():
    step = _runs()[0].steps[-1]
    assert step["tool_calls"][0]["function_name"] == "finish" and step["tool_calls"][0]["arguments"] == {}
    assert "observation" not in step
    assert _ext(step)["message_state"] == "null"


def test_tools_become_definitions_and_calls_link_by_function_name():
    run = _runs()[0]
    assert [d["function"]["name"] for d in run.record["agent"]["tool_definitions"]] == [
        "execute_bash", "finish", "str_replace_editor"]
    assert list(run.tools) == ["execute_bash", "finish", "str_replace_editor"]
    assert [t["atif_index"] for t in run.tools.values()] == [0, 1, 2]
    assert {t["permission"] for t in run.tools.values()} == {"unknown"}
    assert {t["permission_from"] for t in run.tools.values()} == {"unknown"}
    assert run.tools["finish"]["definition"] == {"function": {"description": "Finish the task (synthetic).",
                                                                "name": "finish"}, "type": "function"}
    props = run.tools["execute_bash"]["definition"]["function"]["parameters"]["properties"]
    assert list(props) == ["command"]  # the null padding of the other properties is dropped
    assert "enum" not in props["command"]
    calls = [(e.name, e.extra["tool_id"]) for e in run.events() if e.kind == "call"]
    assert calls[0] == ("str_replace_editor", "str_replace_editor") and calls[-1] == ("finish", "finish")


def test_tool_use_ids_keep_their_index_and_bad_arguments_stay_raw():
    run = _runs()[1]
    assert run.record["agent"]["model_name"] == "claude-3-5-sonnet-20241022"
    calls = _ext(run.steps[2])["calls"]
    assert calls["toolu_01A"]["source_fields"] == {"index": 0} and calls["toolu_01B"]["source_fields"] == {"index": 1}
    bad = run.steps[3]
    assert bad["tool_calls"][0]["arguments"] == {}
    assert _ext(bad)["calls"]["toolu_01C"]["arguments_raw"] == '{"command":"create","path":"/w/f.py","file'
    assert bad["observation"]["results"][0]["source_call_id"] == "toolu_01C"


def test_a_call_to_an_undefined_function_gets_no_tool_id():
    last = _runs()[1].steps[-1]
    assert last["tool_calls"][0]["function_name"] == "view" and "observation" not in last
    assert "tool_id" not in _ext(last)["calls"]["toolu_01D"]
    assert _ext(last)["calls"]["toolu_01D"]["source_fields"] == {"index": 1}


def test_model_name_and_run_id():
    plain, claude, bare, empty = _runs()
    assert plain.record["agent"]["model_name"] == "gpt-4o-2024-08-06"
    assert plain.dataset["run_id"] == "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t0-run_1"
    assert "model_name" not in bare.record["agent"]  # no "_maxiter_" in the run id
    assert bare.dataset["run_id"] == "custom-run"
    assert empty.record["agent"]["model_name"] == "gpt-4o-2024-08-06"


def test_outcome_labels_task_and_dataset_fields():
    plain, claude, bare, empty = _runs()
    assert plain.outcome == {"success": False, "score": None, "raw": {"resolved": False}}
    assert claude.outcome["success"] is True
    assert {l["name"]: l["value"] for l in claude.labels} == {
        "resolved": True, "empty_generation": False, "error_eval": False, "failed_apply_patch": False,
        "test_timeout": False}
    assert {l["name"]: l["value"] for l in plain.labels}["empty_generation"] is True
    label = plain.labels[1]
    assert label["target"] == {"kind": "run", "address": {"trajectory": "0"}}
    assert label["source_ref"] == {"field": "test_result.report.empty_generation"}
    assert plain.task["id"] == "synth__repo-1" and plain.task["text"].startswith("<uploaded_files>")
    assert plain.dataset["test_result"]["report"]["empty_generation"] is True
    assert plain.dataset["test_result"]["test_output"] is None and "messages" not in plain.dataset
    assert claude.dataset["test_result"]["git_patch"] == "diff --git a/f b/f" and "tools" not in claude.dataset


def test_a_row_without_resolved_tools_or_test_result():
    bare = _runs()[2]
    assert bare.outcome == {"success": None, "score": None, "raw": {"resolved": None}}
    assert bare.labels == [] and bare.tools == {}
    assert "tool_definitions" not in bare.record["agent"]
    assert bare.dataset["test_result"] is None
    assert [s["source"] for s in bare.steps] == ["system", "user", "agent"]
    assert ap.validate(bare) == []


def test_an_empty_message_list_still_becomes_a_run():
    empty = _runs()[3]
    assert len(empty.steps) == 1 and _ext(empty.steps[0]) == {"message_state": "missing", "empty_source": True}
    assert empty.task["text"] is None and empty.outcome["success"] is False
    assert [l["name"] for l in empty.labels][0] == "resolved"


def test_null_padding_is_dropped_and_named():
    plain, claude, bare, empty = _runs()
    assert plain.source["null_members_dropped"] == [
        "messages[].function_call", "messages[].name", "messages[].tool_call_id", "messages[].tool_calls",
        "messages[].tool_calls[].index", "tools[].*"]
    assert "messages[].tool_calls[].index" not in claude.source["null_members_dropped"]  # every call has an index
    assert bare.source["null_members_dropped"][-1] == "messages[].tool_calls"
    assert empty.source["null_members_dropped"] == ["tools[].*"]
    for run in (plain, bare):  # no padded key survives into a step, a result or a call
        for step in run.steps:
            sx = _ext(step)
            assert "source_fields" not in sx
            assert all("source_fields" not in c for c in sx.get("calls", {}).values())
            assert all(f is None or "tool_call_id" not in f for f in sx.get("result_fields", []))


def test_a_missing_shard_set_and_options_are_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(ap.load("swe_gym_openhands", root=str(tmp_path)))
    with pytest.raises(TypeError):
        _runs(subset="x")


def test_fixture_runs_validate_against_harbor():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / "swe_gym_openhands") if data else None
    runs = steps = calls = results = multi = multi_answered = resolved = 0
    for run in ap.load("swe_gym_openhands", root=root, verify=True):  # streamed: a run can hold 200k characters of logs
        runs += 1
        steps += len(run.steps)
        resolved += bool(run.outcome["success"])
        assert ap.validate(run) == []
        for step in run.steps:
            step_calls = step.get("tool_calls") or []
            step_results = (step.get("observation") or {}).get("results") or []
            calls += len(step_calls)
            results += len(step_results)
            assert all(r["source_call_id"] is not None for r in step_results)
            if len(step_calls) > 1:
                multi += 1
                if step_results:
                    multi_answered += 1
                    assert [r["source_call_id"] for r in step_results] == [c["tool_call_id"] for c in step_calls]
    assert (runs, resolved) == (6055, 491)
    assert (steps, calls, results) == (136848, 106882, 101003)
    assert (multi, multi_answered) == (2179, 2136)


def test_a_name_defined_twice_links_no_call():
    raw = {"instance_id": "synth__x-1", "run_id": None, "resolved": None, "test_result": None,
           "tools": [{"type": "function", "function": {"name": "run", "description": "first"}},
                     {"type": "function", "function": {"name": "run", "description": "second"}}],
           "messages": [{"role": "user", "content": "go"},
                        {"role": "assistant", "content": None, "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]}]}
    where = {"file": "f.parquet", "config": "default", "split": "train.raw", "row_in_file": 0}
    run = ap.datasets.get("swe_gym_openhands").convert("0", raw, where)
    assert list(run.tools) == ["run", "tool-1"] and ap.validate(run) == []
    assert run.steps[-1]["tool_calls"][0]["tool_call_id"] == "c1"
    assert "tool_id" not in (_ext(run.steps[-1]).get("calls") or {}).get("c1", {})
