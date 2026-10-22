"""SWE-rebench OpenHands loader on a synthetic copy of the pinned file layout (no dataset content)."""

import json
import os
import shutil
from pathlib import Path

import pytest

import agentpile as ap

pytest.importorskip("pyarrow")

NAME = "swe_rebench_openhands"
ROOT = Path(__file__).parent / "fixtures" / NAME
TOOL_NAMES = ["execute_bash", "think", "finish", "task_tracker", "str_replace_editor"]


def _runs(**kw):
    return list(ap.load(NAME, root=str(ROOT), **kw))


def _sx(step):
    return (step.get("extra") or {}).get("agentpile") or {}


def test_one_run_per_row_in_file_order():
    runs = _runs()
    assert [r.id for r in runs] == ["syn-traj-1", "syn-traj-2", "syn-traj-3", "syn-traj-4"]
    assert all(ap.validate(r) == [] for r in runs)
    first = runs[0]
    assert [r.source["row_index"] for r in runs] == [0, 1, 2, 3]
    assert first.source["file"] == "trajectories.parquet" and first.source["record_key"] == "syn-traj-1"
    assert first.source["config"] == "default" and first.source["split"] == "train"
    assert first.source["revision"] == ap.datasets.get(NAME).revision
    assert first.source["lost_fields"] == ["tools"]
    assert first.record["session_id"] == first.record["trajectory_id"] == "syn-traj-1"


def test_batch_size_does_not_change_the_runs(monkeypatch):
    whole = [r.record for r in _runs()]
    monkeypatch.setattr(ap.datasets.get(NAME), "batch_rows", 1)
    assert [r.record for r in _runs()] == whole
    assert [r.id for r in _runs(limit=2)] == ["syn-traj-1", "syn-traj-2"]


def test_options_are_rejected():
    with pytest.raises(TypeError):
        _runs(subset="x")


def test_agent_block_states_the_card_and_carries_tools_json():
    run = _runs()[0]
    agent = run.record["agent"]
    assert (agent["name"], agent["version"], agent["model_name"]) == (
        "OpenHands", "0.54.0", "Qwen/Qwen3-Coder-480B-A35B-Instruct")
    assert agent["tool_definitions"] == json.loads((ROOT / "tools.json").read_text(encoding="utf-8"))
    assert run.dataset["agent_from"] == "README.md" and run.dataset["tool_definitions_from"] == "tools.json"
    assert run.agents == []


def test_tools_map_has_one_entry_per_definition():
    run = _runs()[0]
    defs = run.record["agent"]["tool_definitions"]
    assert list(run.tools) == [f"openhands.{n}" for n in TOOL_NAMES]
    for i, (tid, tool) in enumerate(run.tools.items()):
        assert tool["atif_index"] == i and tool["name"] == TOOL_NAMES[i] and tool["definition"] == defs[i]
        assert (tool["definition_format"], tool["permission"], tool["permission_from"]) == ("openai", "unknown", "unknown")
        assert tool["type"] == "function" and tool["agent_or_server"] is None


def test_runs_do_not_share_definition_objects():
    a, b = _runs()[:2]
    a.record["agent"]["tool_definitions"][0]["function"]["name"] = "changed"
    assert b.record["agent"]["tool_definitions"][0]["function"]["name"] == "execute_bash"


def test_steps_follow_messages_and_results_sit_on_the_calling_step():
    run = _runs()[0]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "agent"]
    think, bash, finish = run.steps[2:]
    assert think["message"] == "" and "message_state" not in _sx(think)  # empty content is a real value
    assert think["tool_calls"] == [{"tool_call_id": "syn-call-1a", "function_name": "think",
                                    "arguments": {"thought": "plan the fix"}}]
    assert think["observation"]["results"] == [{"source_call_id": "syn-call-1a", "content": "Your thought has been logged."}]
    assert _sx(think)["result_refs"] == [3] and _sx(think)["source_ref"] == 2
    assert bash["message"] == "Running the tests." and bash["observation"]["results"][0]["content"] == "1 passed"
    assert "observation" not in finish and finish["tool_calls"][0]["function_name"] == "finish"  # the last call has no result
    assert "actor" not in _sx(think)


def test_parquet_null_padding_is_dropped_and_real_members_are_kept():
    run = _runs()[0]
    system, user, think = run.steps[:3]
    assert _sx(system) == {"source_ref": 0} and _sx(user) == {"source_ref": 1}
    assert _sx(think)["result_fields"] == [{"name": "think"}]  # the tool message's name, not tool_calls: None
    assert "source_fields" not in _sx(think)
    no_call = _runs()[1].steps[3]  # an assistant turn whose tool_calls member is null
    assert "tool_calls" not in no_call and _sx(no_call) == {"source_ref": 5}


def test_calls_carry_tool_id_and_type():
    run = _runs()[0]
    for step, tid in zip(run.steps[2:], ["openhands.think", "openhands.execute_bash", "openhands.finish"]):
        cid = step["tool_calls"][0]["tool_call_id"]
        assert _sx(step)["calls"][cid] == {"tool_id": tid, "source_fields": {"type": "function"}}
    kinds = [(e.kind, e.name, e.extra.get("tool_id")) for e in run.events() if e.kind == "call"]
    assert kinds == [("call", "think", "openhands.think"), ("call", "execute_bash", "openhands.execute_bash"),
                     ("call", "finish", "openhands.finish")]


def test_two_call_turn_unknown_function_and_user_follow_up():
    run = _runs()[1]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "user", "agent", "agent"]
    two = run.steps[2]
    assert [c["tool_call_id"] for c in two["tool_calls"]] == ["syn-call-2a", "syn-call-2b"]
    assert [r["source_call_id"] for r in two["observation"]["results"]] == ["syn-call-2a", "syn-call-2b"]
    assert [r["content"] for r in two["observation"]["results"]] == ["a.py", "file body"]
    assert _sx(two)["result_refs"] == [3, 4]
    assert [f["name"] for f in _sx(two)["result_fields"]] == ["execute_bash", "str_replace_editor"]
    assert run.steps[4]["message"] == "SYNTHETIC continue prompt." and "role" not in _sx(run.steps[4])
    unknown = run.steps[5]
    assert unknown["tool_calls"][0]["function_name"] == "browser_goto" and unknown["tool_calls"][0]["arguments"] == {}
    cx = _sx(unknown)["calls"]["syn-call-2c"]
    assert "tool_id" not in cx and cx["source_fields"] == {"type": "function"} and "arguments_raw" not in cx
    assert unknown["observation"]["results"][0]["source_call_id"] == "syn-call-2c"


def test_arguments_that_are_not_an_object_keep_the_source_string():
    step = _runs()[3].steps[2]
    assert step["tool_calls"][0]["arguments"] == {}
    cx = _sx(step)["calls"]["syn-call-4a"]
    assert cx["arguments_raw"] == "not json" and cx["arguments_state"] == "unparsed" and cx["tool_id"] == "openhands.think"
    call = next(e for e in _runs()[3].events() if e.kind == "call")
    assert call.content == "not json"


def test_empty_message_list_still_becomes_a_run():
    run = _runs()[2]
    assert len(run.steps) == 1 and _sx(run.steps[0]) == {"message_state": "missing", "empty_source": True}
    assert run.outcome["success"] is None and run.outcome["raw"] == {"resolved": None}
    assert ap.validate(run) == []


def test_outcome_task_and_labels():
    a, b, c, d = _runs()
    assert a.task == {"id": "synorg__synrepo-1", "text": None}
    assert a.outcome == {"success": True, "score": None, "raw": {"resolved": 1}}
    assert b.outcome == {"success": False, "score": None, "raw": {"resolved": 0}}
    assert [l["name"] for l in a.labels] == ["resolved", "exit_status", "gen_tests_correct", "pred_passes_gen_tests"]
    assert [l["value"] for l in a.labels] == [1, "submit", 1.0, 0.0]
    assert [l["value"] for l in b.labels] == [0, "RuntimeError: synthetic iteration limit", None, None]  # nulls stay null
    assert [l["value"] for l in c.labels] == [None, "unknown", 0.0, 1.0]
    for label in a.labels:
        assert label["target"] == {"kind": "run", "address": {"trajectory": "syn-traj-1"}}
        assert label["label_source"] == NAME and label["source_ref"] == {"field": label["name"]}
    assert d.labels[2]["value"] is None and d.outcome["success"] is True


def test_remaining_row_fields_are_kept_in_dataset():
    a, b = _runs()[:2]
    assert a.dataset["instance_id"] == "synorg__synrepo-1" and a.dataset["repo"] == "synorg/synrepo"
    assert a.dataset["model_patch"].startswith("diff --git") and b.dataset["model_patch"] == ""
    assert "trajectory" not in a.dataset and "tools" not in a.dataset


def test_duplicate_function_names_in_tools_json_are_rejected(tmp_path):
    shutil.copy(ROOT / "trajectories.parquet", tmp_path / "trajectories.parquet")
    defs = json.loads((ROOT / "tools.json").read_text(encoding="utf-8"))
    (tmp_path / "tools.json").write_text(json.dumps(defs + [defs[0]]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique function name"):
        list(ap.load(NAME, root=str(tmp_path)))


def test_manifest_pins_the_files_the_loader_reads():
    m = ap.datasets.get(NAME).manifest
    assert m["hf_id"] == "nebius/SWE-rebench-openhands-trajectories" and len(m["revision"]) == 40
    assert "trajectories.parquet" in m["allow_patterns"] and "tools.json" in m["allow_patterns"]
    assert m["counts"]["runs_after_load"]["value"] == 67074


def test_harbor_accepts_every_fixture_run():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / NAME) if data else None
    calls = results = resolved = n = 0
    ids = set()
    for run in ap.load(NAME, root=root, verify=True):
        n += 1
        ids.add(run.id)
        resolved += bool(run.outcome["success"])
        for step in run.steps:
            calls += len(step.get("tool_calls") or [])
            results += len((step.get("observation") or {}).get("results") or [])
        if n % 500 == 1:
            assert ap.validate(run) == [], run.id
    assert n == 67074 and len(ids) == 67074
    assert (calls, results, resolved) == (4316760, 4249707, 32161)
    if root is not None and (Path(root) / "README.md").is_file():
        card = (Path(root) / "README.md").read_text(encoding="utf-8")
        assert "OpenHands (v0.54.0)" in card and "Qwen/Qwen3-Coder-480B-A35B-Instruct" in card
