"""tau-bench loader on a synthetic copy of the pinned file layout (no dataset content).

The fixture has three files in the shape of the real ones: ``model-a`` (tool
messages with call ids, a placeholder row with no messages, a score/db_match
split, an empty function name), ``model-gem`` (results stored as id-less user
messages, bare ``meta.id``, no ``meta.task_id``) and ``model-think`` (a user
message after every tool result, a ``model_path`` that does not match the file).
"""

import os
from pathlib import Path

import pytest

import agentpile as ap

ROOT = Path(__file__).parent / "fixtures" / "tau_bench"


def _runs(**kw):
    return list(ap.load("tau_bench", root=str(ROOT), **kw))


def _by_id():
    return {r.id: r for r in _runs()}


def test_rows_become_runs_in_file_order_with_unique_keys():
    runs = _runs()
    assert [r.id for r in runs] == [
        "model-a/airline/0", "model-a/airline/1", "model-a/retail/5", "model-a/retail/6",
        "model-gem/retail/0", "model-gem/airline/0", "model-think/airline/2"]
    assert len({r.id for r in runs}) == len(runs)
    assert all(ap.validate(r) == [] for r in runs)
    src = runs[3].source
    assert (src["file"], src["row"], src["record_key"], src["config"], src["split"]) == (
        "model-a.jsonl", 3, "model-a/retail/6", "default", "train")
    assert src["revision"] == ap.datasets.get("tau_bench").revision


def test_filters_by_model_and_task_name():
    assert [r.id for r in _runs(model="model-gem")] == ["model-gem/retail/0", "model-gem/airline/0"]
    assert [r.id for r in _runs(model=["model-think", "model-gem"], task_name="airline")] == [
        "model-gem/airline/0", "model-think/airline/2"]
    with pytest.raises(ValueError, match="unknown model"):
        _runs(model="no-such-model")
    with pytest.raises(TypeError):
        _runs(split="train")


def test_model_comes_from_the_file_stem_not_model_path():
    think = _by_id()["model-think/airline/2"]
    assert think.record["agent"] == {"name": "unknown", "version": "unknown", "model_name": "model-think"}
    assert think.dataset["model_path"] == "vendor/model-a"  # the source value stays, but it is not the identity
    assert [a["id"] for a in think.agents] == ["model-think"] and think.agents[0]["model"] == "model-think"
    assert {s["extra"]["agentpile"]["actor"] for s in think.steps if s["source"] == "agent"} == {"model-think"}
    assert think.record["agent"].get("tool_definitions") is None and think.tools == {}


def test_call_and_result_share_one_agent_step_paired_by_id():
    run = _by_id()["model-a/airline/0"]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "agent", "user"]
    step = run.steps[2]
    call = step["tool_calls"][0]
    assert (call["tool_call_id"], call["function_name"], call["arguments"]) == ("call_a0", "get_user_details", {"user_id": "u1"})
    res = step["observation"]["results"]
    assert len(res) == 1 and res[0]["source_call_id"] == "call_a0" and res[0]["content"] == '{"name": "Synthetic User"}'
    sx = step["extra"]["agentpile"]
    assert sx["result_refs"] == [3] and sx["result_fields"] == [{"turn_idx": 3, "name": "get_user_details"}]
    assert sx["calls"]["call_a0"]["source_fields"] == {"index": 0}  # call.index stays in the extension
    assert [e.kind for e in run.events()][:5] == ["message", "message", "message", "call", "result"]


def test_null_and_empty_content_and_turn_idx():
    run = _by_id()["model-a/airline/0"]
    null_step, empty_step = run.steps[2], run.steps[3]
    assert null_step["message"] == "" and null_step["extra"]["agentpile"]["message_state"] == "null"
    assert empty_step["message"] == "" and "message_state" not in empty_step["extra"]["agentpile"]  # "" is a real value
    assert empty_step["observation"]["results"][0]["content"] == ""
    assert [s["extra"]["agentpile"]["source_fields"]["turn_idx"] for s in run.steps] == [0, 1, 2, 4, 6, 7]
    assert "index" not in empty_step["extra"]["agentpile"].get("calls", {}).get("call_a1", {}).get("source_fields", {})


def test_call_with_empty_function_name_and_nameless_tool_message_are_kept_as_given():
    run = _by_id()["model-a/airline/1"]
    step = run.steps[2]
    assert step["tool_calls"][0]["function_name"] == "" and step["tool_calls"][0]["arguments"] == {"user_id": "u2"}
    assert step["extra"]["agentpile"]["calls"]["call_b0"]["source_fields"] == {"index": None}
    assert step["observation"]["results"][0]["source_call_id"] == "call_b0"
    assert step["extra"]["agentpile"]["result_fields"] == [{"turn_idx": 3}]  # no name in the source, none invented


def test_raw_model_markup_stays_in_the_message():
    step = _by_id()["model-a/retail/6"].steps[2]
    assert step["message"] == "<think>plan the cancel</think>Let me look."
    assert "reasoning_content" not in step


def test_placeholder_row_is_a_run_with_an_empty_source_step():
    run = _by_id()["model-a/retail/5"]
    assert len(run.steps) == 1 and run.steps[0]["source"] == "system" and run.steps[0]["message"] == ""
    assert run.steps[0]["extra"]["agentpile"] == {"message_state": "missing", "empty_source": True}
    assert all(l["source_ref"].get("empty_run") is True for l in run.labels)
    assert {l["name"]: l["value"] for l in run.labels}["db_match"] is False
    assert run.task == {"id": "retail/5", "text": ""}


def test_results_without_a_call_id_stay_user_steps_and_calls_stay_unanswered():
    run = _by_id()["model-gem/airline/0"]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "user", "agent", "user", "agent", "user"]
    calls = [c for s in run.steps for c in s.get("tool_calls") or []]
    assert [c["tool_call_id"] for c in calls] == ["sgl_gemini_tool_call_bbbb", "sgl_gemini_tool_call_cccc"]
    assert all("observation" not in s for s in run.steps)  # nothing paired by adjacency
    assert run.steps[3]["message"] == '{"status": "on time"}' and run.steps[3]["extra"]["agentpile"].get("role") is None
    assert all(e.kind != "result" for e in run.events())


def test_bare_ids_repeating_across_domains_still_give_unique_keys_and_task_ids():
    ids = [r.id for r in _runs(model="model-gem")]
    assert ids == ["model-gem/retail/0", "model-gem/airline/0"]
    retail, airline = _runs(model="model-gem")
    assert (retail.task["id"], airline.task["id"]) == ("retail/0", "airline/0")
    assert retail.dataset["meta"]["id"] == "0" and "task_id" not in retail.dataset["meta"]  # as the source has it
    assert _by_id()["model-a/airline/0"].task["id"] == airline.task["id"]  # same task, prefixed or bare id


def test_user_message_after_a_tool_result_stays_a_user_step():
    run = _by_id()["model-think/airline/2"]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "user", "agent", "user"]
    assert run.steps[3]["message"] == "The tool has finished running. Please continue your reasoning."
    assert run.steps[2]["observation"]["results"][0]["source_call_id"] == "toolu_001"


def test_outcome_score_and_labels_expose_db_match_and_score_separately():
    runs = _by_id()
    ok, split, failed = runs["model-a/airline/0"], runs["model-a/airline/1"], runs["model-a/retail/6"]
    assert ok.outcome == {"success": True, "score": 1.0,
                          "raw": {"eval_result": {"score": 1.0, "db_match": True}, "meta": {"is_correct": True}}}
    assert split.outcome["success"] is False and split.outcome["score"] == 1.0  # score 1.0, db_match false
    assert split.outcome["raw"]["meta"] == {"is_correct": False}
    assert failed.outcome["success"] is False and failed.outcome["score"] == 0.0
    labels = {l["name"]: l for l in split.labels}
    assert list(labels) == ["db_match", "score", "db_reward", "finish_reason", "gold_actions"]
    assert (labels["db_match"]["value"], labels["score"]["value"], labels["db_reward"]["value"]) == (False, 1.0, None)
    assert labels["finish_reason"]["value"] == "user_stop" and labels["gold_actions"]["value"] == []
    assert labels["db_match"]["target"] == {"kind": "run", "address": {"trajectory": "model-a/airline/1"}}
    assert labels["db_match"]["source_ref"] == {"field": "eval_result.db_match"}
    assert labels["db_reward"]["source_ref"]["field"] == "meta.reward_details.db_reward"
    assert failed.labels[-1]["value"] == [{"name": "cancel_pending_order", "kwargs": {"order_id": "#W1"}}]


def test_task_and_dataset_fields():
    run = _by_id()["model-a/airline/0"]
    assert run.task == {"id": "airline/0", "text": "You are user u1 and want a synthetic booking."}
    d = run.dataset
    assert "messages" not in d
    assert d["user_model_path"] == "vendor/user-sim" and d["benchmark_name"] == "taubench"
    assert d["sampling_params"] == {"max_tokens": None, "temperature": 0.0}
    assert d["meta"]["id"] == "airline_0" and d["meta"]["source_file"] == "synthetic-airline-airline_0.json"
    assert d["meta"]["agent_cost"] == 0.5 and d["meta"]["task_id"] == "0"


def test_harbor_accepts_every_fixture_run():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    runs = list(ap.load("tau_bench", root=str(Path(data) / "tau_bench") if data else None, verify=True))
    assert len(runs) == 4950 and len({r.id for r in runs}) == 4950
    assert sum(len(r.steps) for r in runs) == 116965  # 147974 messages - 31103 tool messages + 94 placeholder steps
    assert all(ap.validate(r) == [] for r in runs)
    calls = sum(len(s.get("tool_calls") or []) for r in runs for s in r.steps)
    results = sum(len((s.get("observation") or {}).get("results") or []) for r in runs for s in r.steps)
    assert (calls, results) == (34758, 31103)  # the 3655 gemini results are user steps, not results
    assert sum(len(r.steps) == 1 and r.steps[0]["extra"]["agentpile"].get("empty_source", False) for r in runs) == 94
    assert sum(r.outcome["success"] is True for r in runs) == 2073
    assert sum(r.outcome["score"] == 1.0 and r.outcome["success"] is False for r in runs) == 651
    assert all(r.outcome["raw"]["meta"]["is_correct"] == r.outcome["success"] for r in runs)
