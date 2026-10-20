"""tau2-bench loader on a synthetic copy of the pinned file layout (no dataset content)."""

import collections
import copy
import os
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.corpora.tau2_bench import GREETING, dataset

ROOT = Path(__file__).parent / "fixtures" / "tau2_bench"

A0 = "alpha-model/telecom/[synthetic_issue]case_one[PERSONA:None]"
A1 = "alpha-model/airline/7"
B0 = "beta-model/retail/3"
B1 = "beta-model/airline/row1"
B2 = "beta-model/telecom/[synthetic_issue]case_one[PERSONA:None]"


def _runs(**kw):
    return list(ap.load("tau2_bench", root=str(ROOT), **kw))


def _by_id():
    return {r.id: r for r in _runs()}


def _sx(step):
    return step["extra"]["agentpile"]


def test_rows_become_runs_in_file_and_row_order():
    runs = _runs()
    assert [r.id for r in runs] == [A0, A1, B0, B1, B2]
    assert all(ap.validate(r) == [] for r in runs)
    assert [r.id for r in _runs(model="beta-model")] == [B0, B1, B2]
    assert [r.id for r in _runs(domain="telecom")] == [A0, B2]
    assert [r.id for r in _runs(model="alpha-model", domain="airline")] == [A1]
    with pytest.raises(TypeError):
        _runs(subset="x")


def test_results_pair_with_calls_by_position_and_are_flagged():
    run = _by_id()[A0]
    steps = run.steps
    assert [s["source"] for s in steps] == ["system", "agent", "user", "agent", "agent", "agent", "agent", "user"]
    multi = steps[4]  # one message with two calls followed by two result messages
    assert [c["tool_call_id"] for c in multi["tool_calls"]] == ["a-c2", "a-c3"]
    assert [(r["source_call_id"], r["content"]) for r in multi["observation"]["results"]] == [
        ("a-c2", "usage-result"), ("a-c3", "bill-result")]
    assert _sx(multi)["result_refs"] == [6, 7]
    assert _sx(multi)["result_fields"] == [{"turn_idx": 6, "paired_by": "position"},
                                          {"turn_idx": 7, "paired_by": "position"}]
    single = steps[3]
    assert single["message"] == "" and _sx(single)["message_state"] == "null"  # null content kept as null
    assert single["observation"]["results"][0]["source_call_id"] == "a-c1"
    # every result in the fixture came from a position pairing
    flags = [f.get("paired_by") for s in steps for f in _sx(s).get("result_fields", [])]
    assert flags == ["position"] * 5
    events = [(e.kind, e.call_id) for e in run.events() if e.kind != "message"]
    assert events == [("call", "a-c1"), ("result", "a-c1"), ("call", "a-c2"), ("call", "a-c3"),
                      ("result", "a-c2"), ("result", "a-c3"), ("call", "a-u1"), ("call", "a-u2"),
                      ("result", "a-u1"), ("result", "a-u2")]


def test_calls_without_results_and_extra_results_stay_unpaired():
    cut = _by_id()[A1].steps[-1]  # a max_steps run ends on a call message with no results
    assert len(cut["tool_calls"]) == 2 and "observation" not in cut
    assert cut["message"] == "" and "message_state" not in _sx(cut)  # the source text was an empty string
    extra = _by_id()[B0].steps[4]  # two results after one call: the second answers no call
    assert [r["source_call_id"] for r in extra["observation"]["results"]] == ["b-c1", None]
    assert _sx(extra)["result_fields"] == [{"turn_idx": 6, "paired_by": "position"}, {"turn_idx": 7}]
    assert extra["tool_calls"][0]["function_name"] == ""  # an empty call name is kept as given


def test_user_simulator_calls_are_agent_steps_with_actor_user():
    run = _by_id()[A0]
    sim = run.steps[6]
    assert sim["source"] == "agent" and _sx(sim)["actor"] == "user" and _sx(sim)["role"] == "user"
    assert [c["function_name"] for c in sim["tool_calls"]] == ["check_status", "toggle_mode"]
    assert _sx(sim)["calls"]["a-u1"] == {"source_fields": {"requestor": "user"}}
    agent = run.steps[3]
    assert _sx(agent)["actor"] == "assistant" and "role" not in _sx(agent)
    assert _sx(agent)["calls"]["a-c1"] == {"source_fields": {"requestor": "assistant"}}
    plain_user = run.steps[2]  # a user message without calls is an ordinary user step
    assert plain_user["source"] == "user" and "actor" not in _sx(plain_user)
    assert run.steps[7]["message"] == "Thanks, it works now. ###STOP###"  # control tokens stay
    mixed = _by_id()[B0].steps[3]  # user text and a call in one message
    assert mixed["source"] == "agent" and mixed["message"] == "Checking the app myself." and _sx(mixed)["actor"] == "user"
    assert _by_id()[B0].steps[4]["message"].startswith("<think>")  # raw model markup stays


def test_scripted_greeting_is_flagged_and_credits_no_model():
    run = _by_id()[A0]
    assert run.steps[1]["message"] == GREETING and _sx(run.steps[1])["scripted"] is True
    assert not any("scripted" in _sx(s) for i, s in enumerate(run.steps) if i != 1)
    assert not any("model_name" in s for s in run.steps)
    assert run.record["agent"] == {"name": "alpha-model", "version": "unknown", "model_name": "synthetic/alpha-model"}
    where = {"file": "alpha-model.jsonl", "config": "default", "split": "train", "row": 0}
    key, raw, _ = next(dataset.records(ROOT))
    for other in ("Hello there.", None):
        changed = copy.deepcopy(raw)
        changed["messages"][1]["content"] = other
        assert not any("scripted" in _sx(s) for s in dataset.convert(key, changed, where).steps)


def test_outcome_and_labels():
    runs = _by_id()
    a0 = runs[A0]
    assert a0.outcome == {"success": True, "score": 1.0, "raw": {"score": 1.0, "db_match": False}}
    labels = {l["name"]: l for l in a0.labels}
    assert set(labels) == {"score", "db_match", "finish_reason", "reward_breakdown.ENV_ASSERTION"}
    assert all(l["target"] == {"kind": "run", "address": {"trajectory": A0}} for l in a0.labels)
    assert labels["score"]["value"] == 1.0 and labels["finish_reason"]["value"] == "user_stop"
    assert labels["db_match"]["value"] is False and "telecom" in labels["db_match"]["source_ref"]["note"]
    assert labels["reward_breakdown.ENV_ASSERTION"]["source_ref"]["field"] == \
        "meta.reward_details.reward_breakdown.ENV_ASSERTION"
    a1 = runs[A1]  # truncated: score 0.0, no breakdown, finish_reason says why
    assert a1.outcome["success"] is False
    assert {l["name"]: l["value"] for l in a1.labels} == {"score": 0.0, "db_match": False, "finish_reason": "max_steps"}
    b0 = runs[B0]
    assert {l["name"] for l in b0.labels} == {"score", "db_match", "finish_reason",
                                             "reward_breakdown.COMMUNICATE", "reward_breakdown.DB"}
    assert "note" not in {l["name"]: l for l in b0.labels}["db_match"]["source_ref"]
    assert runs[B2].outcome["success"] is None and runs[B2].outcome["score"] == 0.5  # neither 0 nor 1


def test_identity_task_and_dataset_fields():
    runs = _by_id()
    a0, b2 = runs[A0], runs[B2]
    assert a0.task == {"id": "telecom/[synthetic_issue]case_one[PERSONA:None]", "text": None}
    assert b2.task == a0.task  # a bare id and a prefixed id name the same task in two models
    assert runs[A1].task["id"] == "airline/7" and runs[B0].task["id"] == "retail/3"
    src = a0.source
    assert (src["file"], src["row"], src["record_key"], src["config"], src["split"]) == (
        "alpha-model.jsonl", 0, A0, "default", "train")
    assert src["revision"] == ap.datasets.get("tau2_bench").revision and src["lost_fields"] == []
    assert runs[B2].source["row"] == 2
    assert "messages" not in a0.dataset and a0.dataset["task_name"] == "telecom"
    assert a0.dataset["meta"]["task_id"] == "[synthetic_issue]case_one[PERSONA:None]"
    assert a0.dataset["meta"]["task_instructions"]["reason_for_call"] == "Synthetic reason."
    assert a0.dataset["meta"]["duration_seconds"] == 8.0 and a0.dataset["sampling_params"]["temperature"] == 0.0
    assert [(a["id"], a["name"], a["model"]) for a in a0.agents] == [
        ("assistant", "alpha-model", "synthetic/alpha-model"), ("user", "user_simulator", "synthetic/user-sim")]
    assert a0.tools == {}
    assert a0.steps[0]["message"] == "Synthetic policy text." and _sx(a0.steps[0])["source_fields"] == {"turn_idx": -1}


def test_row_without_messages_or_id_still_becomes_a_run():
    run = _by_id()[B1]
    assert len(run.steps) == 1 and _sx(run.steps[0])["empty_source"] is True
    assert run.task["id"] is None and run.source["record_key"] == B1
    assert run.outcome["success"] is False and ap.validate(run) == []


def test_fixture_validates_against_harbor():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


def test_count_check_uses_the_manifest():
    with pytest.raises(ap.AgentpileError):
        list(ap.load("tau2_bench", root=str(ROOT), verify=True))  # 5 fixture rows, manifest says 8340
    assert len(_runs()) == 5


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / "tau2_bench") if data else None
    ids, n = set(), collections.Counter()
    for run in ap.load("tau2_bench", root=root, verify=True):  # streamed: 8340 runs do not fit comfortably in memory
        ids.add(run.id)
        assert ap.validate(run) == []
        n["success"] += run.outcome["success"] is True
        n["failure"] += run.outcome["success"] is False
        for s in run.steps:
            n["steps"] += 1
            n["calls"] += len(s.get("tool_calls") or [])
            results = (s.get("observation") or {}).get("results") or []
            n["results"] += len(results)
            n["unpaired"] += sum(x["source_call_id"] is None for x in results)
            n["by_position"] += sum(f.get("paired_by") == "position" for f in _sx(s).get("result_fields", []))
            n["calls_without_results"] += bool(s.get("tool_calls") and not results)
            n["scripted"] += bool(_sx(s).get("scripted"))
            n["user_agent_steps"] += s["source"] == "agent" and _sx(s)["actor"] == "user"
    assert len(ids) == 8340
    assert dict(n) == {"success": 3984, "failure": 4356, "steps": 265488, "calls": 113603, "results": 113424,
                       "unpaired": 0, "by_position": 113424, "calls_without_results": 167, "scripted": 8340,
                       "user_agent_steps": 29477}
