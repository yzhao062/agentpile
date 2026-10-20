"""SWE-agent trajectories (Nebius) loader on a synthetic copy of the pinned Parquet layout (no dataset content)."""

import os
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.corpora import swe_agent_nebius as mod

pytest.importorskip("pyarrow")

ROOT = Path(__file__).parent / "fixtures" / "swe_agent_nebius"
PAD = "\n(Open file: n/a)\n(Current directory: /synth)\nbash-$"


def _runs(**kw):
    return list(ap.load("swe_agent_nebius", root=str(ROOT), **kw))


def _ext(step):
    return step["extra"]["agentpile"]


def test_rows_become_runs_in_shard_order():
    runs = _runs()
    assert [r.id for r in runs] == ["train-00000-of-00012:0", "train-00000-of-00012:1",
                                    "train-00001-of-00012:0", "train-00001-of-00012:1"]
    assert all(ap.validate(r) == [] for r in runs)
    assert runs[0].source["file"] == "data/train-00000-of-00012.parquet"
    assert runs[3].source["record_key"] == "train-00001-of-00012:1" and runs[3].source["split"] == "train"
    assert runs[0].source["revision"] == ap.datasets.get("swe_agent_nebius").revision


def test_row_index_runs_on_across_batches(monkeypatch):
    monkeypatch.setattr(mod, "BATCH_ROWS", 1)
    assert [r.id for r in _runs()] == ["train-00000-of-00012:0", "train-00000-of-00012:1",
                                       "train-00001-of-00012:0", "train-00001-of-00012:1"]


def test_turns_become_steps_and_no_call_is_invented():
    run = _runs()[0]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "agent"]
    assert run.steps[0]["message"].startswith("SYNTHETIC SETTING")  # system_prompt is the system message
    assert _ext(run.steps[0])["source_fields"] == {"cutoff_date": "01.01.2023", "mask": False}
    assert run.steps[1]["message"].startswith("Synthetic issue") and "role" not in _ext(run.steps[1])
    assert [_ext(s).get("role") for s in run.steps[2:]] == ["ai", "ai", "ai"]  # original role string kept
    assert [_ext(s)["source_ref"] for s in run.steps] == [0, 1, 2, 4, 6]  # turn indexes, observations skipped
    assert all(_ext(s)["source_fields"]["mask"] is (s["source"] == "agent") for s in run.steps)
    assert all("tool_calls" not in s and "calls" not in _ext(s) for s in run.steps)
    two_fences = run.steps[3]["message"]  # both fenced blocks stay in the one message, unsplit
    assert two_fences.count("```") == 4 and "reasoning_content" not in run.steps[3]
    assert run.record["agent"] == {"name": "SWE-agent", "version": "unknown", "model_name": "swe-agent-llama-70b"}
    assert run.tools == {} and run.agents == []


def test_later_user_turns_are_observation_results_without_call_ids():
    run = _runs()[0]
    first, second, last = run.steps[2:]
    for step, turn in ((first, 3), (second, 5)):
        (result,) = step["observation"]["results"]
        assert result["source_call_id"] is None and result["content"].endswith(PAD)
        assert _ext(step)["result_refs"] == [turn]
        assert _ext(step)["result_fields"] == [{"mask": False, "role": "user"}]  # the original role is kept
    assert "observation" not in last  # the final agent turn gets no observation
    assert [(e.kind, e.role) for e in run.events()] == [
        ("message", "system"), ("message", "user"), ("message", "agent"), ("result", "tool"),
        ("message", "agent"), ("result", "tool"), ("message", "agent")]


def test_user_only_row_has_no_agent_step():
    run = _runs()[1]
    assert [s["source"] for s in run.steps] == ["system", "user"]
    assert all("observation" not in s for s in run.steps)
    assert run.task["text"].startswith("Synthetic issue with no agent turn")


def test_empty_trajectory_is_marked_not_invented():
    run = _runs()[2]
    (step,) = run.steps
    assert step["message"] == "" and _ext(step) == {"message_state": "missing", "empty_source": True}
    assert run.task == {"id": "synth__repo-3", "text": None} and run.labels == []  # exit_status null: no label


def test_odd_turns_keep_what_the_source_gave():
    run = _runs()[3]
    assert [s["source"] for s in run.steps] == ["system", "user", "user", "agent", "agent"]
    assert (run.steps[0]["message"], _ext(run.steps[0])["message_state"]) == ("", "null")  # null system_prompt
    assert run.steps[2]["message"].startswith("A second user turn") and "role" not in _ext(run.steps[2])
    assert (run.steps[3]["message"], _ext(run.steps[3])["message_state"]) == ("", "null")  # null ai text
    # a non-null cutoff_date or system_prompt on a non-system turn is kept, not dropped
    assert _ext(run.steps[3])["result_fields"] == [
        {"cutoff_date": "02.02.2023", "mask": False, "system_prompt": "stray", "role": "user"}]
    assert run.task["text"].startswith("Synthetic issue:")  # the first user turn, not the second


def test_task_outcome_label_and_row_fields():
    a, b, c, d = _runs()
    assert a.task["id"] == "synth__repo-1" and a.task["text"] == a.steps[1]["message"]
    assert a.outcome == {"success": True, "score": None, "raw": {"target": True}}
    assert b.outcome["success"] is False and d.outcome["raw"] == {"target": False}
    (label,) = a.labels
    assert label == {"name": "exit_status", "target": {"kind": "run", "address": {"trajectory": "train-00000-of-00012:0"}},
                     "value": "submitted", "label_source": "swe_agent_nebius", "source_ref": {"field": "exit_status"}}
    assert d.labels[0]["value"] == "submitted (exit_context)"
    assert a.dataset == {"instance_id": "synth__repo-1", "model_name": "swe-agent-llama-70b", "target": True,
                         "exit_status": "submitted", "generated_patch": "diff --git a/parse.py b/parse.py\n",
                         "eval_logs": "PASSED"}
    assert b.dataset["generated_patch"] is None and c.dataset["generated_patch"] == ""  # null and empty stay distinct
    assert a.id != d.id and a.task["id"] == d.task["id"]  # one instance, two runs: the run id is shard and row


def test_unexpected_role_and_options_are_refused():
    raw = {"instance_id": "x", "trajectory": [{"role": "assistant", "mask": True, "text": "hi"}]}
    with pytest.raises(ValueError, match="unexpected turn role 'assistant'"):
        mod.dataset.convert("k", raw, {"file": "f", "config": "default", "split": "train"})
    with pytest.raises(TypeError, match="takes no options"):
        _runs(subset="x")


def test_atif_part_validates_against_harbor():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / "swe_agent_nebius") if data else None
    runs = steps = results = 0
    seen = set()
    for run in ap.load("swe_agent_nebius", root=root, verify=True):  # streamed: 80,036 runs do not fit in memory at once
        runs += 1
        steps += len(run.steps)
        results += sum(len(s.get("observation", {}).get("results", ())) for s in run.steps)
        seen.add(run.id)
        if runs % 400 == 1:  # a deterministic sample; the full structural pass is in the unit's verification log
            assert ap.validate(run) == []
            assert run.steps[0]["source"] == "system" and run.record["agent"]["name"] == "SWE-agent"
    assert runs == 80036 and len(seen) == 80036
    # 80,036 system + 80,036 issue + 2,115,623 ai turns are steps; the other 2,035,588 user turns are results
    assert steps == 2275695 and results == 2035588
