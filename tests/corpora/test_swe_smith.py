"""SWE-smith loader on a synthetic copy of the pinned file layout (no dataset content)."""

import os
import sys
from pathlib import Path

import pytest

import agentpile as ap

pytest.importorskip("pyarrow")

ROOT = Path(__file__).parent / "fixtures" / "swe_smith"
A = "synth__repo-a.0000000.pr_1.aaaa1111"
B = "synth__repo-b.0000000.pr_2.bbbb2222"
C = "synth__repo-c.0000000.pr_3.cccc3333"
D = "synth__repo-d.0000000.pr_4.dddd4444"
E = "synth__repo-e.0000000.pr_5.eeee5555"
F = "synth__repo-f.0000000.pr_6.ffff6666"
G = "data/tool-00001-of-00008.parquet#3"  # the row without a traj_id is keyed by file and row


def _runs(**kw):
    return list(ap.load("swe_smith", root=str(ROOT), **kw))


def _by_id():
    return {r.id: r for r in _runs()}


def test_every_row_of_every_shard_becomes_a_run_in_order():
    runs = _runs()
    assert [r.id for r in runs] == [A, B, C, D, E, F, G]
    assert all(ap.validate(r) == [] for r in runs)
    src = runs[3].source
    assert src["file"] == "data/tool-00001-of-00008.parquet" and src["row"] == 0 and src["split"] == "tool"
    assert src["config"] == "default" and src["record_key"] == runs[3].id
    assert src["revision"] == ap.datasets.get("swe_smith").revision
    assert runs[6].source["row"] == 3 and runs[6].source["record_key"] == G


def test_native_calls_and_results_pair_by_id():
    run = _by_id()[A]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "agent"]
    first = run.steps[2]
    assert first["tool_calls"] == [{"tool_call_id": "toolu_synth_1", "function_name": "str_replace_editor",
                                    "arguments": {"command": "view", "path": "/testbed/a.py"}}]
    assert first["observation"]["results"] == [{"source_call_id": "toolu_synth_1", "content": "OBSERVATION:\nfile a"}]
    sx = first["extra"]["agentpile"]
    assert sx["actor"] == "main" and sx["result_refs"] == [3]
    assert sx["calls"]["toolu_synth_1"] == {"source_fields": {"index": 1}}  # provider artifact, kept
    assert sx["source_fields"]["thought"] == "Look at the file." and sx["source_fields"]["message_type"] == "action"
    # tool_call_ids (a list) is the source's own key; it stays on the result next to the other tool keys
    assert sx["result_fields"][0]["tool_call_ids"] == ["toolu_synth_1"]
    assert sx["result_fields"][0]["cache_control"] == {"type": "ephemeral"}
    second = run.steps[3]
    assert second["message"] == "" and "message_state" not in second["extra"]["agentpile"]  # '' is the source value
    assert second["observation"]["results"][0]["content"] == "OBSERVATION:\nlisting"  # a plain string stays one
    assert "content_form" not in second["extra"]["agentpile"]["result_fields"][0]
    last = run.steps[4]  # the final submit call has no result
    assert last["tool_calls"][0]["function_name"] == "submit" and last["tool_calls"][0]["arguments"] == {}
    assert "observation" not in last


def test_events_follow_the_call_pairing():
    events = list(_by_id()[A].events())
    assert [e.kind for e in events if e.address["step"] == 3] == ["message", "call", "result"]
    result = next(e for e in events if e.kind == "result")
    assert result.call_id == "toolu_synth_1" and result.content == "OBSERVATION:\nfile a"


def test_text_parts_are_joined_and_marked_other_lists_pass_through():
    runs = _by_id()
    user = runs[A].steps[1]
    assert user["message"] == "Fix the synthetic bug."
    assert user["extra"]["agentpile"]["source_fields"]["content_form"] == "text_parts"
    assert runs[A].steps[2]["extra"]["agentpile"]["result_fields"][0]["content_form"] == "text_parts"
    assert runs[B].steps[1]["message"] == "Plain string task."  # a string in the source, nothing to mark
    assert "content_form" not in runs[B].steps[1]["extra"]["agentpile"]["source_fields"]
    two = runs[G].steps[1]["message"]  # two parts are not joined
    assert two == [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]
    assert runs[G].task["text"] is None  # task text is only taken from a string


def test_final_assistant_text_without_a_call():
    last = _by_id()[B].steps[-1]
    assert last["source"] == "agent" and last["message"] == "I could not finish."
    assert "tool_calls" not in last and last["extra"]["agentpile"]["source_fields"]["action"] == ""


def test_ids_that_do_not_pair_stay_unpaired():
    run = _by_id()[G]
    results = [r for s in run.steps for r in (s.get("observation") or {}).get("results", [])]
    assert [r["source_call_id"] for r in results] == [None, None]  # two ids in one list, then an id with no call
    fields = [f for s in run.steps for f in s["extra"]["agentpile"].get("result_fields", [])]
    assert fields[0]["tool_call_ids"] == ["toolu_synth_g1", "toolu_synth_g9"] and "tool_call_id" not in fields[0]
    assert fields[1]["tool_call_id"] == "toolu_nomatch"


def test_text_rows_load_as_plain_messages():
    run = _by_id()[C]
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "user", "agent"]
    assert all("tool_calls" not in s and "observation" not in s for s in run.steps)
    assert run.steps[2]["message"].startswith("Listing.\n<function=bash>")  # the call text is not parsed
    assert run.steps[3]["message"] == "OBSERVATION:\nfile.py"
    assert all("actor" not in s["extra"]["agentpile"] for s in run.steps) and run.agents == []
    assert run.dataset["message_format"] == "text"


def test_native_and_text_format_flags_and_system_hash():
    runs = _by_id()
    assert [runs[k].dataset["message_format"] for k in (A, B, C, G)] == ["native", "native", "text", "native"]
    assert runs[A].dataset["system_prompt_sha256"] == runs[B].dataset["system_prompt_sha256"]
    assert runs[A].dataset["system_prompt_sha256"] != runs[C].dataset["system_prompt_sha256"]
    assert len(runs[A].dataset["system_prompt_sha256"]) == 64


def test_tools_come_from_the_function_blocks_of_the_system_prompt():
    runs = _by_id()
    assert list(runs[C].tools) == ["bash", "submit"]
    bash = runs[C].tools["bash"]
    assert bash["name"] == "bash" and bash["permission"] == "unknown" and bash["permission_from"] == "unknown"
    assert bash["definition"] is None and bash["atif_index"] is None
    dup = runs[D].tools
    assert list(dup) == ["bash", "bash#2"] and dup["bash#2"]["name"] == "bash"
    assert runs[A].tools == {} and "tool_definitions" not in runs[A].record["agent"]  # native rows define none


def test_run_level_fields():
    run = _by_id()[A]
    assert run.record["agent"] == {"name": "SWE-agent", "version": "unknown", "model_name": "claude-3-7-sonnet-20250219"}
    assert run.task == {"id": "synth__repo-a.0000000.pr_1", "text": "Fix the synthetic bug."}
    assert run.outcome == {"success": True, "score": None, "raw": {"resolved": True}}
    assert run.labels == [{"name": "resolved", "target": {"kind": "run", "address": {"trajectory": A}}, "value": True,
                           "label_source": "swe_smith", "source_ref": {"field": "resolved"}}]
    assert [a["id"] for a in run.agents] == ["main"]
    d = run.dataset
    assert d["patch"] == "diff --git a/x b/x\n" and d["traj_id"] == A and d["instance_id"] == run.task["id"]
    assert "messages" not in d and "not this run's output" in d["patch_note"]
    assert _by_id()[B].outcome["success"] is False and _by_id()[B].record["agent"]["model_name"] == "gpt-4o-2024-08-06"


def test_empty_and_undecodable_messages_still_become_runs():
    runs = _by_id()
    empty, bad = runs[E], runs[F]
    for run in (empty, bad):
        assert len(run.steps) == 1 and run.steps[0]["extra"]["agentpile"]["empty_source"] is True
        assert run.dataset["message_format"] is None and run.dataset["system_prompt_sha256"] is None
    assert "messages_raw" not in empty.dataset and bad.dataset["messages_raw"] == "this is not json"
    assert "model_name" not in empty.record["agent"] and empty.dataset["model"] is None  # a null model stays null
    assert empty.task["text"] is None


def test_message_format_selects_one_format():
    assert [r.id for r in _runs(message_format="text")] == [C, D]
    assert [r.id for r in _runs(message_format="native")] == [A, B, G]
    with pytest.raises(ValueError, match="message_format"):
        _runs(message_format="xml")
    with pytest.raises(TypeError, match="message_format"):
        _runs(split="xml")


def test_missing_pyarrow_is_reported_with_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)  # makes the import raise ImportError
    with pytest.raises(ImportError, match="agentpile\\[hf\\]"):
        _runs()


def test_harbor_accepts_the_records():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / "swe_smith") if data else None
    n = steps = calls = results = unpaired = 0
    formats = {"native": 0, "text": 0}
    ids = set()
    for run in ap.load("swe_smith", root=root, verify=True):
        n += 1
        steps += len(run.steps)
        formats[run.dataset["message_format"]] += 1
        ids.add(run.id)
        for step in run.steps:
            calls += len(step.get("tool_calls") or [])
            for res in (step.get("observation") or {}).get("results", []):
                results += 1
                unpaired += res["source_call_id"] is None
        assert ap.validate(run) == []
    assert n == 24100 and len(ids) == 24100
    assert formats == {"native": 9116, "text": 14984}
    assert steps == 1277278 and calls == 277520 and results == 269242 and unpaired == 0
