"""AgentDojo loader on a synthetic copy of the pinned file layout (no dataset content)."""

import os
from pathlib import Path

import pytest

import agentpile as ap

ROOT = Path(__file__).parent / "fixtures" / "agentdojo"


def _runs(**kw):
    return list(ap.load("agentdojo", root=str(ROOT), **kw))


def test_files_become_runs_by_path():
    runs = _runs()
    assert [r.id for r in runs] == [
        "model-a/banking/user_task_0/important_instructions/injection_task_0",
        "model-a/banking/user_task_0/none/none",
        "model-a/slack/user_task_1/important_instructions/injection_task_0"]
    assert all(ap.validate(r) == [] for r in runs)
    assert [r.id for r in _runs(suite="slack")] == [runs[2].id]


def test_parallel_calls_pair_by_id_and_errors_survive():
    run = _runs()[0]
    step = run.steps[2]
    assert [c["tool_call_id"] for c in step["tool_calls"]] == ["t-1", "t-2"]
    assert "placeholder_args" not in str(step.get("extra"))
    assert [r["source_call_id"] for r in step["observation"]["results"]] == ["t-1", "t-2"]
    fields = step["extra"]["agentpile"]["result_fields"]
    assert fields[0] == {"content_form": "blocks", "tool_call_copy": "identical"}
    errors = [e.error for e in run.events() if e.kind == "result"]
    assert errors == [None, "ValueError: no account (synthetic)"]
    assert step["extra"]["agentpile"]["message_state"] == "null"
    assert run.steps[0]["message"] == "You help with banking (synthetic)."


def test_labels_and_placeholder_security():
    attack, plain, unlabelled = _runs()
    by = {l["name"]: l for l in attack.labels}
    assert by["utility"]["value"] is True and by["security"]["value"] is False
    assert "placeholder" not in by["security"]["source_ref"]
    sec = [l for l in plain.labels if l["name"] == "security"][0]
    assert sec["value"] is True and sec["source_ref"]["placeholder"] is True
    assert not any(l["name"] in ("utility", "security") for l in unlabelled.labels)
    assert attack.outcome["success"] is True and unlabelled.outcome["success"] is None
    assert attack.record["agent"]["model_name"] == "model-a"
    assert attack.dataset["injections"] == {"injection_bill_text": "IGNORE PREVIOUS (synthetic)"}


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    runs = list(ap.load("agentdojo", root=str(Path(data) / "agentdojo") if data else None, verify=True))
    assert len(runs) == 15095
    assert sum(len(s.get("tool_calls") or []) for r in runs for s in r.steps) == 39367
