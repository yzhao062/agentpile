"""Who&When loader on a synthetic copy of the pinned file layout (no dataset content)."""

import os
from pathlib import Path

import pytest

import agentpile as ap

ROOT = Path(__file__).parent / "fixtures" / "who_and_when"


def _runs(**kw):
    return list(ap.load("who_and_when", root=str(ROOT), **kw))


def test_files_become_runs_in_numeric_order():
    assert [r.id for r in _runs()] == ["Algorithm-Generated/1", "Algorithm-Generated/2", "Hand-Crafted/1"]
    assert all(ap.validate(r) == [] for r in _runs())
    assert [r.id for r in _runs(subset="Hand-Crafted")] == ["Hand-Crafted/1"]


def test_algorithm_generated_steps_are_agent_steps_named_by_name():
    run = _runs()[0]
    assert [s["source"] for s in run.steps] == ["agent"] * 5
    assert [s["extra"]["agentpile"]["actor"] for s in run.steps] == [
        "Math_Expert", "Coder_Expert", "Computer_terminal", "Math_Expert", "Math_Expert"]
    assert "tool_calls" not in run.steps[2]  # no call is recorded, so none is invented
    assert run.steps[3]["message"] == "" and "message_state" not in run.steps[3]["extra"]["agentpile"]
    assert {a["id"]: a["instructions"] for a in run.agents}["Coder_Expert"] == "You write code."


def test_mistake_step_is_zero_based_over_history():
    labels = {l["name"]: l for l in _runs()[0].labels}
    assert labels["mistake_step"]["target"]["address"]["step"] == 5  # mistake_step "4" -> fifth entry
    assert _runs()[1].labels[0]["target"]["address"]["step"] == 1  # "0" -> first entry
    assert labels["mistake_reason"]["target"] == labels["mistake_step"]["target"]


def test_hand_crafted_roles_and_label_spelling():
    run = _runs()[2]
    assert run.steps[0]["source"] == "user"
    assert [s["extra"]["agentpile"].get("actor") for s in run.steps] == [
        None, "Orchestrator", "Orchestrator", "WebSurfer", "Orchestrator"]
    assert run.steps[2]["extra"]["agentpile"]["role"] == "Orchestrator (-> WebSurfer)"
    labels = {l["name"]: l for l in run.labels}
    assert labels["mistake_agent"]["target"]["address"]["agent"] == "WebSurfer"
    assert labels["mistake_agent"]["source_ref"] == {"field": "mistake_agent", "value": "Websurfer", "resolved": True}
    assert labels["mistake_type"]["target"]["kind"] == "run"


def test_outcome_task_and_raw_fields():
    a, _, h = _runs()
    assert a.outcome["success"] is False and a.outcome["raw"] == {"is_correct": False}
    assert h.outcome["raw"] == {"is_corrected": False}
    assert a.task == {"id": "q-synth-1", "text": "What is 2+2 in base 3?"}
    assert h.dataset["groundtruth"] == "Shop A" and "history" not in h.dataset
    assert a.source["revision"] == ap.datasets.get("who_and_when").revision


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / "who_and_when") if data else None
    runs = list(ap.load("who_and_when", root=root, verify=True))
    assert len(runs) == 184 and sum(len(r.steps) for r in runs) == 4092
    assert all(ap.validate(r) == [] for r in runs)
