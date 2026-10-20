"""MAST loader on a synthetic copy of the pinned file layout (no dataset content)."""

import os
from pathlib import Path

import pytest

import agentpile as ap

ROOT = Path(__file__).parent / "fixtures" / "mast"


def _runs():
    return list(ap.load("mast", root=str(ROOT)))


def test_one_raw_log_step_per_row():
    runs = _runs()
    assert [r.id for r in runs] == ["AG2_synth:0", "ChatDev_synth:0"]
    step = runs[0].steps[0]
    assert len(runs[0].steps) == 1 and step["source"] == "agent" and "tool_calls" not in step
    assert step["extra"]["agentpile"]["raw_log"] is True
    assert step["message"].startswith("{'content'")
    assert all(ap.validate(r) == [] for r in runs)


def test_framework_and_model_in_agent_block():
    assert _runs()[0].record["agent"] == {"name": "AG2", "version": "unknown", "model_name": "GPT-4o"}


def test_labels_are_run_level_and_nulls_stay_null():
    a, c = _runs()
    by = {l["name"]: l for l in a.labels}
    assert len(a.labels) == 14 and by["1.3"]["value"] == 1 and by["1.1"]["value"] == 0
    assert by["1.3"]["target"] == {"kind": "run", "address": {"trajectory": "AG2_synth:0"}}
    assert by["1.3"]["label_source"] == "llm_judge" and by["1.3"]["source_ref"]["mode"] == "Step Repetition"
    assert all(l["value"] is None for l in c.labels)


def test_row_fields_kept_without_the_log():
    d = _runs()[0].dataset
    assert d["trace"] == {"key": "AG2_synth", "index": 0} and d["benchmark_name"] == "GSM"


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    runs = list(ap.load("mast", root=str(Path(data) / "mast") if data else None, verify=True))
    assert len(runs) == 1642 and len({r.id for r in runs}) == 1642
