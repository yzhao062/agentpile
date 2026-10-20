"""CatchBench PRE as config records, and gate 3: read, serialize, re-read and invert back to the row."""

import json
import os
from pathlib import Path

import pytest

import agentpile as ap

ROOT = Path(__file__).parent / "fixtures" / "catchbench_pre"


def to_pre_row(run):
    """Inverse projection, test-side: rebuild a CatchBench PRE row from the shared view.

    Capabilities, permissions, labels and identity come from the view; only what
    the view has no field for (provenance, spec tokens, judge metadata) comes
    from the raw row kept in ``dataset``.
    """
    raw = run.dataset
    by_name = {l["name"]: l for l in run.labels if l["target"]["kind"] == "run"}
    excess = [l["target"]["address"]["tool"] for l in run.labels if l["name"] == "excess_capability"]
    labels = {"excess_set": excess, "label_source": raw["labels"]["label_source"]}
    for key in raw["labels"]:
        if key not in labels:
            labels[key] = by_name[key]["value"] if key in by_name else raw["labels"][key]
    return {
        "instance_id": run.task["id"],
        "source": run.source["config"],
        "provenance": raw["provenance"],
        "declared_capabilities": [{"name": t["name"], "type": t["type"], "permission_level": t["permission"]}
                                  for t in run.tools.values()],
        "minimal_reference": by_name["minimal_reference"]["value"],
        "labels": labels,
        "spec_tokens": raw["spec_tokens"],
        "spec_token_overrides": raw["spec_token_overrides"],
    }


def _original_rows(root):
    rows = []
    for corpus in ("crewai", "injecagent", "mcp", "n8n", "sweagent", "synthetic"):
        rows.extend(json.loads((Path(root) / "data" / "pre" / f"{corpus}.json").read_text()))
    return rows


def _gate3(runs, rows, tmp_path):
    assert len(runs) == len(rows)
    out = tmp_path / "pre.jsonl"
    out.write_text("".join(r.to_json() + "\n" for r in runs))
    reread = list(ap.load(out))
    for run, again, row in zip(runs, reread, rows):
        assert again == run and again.kind == "config"
        assert ap.validate(again) == []
        assert to_pre_row(again) == row
        assert {t["name"]: t["permission"] for t in again.tools.values()} == {
            c["name"]: c["permission_level"] for c in row["declared_capabilities"]}


def test_rows_become_config_records():
    runs = list(ap.load("catchbench_pre", root=str(ROOT)))
    assert [r.id for r in runs] == ["crewai-x1", "injecagent-x1", "mcp-x1", "sweagent-x1", "synthetic-x1"]
    crew = runs[0]
    assert crew.kind == "config" and crew.steps == [] and list(crew.events()) == []
    assert crew.tools["FileWriter"] == {"name": "FileWriter", "agent_or_server": "agent", "type": "file",
                                        "definition": None, "definition_format": None, "atif_index": None,
                                        "permission": "write", "permission_from": "declared"}
    assert crew.agents[0]["instructions"] is None  # spec prose is not in the committed rows
    excess = [l for l in crew.labels if l["name"] == "excess_capability"]
    assert excess[0]["target"] == {"kind": "tool", "address": {"trajectory": "crewai-x1", "tool": "FileWriter"}}
    assert [l["value"] for l in runs[1].labels if l["name"] == "attack_type"] == ["direct harm"]


def test_gate3_on_fixture(tmp_path):
    _gate3(list(ap.load("catchbench_pre", root=str(ROOT))), _original_rows(ROOT), tmp_path)


@pytest.mark.network
def test_gate3_on_pinned_rows(tmp_path):
    data = os.environ.get("AGENTPILE_DATA")
    ds = ap.datasets.get("catchbench_pre")
    root = Path(data) / "catchbench_pre" if data else ds.files()
    runs = list(ds.load(root=str(root), verify=True))
    assert len(runs) == 1187
    _gate3(runs, _original_rows(root), tmp_path)
