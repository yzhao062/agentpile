"""Tests for the Trace Commons dataset loader.

Uses synthetic fixtures under tests/corpora/fixtures/trace_commons/.
"""

import os
from pathlib import Path

import pytest

import agentpile as ap

ROOT = Path(__file__).parent / "fixtures" / "trace_commons"


def _runs(**kw):
    return list(ap.load("trace_commons", root=str(ROOT), **kw))


def _by_id():
    return {r.id: r for r in _runs()}


def test_sessions_become_runs_in_file_order_with_file_stem_ids():
    runs = _runs()
    assert [r.id for r in runs] == ["session-tc-01", "session-tc-02"]
    assert len({r.id for r in runs}) == len(runs)
    assert all(ap.validate(r) == [] for r in runs)


def test_cursor_folder_is_not_read():
    runs = _runs()
    ids = [r.id for r in runs]
    assert "cursor-ignored" not in ids
    assert all("cursor" not in r.source.get("file", "") for r in runs)
    assert len(runs) == 2


def test_source_fields_and_manifest_revision():
    runs = _runs()
    src1 = runs[0].source
    assert src1["file"] == "sessions/claude_code/session-tc-01.jsonl"
    assert src1["config"] == "claude_code"
    assert src1["split"] is None
    assert src1["record_key"] == "session-tc-01"
    assert src1["format"] == "claude-code-jsonl"
    assert src1["reader"] == "trace_commons"
    assert src1["dataset"] == ap.datasets.get("trace_commons").repo_id
    assert src1["reader_version"] == "1"
    assert src1["revision"] == ap.datasets.get("trace_commons").revision

    src2 = runs[1].source
    assert src2["file"] == "sessions/claude_code/session-tc-02.jsonl"
    assert src2["config"] == "claude_code"
    assert src2["split"] is None
    assert src2["record_key"] == "session-tc-02"


def test_trace_commons_takes_no_options():
    with pytest.raises(TypeError, match="trace_commons takes no options"):
        _runs(split="train")
    with pytest.raises(TypeError, match="trace_commons takes no options"):
        _runs(unsupported_option=True)


def test_trajectory_records_and_tools():
    by_id = _by_id()
    r1 = by_id["session-tc-01"]
    assert r1.record["agent"]["name"] == "claude-code"
    assert r1.record["agent"]["version"] == "2.1.170"
    assert [s["source"] for s in r1.steps] == ["user", "agent"]

    r2 = by_id["session-tc-02"]
    assert [s["source"] for s in r2.steps] == ["user", "agent", "agent"]
    call_step = r2.steps[1]
    assert len(call_step.get("tool_calls", [])) == 1
    call = call_step["tool_calls"][0]
    assert call["function_name"] == "Bash"
    assert call["tool_call_id"] == "toolu_tc_01"
    assert call_step["observation"]["results"][0]["source_call_id"] == "toolu_tc_01"


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    runs = list(ap.load("trace_commons", root=str(Path(data) / "trace_commons") if data else None, verify=True))
    assert len(runs) == 28 and len({r.id for r in runs}) == 28
    assert all(ap.validate(r) == [] for r in runs)
    total = {k: sum(r.dataset["accounting"][k] for r in runs) for k in (
        "lines", "mapped", "attachment", "counted", "calls", "results", "paired", "calls_without_result", "unpaired_results")}
    # the independent census of the pinned files (plan section 2) counts the same
    assert total == {"lines": 17991, "mapped": 12225, "attachment": 1138, "counted": 4628, "calls": 4264,
                     "results": 4262, "paired": 4262, "calls_without_result": 2, "unpaired_results": 0}
    assert sum(len(r.dataset["nodes"]["rows"]) for r in runs) == 13500
    # no subagent file was published: the 9 Agent results that name one report it missing
    assert sum(len(r.dataset.get("subagents", {}).get("missing_agent_ids", [])) for r in runs) == 9
    fields = [f for r in runs for s in r.steps for f in s["extra"]["agentpile"].get("result_fields") or []]
    assert len(fields) == 4262 and all("uuid" in f and "parentUuid" in f for f in fields)
    usage = [s for r in runs for s in r.steps if "metrics" in s]
    assert len(usage) == 4354 and all("usage" in s["extra"]["agentpile"]["source_fields"] for s in usage)
