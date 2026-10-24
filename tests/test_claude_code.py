"""Tests for Claude Code session file reader (claude-code-jsonl).

Covers synthetic fixtures under tests/fixtures/claude_code/, validation,
downstream views (OTLP, chat, table, show), two-way detection, and negative inputs.
"""

import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats import claude_code
FIX = Path(__file__).parent / "fixtures" / "claude_code"
_spec = importlib.util.spec_from_file_location("claude_code_expected", FIX / "expected.py")
_expected = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_expected)
EXPECTED = _expected.EXPECTED
_spec = importlib.util.spec_from_file_location("claude_code_make_fixtures", FIX / "make_fixtures.py")
_make = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_make)


# -- Fixture loading and expectations -----------------------------------------

@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_loads_and_matches_core_accounting(name):
    exp = EXPECTED[name]
    path = FIX / f"{name}.jsonl"
    run = next(ap.load(path))
    acc = run.dataset["accounting"]
    exp_acc = exp["accounting"]

    for key in ("lines", "mapped", "attachment", "counted", "exact_duplicate",
                "unparseable", "calls", "results", "paired",
                "unpaired_results", "conflicting_records", "conflicting_calls"):
        assert acc.get(key) == exp_acc[key], f"{name} {key}: expected {exp_acc[key]}, got {acc.get(key)}"

    assert acc.get("counted_by_type") == exp_acc.get("counted_by_type")


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_calls_without_result(name):
    exp = EXPECTED[name]
    path = FIX / f"{name}.jsonl"
    run = next(ap.load(path))
    acc = run.dataset["accounting"]
    expected_calls_without_result = exp["accounting"]["calls_without_result"]

    assert acc["calls_without_result"] == expected_calls_without_result


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_blocks_accounting(name):
    exp = EXPECTED[name]
    path = FIX / f"{name}.jsonl"
    run = next(ap.load(path))
    acc = run.dataset["accounting"]
    exp_blocks = exp["accounting"].get("blocks", {})

    assert acc.get("blocks") == exp_blocks


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_events_order_and_call_ids(name):
    exp = EXPECTED[name]
    path = FIX / f"{name}.jsonl"
    run = next(ap.load(path))
    events = list(run.events())
    actual_kinds = [e.kind for e in events]
    actual_call_ids = [e.call_id for e in events]

    assert actual_kinds == exp["event_kinds"]
    assert actual_call_ids == exp["event_call_ids"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_branches(name):
    exp = EXPECTED[name]
    path = FIX / f"{name}.jsonl"
    run = next(ap.load(path))
    branches = run.dataset.get("branches", [])
    actual_labels = [b["label"] for b in branches]
    assert actual_labels == exp.get("branch_labels", [])
    if "branches" in exp:
        assert branches == exp["branches"]


# -- Specific fixture scenarios -----------------------------------------------

def test_split_interleaved_preserves_event_order():
    run = next(ap.load(FIX / "split_interleaved.jsonl"))
    events = list(run.events())
    assert [e.kind for e in events] == ["message", "message", "call", "call", "result", "result"]
    assert [e.call_id for e in events] == [None, None, "toolu_fx_a1", "toolu_fx_a2", "toolu_fx_a1", "toolu_fx_a2"]
    # Check that each call and result keeps its source ref
    step = run.steps[1]
    assert step["extra"]["agentpile"]["calls"]["toolu_fx_a1"]["source_ref"] == 4
    assert step["extra"]["agentpile"]["calls"]["toolu_fx_a2"]["source_ref"] == 6
    assert step["extra"]["agentpile"]["result_refs"] == [5, 7]


def test_early_result_pairs_with_later_call_record():
    run = next(ap.load(FIX / "early_result.jsonl"))
    step = run.steps[1]
    assert len(step.get("tool_calls", [])) == 1
    assert step["tool_calls"][0]["tool_call_id"] == "toolu_fx_b1"
    assert step["observation"]["results"][0]["source_call_id"] == "toolu_fx_b1"
    assert run.dataset["accounting"]["paired"] == 1


def test_edit_branch_labels_two_user_texts():
    run = next(ap.load(FIX / "edit_branch.jsonl"))
    branches = run.dataset.get("branches", [])
    assert len(branches) == 1
    assert branches[0]["label"] == "user texts"
    assert len(branches[0]["children"]) == 2


def test_compaction_logical_parent_resolves_in_node_index():
    run = next(ap.load(FIX / "compaction.jsonl"))
    rows = run.dataset["nodes"]["rows"]
    fields = run.dataset["nodes"]["fields"]
    uuid_idx = fields.index("uuid")
    log_parent_idx = fields.index("logicalParentUuid")

    all_uuids = {r[uuid_idx] for r in rows if r[uuid_idx] is not None}
    compact_row = next(r for r in rows if r[log_parent_idx] is not None)
    log_parent = compact_row[log_parent_idx]
    assert log_parent == EXPECTED["compaction"]["logical_parent_uuid"]
    assert log_parent in all_uuids


def test_api_error_branch_labeled_assistant_and_api_error():
    run = next(ap.load(FIX / "api_error.jsonl"))
    branches = run.dataset.get("branches", [])
    assert len(branches) == 1
    assert branches[0]["label"] == "assistant and api_error"


def test_images_follow_part_rule_with_stand_in_and_raw():
    run = next(ap.load(FIX / "images.jsonl"))
    user_step = run.steps[0]
    assert isinstance(user_step["message"], list)
    assert user_step["message"][1]["text"] == "[image part: original in agentpile extension]"
    assert "message_raw" in user_step["extra"]["agentpile"]

    agent_step = run.steps[1]
    res = agent_step["observation"]["results"][0]
    assert isinstance(res["content"], list)
    assert res["content"][1]["text"] == "[image part: original in agentpile extension]"
    assert "content_raw" in agent_step["extra"]["agentpile"]["result_fields"][0]


def test_unknown_record_and_block_types():
    run = next(ap.load(FIX / "unknown.jsonl"))
    acc = run.dataset["accounting"]
    assert acc["counted_by_type"]["future-widget-event"] == 1
    agent_step = run.steps[1]
    assert agent_step["message"][0]["text"] == "[future_gadget part: original in agentpile extension]"
    assert agent_step["extra"]["agentpile"]["message_raw"][0]["type"] == "future_gadget"


def test_duplicates_exact_and_conflicting_calls():
    run = next(ap.load(FIX / "duplicates.jsonl"))
    acc = run.dataset["accounting"]
    assert acc["exact_duplicate"] == 1
    assert acc["conflicting_records"] == 1
    assert acc["conflicting_calls"] == 1

    agent_step = run.steps[3]
    gen_call = agent_step["tool_calls"][0]
    assert gen_call["tool_call_id"].startswith("ap-")
    cx = agent_step["extra"]["agentpile"]["calls"][gen_call["tool_call_id"]]
    assert cx["source_id"] == "toolu_fx_reuse"
    assert cx["id_generated"] is True


def test_unpaired_results_and_unanswered_calls():
    run = next(ap.load(FIX / "unpaired.jsonl"))
    acc = run.dataset["accounting"]
    assert acc["unpaired_results"] == 2
    assert acc["calls_without_result"] == 1

    first_step = run.steps[0]
    assert first_step["source"] == "user"
    assert first_step["extra"]["agentpile"]["role"] == "tool"

    agent_step = run.steps[2]
    res = agent_step["observation"]["results"][0]
    assert res["source_call_id"] is None
    rf = agent_step["extra"]["agentpile"]["result_fields"][0]
    assert rf["tool_call_id"] == "toolu_fx_ghost"


def test_attachment_beside_result_reports_no_branch():
    run = next(ap.load(FIX / "attachment_beside_result.jsonl"))
    assert run.dataset.get("branches", []) == []
    assert len(run.dataset["attachments"]) == 1
    assert run.dataset["attachments"][0]["record"]["type"] == "attachment"


def test_subagents_linking_nesting_and_snapshots(tmp_path):
    # the sidecar files Claude Code writes beside subagent files are counted, not read
    shutil.copy(FIX / "subagents.jsonl", tmp_path / "subagents.jsonl")
    shutil.copytree(FIX / "subagents", tmp_path / "subagents")
    for rel, records in _make.SIDECARS.items():
        (tmp_path / "subagents" / rel).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    run = next(ap.load(tmp_path / "subagents.jsonl"))
    sub_meta = run.dataset["subagents"]
    exp_sub = EXPECTED["subagents"]["subagents"]

    assert sub_meta["files"] == exp_sub["files"]
    assert sub_meta["missing_agent_ids"] == exp_sub["missing_agent_ids"]
    assert sub_meta["missing_runs"] == exp_sub["missing_runs"]
    assert sub_meta["unreferenced"] == exp_sub["unreferenced"]
    assert sub_meta["other_files"] == exp_sub["other_files"]

    root_subs = [s["trajectory_id"] for s in run.record.get("subagent_trajectories", [])]
    assert root_subs == EXPECTED["subagents"]["root_subagent_ids"]

    parent_sub = next(s for s in run.record["subagent_trajectories"] if s["trajectory_id"] == "afx0000000000003")
    nested = [s["trajectory_id"] for s in parent_sub.get("subagent_trajectories", [])]
    assert nested == EXPECTED["subagents"]["nested_subagents"]["afx0000000000003"]

    metrics = parent_sub["steps"][1]["metrics"]
    assert metrics["prompt_tokens"] == EXPECTED["subagents"]["usage_snapshots"]["prompt_tokens"]
    assert metrics["completion_tokens"] == EXPECTED["subagents"]["usage_snapshots"]["completion_tokens"]
    # the raw usage of the last record stays with the source fields, and ATIF metrics hold only numbers
    usage = parent_sub["steps"][1]["extra"]["agentpile"]["source_fields"]["usage"]
    assert usage["output_tokens"] == EXPECTED["subagents"]["usage_snapshots"]["completion_tokens"]
    assert "extra" not in metrics
    assert sub_meta["ambiguous_agent_ids"] == []


def _session(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_links_reported_without_a_subagents_folder(tmp_path):
    sid = "00000000-0000-4000-8000-0000000000a0"
    path = _session(tmp_path / "s.jsonl", [
        _make._user_text(sid, "u1", None, "go"),
        _make._assistant_rec(sid, "u2", "u1", "msg_a", {"type": "tool_use", "id": "toolu_a", "name": "Agent", "input": {}}),
        _make._user_result(sid, "u3", "u2", "toolu_a", "done", tool_use_result={"agentId": "a_absent"}),
        _make._assistant_rec(sid, "u4", "u3", "msg_b", {"type": "tool_use", "id": "toolu_b", "name": "Workflow", "input": {}},
                             ts="2026-10-01T12:00:04.000Z"),
        _make._user_result(sid, "u5", "u4", "toolu_b", "done", ts="2026-10-01T12:00:05.000Z",
                           tool_use_result={"runId": "wf_absent"}),
    ])
    report = next(ap.load(path)).dataset["subagents"]
    assert report["missing_agent_ids"] == ["a_absent"] and report["missing_runs"] == ["wf_absent"]
    assert report["files"] == {} and report["unreferenced"] == []


def test_agent_id_shared_by_two_files_is_not_linked(tmp_path):
    sid = "00000000-0000-4000-8000-0000000000b0"
    path = _session(tmp_path / "s.jsonl", [
        _make._user_text(sid, "u1", None, "go"),
        _make._assistant_rec(sid, "u2", "u1", "msg_a", {"type": "tool_use", "id": "toolu_a", "name": "Agent", "input": {}}),
        _make._user_result(sid, "u3", "u2", "toolu_a", "done", tool_use_result={"agentId": "a_twice"}),
    ])
    for rel in ("subagents/agent-a_twice.jsonl", "subagents/workflows/wf_x/agent-a_twice.jsonl"):
        (tmp_path / "s" / rel).parent.mkdir(parents=True, exist_ok=True)
        _session(tmp_path / "s" / rel, [_make._user_text(sid, "v1", None, "sub", is_sidechain=True, agent_id="a_twice")])
    run = next(ap.load(path))
    report = run.dataset["subagents"]
    assert report["ambiguous_agent_ids"] == ["a_twice"] and report["missing_agent_ids"] == []
    assert sorted(report["unreferenced"]) == ["a_twice", "a_twice'"]
    assert "subagent_trajectory_ref" not in run.steps[1]["observation"]["results"][0]


def test_results_keep_uuid_and_parent():
    for path in sorted(FIX.glob("*.jsonl")):
        run = next(ap.load(path))
        rows = {row[0]: row for row in run.dataset["nodes"]["rows"]}
        for step in run.steps:
            sx = step["extra"]["agentpile"]
            for line, fields in zip(sx.get("result_refs") or [], sx.get("result_fields") or []):
                assert (fields["uuid"], fields["parentUuid"]) == (rows[line][1], rows[line][2]), (path.name, line)


def test_results_before_any_message_keep_their_content_markers(tmp_path):
    sid = "00000000-0000-4000-8000-0000000000c0"
    image = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}]
    absent = _make._user_result(sid, "u3", "u2", "toolu_z", None, ts="2026-10-01T12:00:03.000Z")
    del absent["message"]["content"][0]["content"]
    path = _session(tmp_path / "s.jsonl", [
        _make._user_result(sid, "u1", None, "toolu_x", image, ts="2026-10-01T12:00:01.000Z"),
        _make._user_result(sid, "u2", "u1", "toolu_y", None, ts="2026-10-01T12:00:02.000Z"),
        absent,
    ])
    run = next(ap.load(path))
    assert [s["extra"]["agentpile"]["role"] for s in run.steps] == ["tool", "tool", "tool"]
    assert run.steps[0]["extra"]["agentpile"]["source_fields"]["parentUuid"] is None
    contents = [e.content for e in run.events() if e.kind == "message"]
    assert contents == [image, None]  # the raw image list, a null, and no message for absent content
    assert ap.validate(run.record) == []


# -- Validation and Downstream Exports ----------------------------------------

@pytest.mark.parametrize("path", sorted(FIX.glob("*.jsonl")), ids=lambda p: p.name)
def test_all_fixtures_pass_validation_and_downstream_views(path):
    run = next(ap.load(path))
    assert ap.validate(run.record) == []
    for sub in run.subagents:
        assert ap.validate(sub.record) == []

    # Downstream exports run without exception
    assert isinstance(run.to_otlp(), (str, bytes))
    assert list(ap.to_chat(run)) is not None
    assert ap.table(run, "events") is not None
    buf = io.StringIO()
    ap.show(run, file=buf)
    assert len(buf.getvalue()) > 0


# -- Two-way Detection --------------------------------------------------------

TC_FIX = Path(__file__).parent / "corpora" / "fixtures" / "trace_commons" / "sessions" / "claude_code"


def test_every_claude_code_fixture_recognized_by_exact_reader():
    paths = sorted(FIX.rglob("*.jsonl")) + sorted(TC_FIX.glob("*.jsonl"))
    for p in paths:
        claims = [r for r in ap.readers.names() if ap.readers.get(r).sniff(p)]
        assert claims == ["claude-code-jsonl"], f"{p.name} claimed by {claims}"


def test_no_other_fixtures_claimed_by_claude_code():
    tests = Path(__file__).parent
    other_files = [p for d in (tests / "fixtures", tests / "corpora" / "fixtures") for p in d.rglob("*")
                   if p.is_file() and FIX not in p.parents and TC_FIX not in p.parents]
    assert other_files
    for p in other_files:
        assert not claude_code.sniff(p), f"Non-claude-code fixture {p} unexpectedly sniffed by claude-code-jsonl"


# -- Negative Inputs (Inline with tmp_path) -----------------------------------

def test_negative_bookkeeping_only_file(tmp_path):
    bk_file = tmp_path / "bookkeeping.jsonl"
    lines = [
        '{"type": "queue-operation", "operation": "enqueue", "sessionId": "bk-sess"}\n',
        '{"type": "permission-mode", "permissionMode": "default", "sessionId": "bk-sess"}\n',
        '{"type": "file-history-snapshot", "isSnapshotUpdate": false}\n',
    ] * 20
    bk_file.write_text("".join(lines), encoding="utf-8")

    assert not claude_code.sniff(bk_file)
    with pytest.raises(ap.UnrecognizedSource):
        next(ap.load(bk_file))

    runs = list(ap.load(bk_file, reader="claude-code-jsonl"))
    assert len(runs) == 1
    run = runs[0]
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step["source"] == "system"
    assert step["message"] == ""
    assert step["extra"]["agentpile"]["message_state"] == "missing"
    assert step["extra"]["agentpile"]["empty_source"] is True
    assert ap.validate(run.record) == []


def test_negative_non_json_line_counted_unparseable(tmp_path):
    bad_file = tmp_path / "corrupt.jsonl"
    bad_file.write_text(
        '{"type": "user", "sessionId": "s-err", "uuid": "u-01", "version": "2.1.170", "message": {"role": "user", "content": "hi"}}\n'
        'THIS LINE IS NOT VALID JSON\n',
        encoding="utf-8"
    )
    run = next(ap.load(bad_file, reader="claude-code-jsonl"))
    acc = run.dataset["accounting"]
    assert acc["unparseable"] == 1
    assert acc["lines"] == 2


def test_negative_json_file_not_recognized(tmp_path):
    json_file = tmp_path / "session.json"
    json_file.write_text(
        '{"type": "user", "sessionId": "s-json", "uuid": "u-json", "version": "2.1.170", "message": {"role": "user", "content": "hi"}}\n',
        encoding="utf-8"
    )
    assert not claude_code.sniff(json_file)
    with pytest.raises(ap.UnrecognizedSource):
        next(ap.load(json_file))
