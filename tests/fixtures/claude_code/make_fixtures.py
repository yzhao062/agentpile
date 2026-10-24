"""Write the synthetic Claude Code test fixtures (no real session content).

Generates 11 synthetic session files and their subagent/workflow sidecars into
tests/fixtures/claude_code/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).parent
VERSION = "2.1.170"
MODEL = "claude-fx-1"
CWD = "/repo"
BRANCH = "main"


def _envelope(rec_type: str, session_id: str, uuid: str, parent_uuid: Optional[str] = None,
              timestamp: str = "2026-10-01T12:00:00.000Z", is_sidechain: bool = False,
              agent_id: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": rec_type,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": is_sidechain,
        "userType": "external",
        "cwd": CWD,
        "sessionId": session_id,
        "version": VERSION,
        "gitBranch": BRANCH,
        "timestamp": timestamp,
        "entrypoint": "cli",
    }
    if agent_id is not None:
        out["agentId"] = agent_id
    out.update(extra)
    return out


def _queue_op(session_id: str, op: str = "enqueue", ts: str = "2026-10-01T11:59:59.000Z") -> Dict[str, Any]:
    return {"type": "queue-operation", "operation": op, "timestamp": ts, "sessionId": session_id}


def _user_text(session_id: str, uuid: str, parent_uuid: Optional[str], text: str,
               ts: str = "2026-10-01T12:00:01.000Z", **kw: Any) -> Dict[str, Any]:
    msg = {"role": "user", "content": text}
    return _envelope("user", session_id, uuid, parent_uuid, timestamp=ts, message=msg, **kw)


def _user_result(session_id: str, uuid: str, parent_uuid: Optional[str], tool_use_id: str,
                 content: Any, ts: str = "2026-10-01T12:00:03.000Z", tool_use_result: Any = None,
                 is_error: bool = False, **kw: Any) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    msg = {"role": "user", "content": [block]}
    extra = {}
    if tool_use_result is not None:
        extra["toolUseResult"] = tool_use_result
    return _envelope("user", session_id, uuid, parent_uuid, timestamp=ts, message=msg, **extra, **kw)


def _assistant_rec(session_id: str, uuid: str, parent_uuid: Optional[str], msg_id: str,
                   block: Dict[str, Any], ts: str = "2026-10-01T12:00:02.000Z",
                   req_id: str = "req_fx_01", usage: Optional[Dict[str, Any]] = None,
                   stop_reason: Optional[str] = None, **kw: Any) -> Dict[str, Any]:
    if usage is None:
        usage = {"input_tokens": 100, "output_tokens": 20, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    msg: Dict[str, Any] = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [block],
        "stop_reason": stop_reason,
        "usage": usage,
    }
    return _envelope("assistant", session_id, uuid, parent_uuid, timestamp=ts, requestId=req_id, message=msg, **kw)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# -----------------------------------------------------------------------------
# Fixtures a - k
# -----------------------------------------------------------------------------

def make_split_interleaved() -> None:
    sid = "00000000-0000-4000-8000-000000000001"
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000011", None, "Run task A and task B",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000012", "00000000-0000-4000-8000-000000000011",
                       "msg_fx_a", {"type": "text", "text": "I will run task A and task B."},
                       ts="2026-10-01T12:00:02.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000013", "00000000-0000-4000-8000-000000000012",
                       "msg_fx_a", {"type": "tool_use", "id": "toolu_fx_a1", "name": "Bash", "input": {"command": "taskA"}},
                       ts="2026-10-01T12:00:03.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000014", "00000000-0000-4000-8000-000000000013",
                     "toolu_fx_a1", "result of task A", ts="2026-10-01T12:00:04.000Z",
                     tool_use_result={"stdout": "result of task A"}),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000015", "00000000-0000-4000-8000-000000000014",
                       "msg_fx_a", {"type": "tool_use", "id": "toolu_fx_a2", "name": "Bash", "input": {"command": "taskB"}},
                       ts="2026-10-01T12:00:05.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000016", "00000000-0000-4000-8000-000000000015",
                     "toolu_fx_a2", "result of task B", ts="2026-10-01T12:00:06.000Z",
                     tool_use_result={"stdout": "result of task B"}),
    ]
    _write_jsonl(HERE / "split_interleaved.jsonl", records)


def make_early_result() -> None:
    sid = "00000000-0000-4000-8000-000000000002"
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000021", None, "Check disk space",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000022", "00000000-0000-4000-8000-000000000021",
                       "msg_fx_b", {"type": "text", "text": "Checking disk space now."},
                       ts="2026-10-01T12:00:02.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000023", "00000000-0000-4000-8000-000000000022",
                     "toolu_fx_b1", "Disk free: 50GB", ts="2026-10-01T12:00:03.000Z",
                     tool_use_result={"stdout": "Disk free: 50GB"}),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000024", "00000000-0000-4000-8000-000000000023",
                       "msg_fx_b", {"type": "tool_use", "id": "toolu_fx_b1", "name": "Bash", "input": {"command": "df -h"}},
                       ts="2026-10-01T12:00:04.000Z"),
    ]
    _write_jsonl(HERE / "early_result.jsonl", records)


def make_edit_branch() -> None:
    sid = "00000000-0000-4000-8000-000000000003"
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000031", None, "Initial user prompt",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000032", "00000000-0000-4000-8000-000000000031",
                       "msg_fx_c", {"type": "text", "text": "Initial response."},
                       ts="2026-10-01T12:00:02.000Z"),
        _envelope("system", sid, "00000000-0000-4000-8000-000000000033", "00000000-0000-4000-8000-000000000032",
                  subtype="turn_duration", durationMs=1200, messageCount=2, isMeta=False,
                  timestamp="2026-10-01T12:00:03.000Z"),
        _user_text(sid, "00000000-0000-4000-8000-000000000034", "00000000-0000-4000-8000-000000000033",
                   "Original follow-up", ts="2026-10-01T12:00:04.000Z"),
        _user_text(sid, "00000000-0000-4000-8000-000000000035", "00000000-0000-4000-8000-000000000033",
                   "Edited follow-up", ts="2026-10-01T12:00:05.000Z"),
    ]
    _write_jsonl(HERE / "edit_branch.jsonl", records)


def make_compaction() -> None:
    sid = "00000000-0000-4000-8000-000000000004"
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000041", None, "Start conversation",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000042", "00000000-0000-4000-8000-000000000041",
                       "msg_fx_d1", {"type": "text", "text": "Conversation before compaction."},
                       ts="2026-10-01T12:00:02.000Z"),
        _envelope("system", sid, "00000000-0000-4000-8000-000000000043", None,
                  logicalParentUuid="00000000-0000-4000-8000-000000000042",
                  subtype="compact_boundary", content="Conversation compacted", level="info",
                  compactMetadata={"trigger": "auto", "preTokens": 2000, "postTokens": 500, "durationMs": 800},
                  isMeta=False, timestamp="2026-10-01T12:00:03.000Z"),
        _user_text(sid, "00000000-0000-4000-8000-000000000044", "00000000-0000-4000-8000-000000000043",
                   "This is the compaction summary.", ts="2026-10-01T12:00:04.000Z",
                   isCompactSummary=True, isVisibleInTranscriptOnly=True),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000045", "00000000-0000-4000-8000-000000000044",
                       "msg_fx_d2", {"type": "text", "text": "Continuing after compaction."},
                       ts="2026-10-01T12:00:05.000Z"),
    ]
    _write_jsonl(HERE / "compaction.jsonl", records)


def make_api_error() -> None:
    sid = "00000000-0000-4000-8000-000000000005"
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000051", None, "Call API",
                   ts="2026-10-01T12:00:01.000Z"),
        _envelope("system", sid, "00000000-0000-4000-8000-000000000052", "00000000-0000-4000-8000-000000000051",
                  subtype="api_error", level="error",
                  error={"type": "overloaded_error", "message": "Server overloaded"},
                  retryInMs=1000, retryAttempt=1, maxRetries=10, timestamp="2026-10-01T12:00:02.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000053", "00000000-0000-4000-8000-000000000051",
                       "msg_fx_e", {"type": "text", "text": "API call succeeded on retry."},
                       ts="2026-10-01T12:00:03.000Z"),
    ]
    _write_jsonl(HERE / "api_error.jsonl", records)


def make_images() -> None:
    sid = "00000000-0000-4000-8000-000000000006"
    img_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    user_content = [{"type": "text", "text": "Analyze this image:"}, img_block]
    res_content = [{"type": "text", "text": "Inspection result:"}, img_block]
    records = [
        _queue_op(sid),
        _envelope("user", sid, "00000000-0000-4000-8000-000000000061", None,
                  message={"role": "user", "content": user_content}, timestamp="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000062", "00000000-0000-4000-8000-000000000061",
                       "msg_fx_f", {"type": "tool_use", "id": "toolu_fx_f1", "name": "Inspect", "input": {"target": "image"}},
                       ts="2026-10-01T12:00:02.000Z"),
        _envelope("user", sid, "00000000-0000-4000-8000-000000000063", "00000000-0000-4000-8000-000000000062",
                  message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_fx_f1",
                                                        "content": res_content, "is_error": False}]},
                  timestamp="2026-10-01T12:00:03.000Z"),
    ]
    _write_jsonl(HERE / "images.jsonl", records)


def make_unknown() -> None:
    sid = "00000000-0000-4000-8000-000000000007"
    records = [
        _queue_op(sid),
        {"type": "future-widget-event", "widgetId": "w_fx_1", "sessionId": sid, "version": VERSION},
        _user_text(sid, "00000000-0000-4000-8000-000000000071", None, "Handle unknown items",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000072", "00000000-0000-4000-8000-000000000071",
                       "msg_fx_g", {"type": "future_gadget", "widget": "data"},
                       ts="2026-10-01T12:00:02.000Z"),
    ]
    _write_jsonl(HERE / "unknown.jsonl", records)


def make_duplicates() -> None:
    sid = "00000000-0000-4000-8000-000000000008"
    orig_rec = _user_text(sid, "00000000-0000-4000-8000-000000000081", None, "Test duplicate handling",
                          ts="2026-10-01T12:00:01.000Z")
    exact_dup = dict(orig_rec)
    conflict_dup = _user_text(sid, "00000000-0000-4000-8000-000000000081", None, "Conflicting duplicate content",
                              ts="2026-10-01T12:00:01.000Z")
    records = [
        _queue_op(sid),
        orig_rec,
        exact_dup,
        conflict_dup,
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000082", "00000000-0000-4000-8000-000000000081",
                       "msg_fx_h1", {"type": "tool_use", "id": "toolu_fx_reuse", "name": "Bash", "input": {"cmd": "ls"}},
                       ts="2026-10-01T12:00:02.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000083", "00000000-0000-4000-8000-000000000082",
                     "toolu_fx_reuse", "file1.txt", ts="2026-10-01T12:00:03.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000084", "00000000-0000-4000-8000-000000000083",
                       "msg_fx_h2", {"type": "tool_use", "id": "toolu_fx_reuse", "name": "Bash", "input": {"cmd": "pwd"}},
                       ts="2026-10-01T12:00:04.000Z"),
    ]
    _write_jsonl(HERE / "duplicates.jsonl", records)


def make_unpaired() -> None:
    sid = "00000000-0000-4000-8000-000000000009"
    records = [
        _queue_op(sid),
        _user_result(sid, "00000000-0000-4000-8000-000000000091", None, "toolu_fx_early_orphan",
                     "result before agent", ts="2026-10-01T12:00:01.000Z"),
        _user_text(sid, "00000000-0000-4000-8000-000000000092", "00000000-0000-4000-8000-000000000091",
                   "Now start normal message", ts="2026-10-01T12:00:02.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000093", "00000000-0000-4000-8000-000000000092",
                       "msg_fx_i", {"type": "tool_use", "id": "toolu_fx_unanswered", "name": "Bash", "input": {"cmd": "sleep 10"}},
                       ts="2026-10-01T12:00:03.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000094", "00000000-0000-4000-8000-000000000093",
                     "toolu_fx_ghost", "result for non-existent call", ts="2026-10-01T12:00:04.000Z"),
    ]
    _write_jsonl(HERE / "unpaired.jsonl", records)


def make_attachment_beside_result() -> None:
    sid = "00000000-0000-4000-8000-000000000010"
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000101", None, "Run tool with hook",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000102", "00000000-0000-4000-8000-000000000101",
                       "msg_fx_j", {"type": "tool_use", "id": "toolu_fx_j1", "name": "Bash", "input": {"cmd": "echo test"}},
                       ts="2026-10-01T12:00:02.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000103", "00000000-0000-4000-8000-000000000102",
                     "toolu_fx_j1", "test", ts="2026-10-01T12:00:03.000Z"),
        _envelope("attachment", sid, "00000000-0000-4000-8000-000000000104", "00000000-0000-4000-8000-000000000102",
                  attachment={"type": "hook_success", "content": "post-tool hook passed"},
                  timestamp="2026-10-01T12:00:04.000Z"),
    ]
    _write_jsonl(HERE / "attachment_beside_result.jsonl", records)


def make_subagents() -> None:
    sid = "00000000-0000-4000-8000-000000000099"
    sub_dir = HERE / "subagents" / "subagents"

    # Main session file
    records = [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000901", None, "Run subagents and workflows",
                   ts="2026-10-01T12:00:01.000Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000902", "00000000-0000-4000-8000-000000000901",
                       "msg_fx_k", {"type": "tool_use", "id": "toolu_fx_k1", "name": "Agent", "input": {"prompt": "Run agent X"}},
                       ts="2026-10-01T12:00:02.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000903", "00000000-0000-4000-8000-000000000902",
                     "toolu_fx_k1", "Agent X finished", ts="2026-10-01T12:00:03.000Z",
                     tool_use_result={"agentId": "afx0000000000001"}),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000904", "00000000-0000-4000-8000-000000000903",
                       "msg_fx_k", {"type": "tool_use", "id": "toolu_fx_k2", "name": "Agent", "input": {"prompt": "Run missing agent"}},
                       ts="2026-10-01T12:00:04.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000905", "00000000-0000-4000-8000-000000000904",
                     "toolu_fx_k2", "Missing agent error", ts="2026-10-01T12:00:05.000Z",
                     tool_use_result={"agentId": "afx_missing_id"}),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000906", "00000000-0000-4000-8000-000000000905",
                       "msg_fx_k", {"type": "tool_use", "id": "toolu_fx_k3", "name": "Workflow", "input": {"flow": "process"}},
                       ts="2026-10-01T12:00:06.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000907", "00000000-0000-4000-8000-000000000906",
                     "toolu_fx_k3", "Workflow W finished", ts="2026-10-01T12:00:07.000Z",
                     tool_use_result={"runId": "wf_fx_w"}),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000908", "00000000-0000-4000-8000-000000000907",
                       "msg_fx_k", {"type": "tool_use", "id": "toolu_fx_k4", "name": "Agent", "input": {"prompt": "Run parent subagent"}},
                       ts="2026-10-01T12:00:08.000Z"),
        _user_result(sid, "00000000-0000-4000-8000-000000000909", "00000000-0000-4000-8000-000000000908",
                     "toolu_fx_k4", "Parent subagent finished", ts="2026-10-01T12:00:09.000Z",
                     tool_use_result={"agentId": "afx0000000000003"}),
    ]
    _write_jsonl(HERE / "subagents.jsonl", records)

    # Subagent 1: afx0000000000001 (referenced by agentId)
    _write_jsonl(sub_dir / "agent-afx0000000000001.jsonl", [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000111", None, "Subagent X prompt",
                   is_sidechain=True, agent_id="afx0000000000001", ts="2026-10-01T12:00:02.100Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000112", "00000000-0000-4000-8000-000000000111",
                       "msg_fx_sub1", {"type": "text", "text": "Subagent X output."},
                       is_sidechain=True, agent_id="afx0000000000001", ts="2026-10-01T12:00:02.200Z"),
    ])

    # Subagent 2: afx0000000000002 (unreferenced agent file)
    _write_jsonl(sub_dir / "agent-afx0000000000002.jsonl", [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000221", None, "Unreferenced subagent prompt",
                   is_sidechain=True, agent_id="afx0000000000002", ts="2026-10-01T12:00:02.300Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000222", "00000000-0000-4000-8000-000000000221",
                       "msg_fx_sub2", {"type": "text", "text": "Unreferenced subagent output."},
                       is_sidechain=True, agent_id="afx0000000000002", ts="2026-10-01T12:00:02.400Z"),
    ])

    # Subagent 3: afx0000000000003 (calls Agent child afx0000000000004, has two usage snapshots)
    usage_snap1 = {"input_tokens": 50, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    usage_snap2 = {"input_tokens": 50, "output_tokens": 35, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    _write_jsonl(sub_dir / "agent-afx0000000000003.jsonl", [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000331", None, "Parent subagent prompt",
                   is_sidechain=True, agent_id="afx0000000000003", ts="2026-10-01T12:00:08.100Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000332", "00000000-0000-4000-8000-000000000331",
                       "msg_fx_sub3", {"type": "thinking", "thinking": "Planning delegation", "signature": "sig_p"},
                       is_sidechain=True, agent_id="afx0000000000003", ts="2026-10-01T12:00:08.200Z",
                       usage=usage_snap1),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000333", "00000000-0000-4000-8000-000000000332",
                       "msg_fx_sub3", {"type": "tool_use", "id": "toolu_fx_sub_child", "name": "Agent", "input": {"prompt": "Run child"}},
                       is_sidechain=True, agent_id="afx0000000000003", ts="2026-10-01T12:00:08.300Z",
                       usage=usage_snap2, stop_reason="tool_use"),
        _user_result(sid, "00000000-0000-4000-8000-000000000334", "00000000-0000-4000-8000-000000000333",
                     "toolu_fx_sub_child", "Child finished", ts="2026-10-01T12:00:08.400Z",
                     is_sidechain=True, agent_id="afx0000000000003",
                     tool_use_result={"agentId": "afx0000000000004"}),
    ])

    # Subagent 4: afx0000000000004 (child of afx0000000000003)
    _write_jsonl(sub_dir / "agent-afx0000000000004.jsonl", [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000441", None, "Child subagent prompt",
                   is_sidechain=True, agent_id="afx0000000000004", ts="2026-10-01T12:00:08.310Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000442", "00000000-0000-4000-8000-000000000441",
                       "msg_fx_sub4", {"type": "text", "text": "Child subagent output."},
                       is_sidechain=True, agent_id="afx0000000000004", ts="2026-10-01T12:00:08.320Z"),
    ])

    # Claude Code also writes agent-<id>.meta.json beside each subagent file and journal.jsonl in a workflow
    # folder. They are no traces, so they stay out of tests/fixtures, where every file must load as one;
    # tests/test_claude_code.py adds them to a copy (SIDECARS below).

    # Workflow folder: subagents/workflows/wf_fx_w/
    wf_dir = sub_dir / "workflows" / "wf_fx_w"

    _write_jsonl(wf_dir / "agent-afx0000000000005.jsonl", [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000551", None, "Workflow worker 1 prompt",
                   is_sidechain=True, agent_id="afx0000000000005", ts="2026-10-01T12:00:06.100Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000552", "00000000-0000-4000-8000-000000000551",
                       "msg_fx_w1", {"type": "text", "text": "Workflow worker 1 output."},
                       is_sidechain=True, agent_id="afx0000000000005", ts="2026-10-01T12:00:06.200Z"),
    ])

    _write_jsonl(wf_dir / "agent-afx0000000000006.jsonl", [
        _queue_op(sid),
        _user_text(sid, "00000000-0000-4000-8000-000000000661", None, "Workflow worker 2 prompt",
                   is_sidechain=True, agent_id="afx0000000000006", ts="2026-10-01T12:00:06.300Z"),
        _assistant_rec(sid, "00000000-0000-4000-8000-000000000662", "00000000-0000-4000-8000-000000000661",
                       "msg_fx_w2", {"type": "text", "text": "Workflow worker 2 output."},
                       is_sidechain=True, agent_id="afx0000000000006", ts="2026-10-01T12:00:06.400Z"),
    ])



SIDECARS = {
    "subagents/agent-afx0000000000001.meta.json": [
        {"agentType": "general-purpose", "description": "Synthetic subagent", "requestNonInteractive": True,
         "spawnDepth": 1, "toolUseId": "toolu_fx_agent1"}],
    "subagents/workflows/wf_fx_w/journal.jsonl": [
        {"agentId": "afx0000000000005", "key": "k1", "label": "worker-1", "phase": "Run"},
        {"agentId": "afx0000000000006", "key": "k2", "label": "worker-2", "phase": "Run"},
        {"agentId": "afx0000000000005", "key": "k1", "result": "done"},
        {"agentId": "afx0000000000006", "key": "k2", "result": "done"}],
}


def main() -> None:
    make_split_interleaved()
    make_early_result()
    make_edit_branch()
    make_compaction()
    make_api_error()
    make_images()
    make_unknown()
    make_duplicates()
    make_unpaired()
    make_attachment_beside_result()
    make_subagents()
    print("All claude_code fixtures written successfully.")


if __name__ == "__main__":
    main()
