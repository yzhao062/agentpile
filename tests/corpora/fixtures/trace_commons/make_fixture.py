"""Write the synthetic trace_commons test fixtures (no real session content).

Writes at least two synthetic sessions to sessions/claude_code/*.jsonl and one
file to sessions/cursor/ that must be ignored by the trace_commons loader.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).parent
VERSION = "2.1.170"
MODEL = "claude-fx-1"
CWD = "/repo"
BRANCH = "main"


def _envelope(rec_type: str, session_id: str, uuid: str, parent_uuid: Any = None,
              timestamp: str = "2026-10-01T12:00:00.000Z", **extra: Any) -> Dict[str, Any]:
    out = {
        "type": rec_type,
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "isSidechain": False,
        "userType": "external",
        "cwd": CWD,
        "sessionId": session_id,
        "version": VERSION,
        "gitBranch": BRANCH,
        "timestamp": timestamp,
        "entrypoint": "cli",
    }
    out.update(extra)
    return out


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def make_fixtures() -> None:
    # Session 1: simple conversation
    sid1 = "session-tc-01"
    s1_records = [
        {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-10-01T11:59:59.000Z", "sessionId": sid1},
        _envelope("user", sid1, "00000000-0000-4000-8000-000000001001", None,
                  message={"role": "user", "content": "List the repository files."},
                  timestamp="2026-10-01T12:00:01.000Z"),
        _envelope("assistant", sid1, "00000000-0000-4000-8000-000000001002", "00000000-0000-4000-8000-000000001001",
                  requestId="req_tc_01",
                  message={
                      "id": "msg_tc_01",
                      "type": "message",
                      "role": "assistant",
                      "model": MODEL,
                      "content": [{"type": "text", "text": "The repository contains app.py and tests."}],
                      "stop_reason": "end_turn",
                      "usage": {"input_tokens": 80, "output_tokens": 15, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                  },
                  timestamp="2026-10-01T12:00:02.000Z"),
    ]
    _write_jsonl(HERE / "sessions" / "claude_code" / f"{sid1}.jsonl", s1_records)

    # Session 2: conversation with tool use
    sid2 = "session-tc-02"
    s2_records = [
        {"type": "queue-operation", "operation": "enqueue", "timestamp": "2026-10-01T11:59:59.000Z", "sessionId": sid2},
        _envelope("user", sid2, "00000000-0000-4000-8000-000000002001", None,
                  message={"role": "user", "content": "Run tests on the codebase."},
                  timestamp="2026-10-01T12:00:01.000Z"),
        _envelope("assistant", sid2, "00000000-0000-4000-8000-000000002002", "00000000-0000-4000-8000-000000002001",
                  requestId="req_tc_02",
                  message={
                      "id": "msg_tc_02",
                      "type": "message",
                      "role": "assistant",
                      "model": MODEL,
                      "content": [{"type": "tool_use", "id": "toolu_tc_01", "name": "Bash", "input": {"command": "pytest"}}],
                      "stop_reason": "tool_use",
                      "usage": {"input_tokens": 90, "output_tokens": 20, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                  },
                  timestamp="2026-10-01T12:00:02.000Z"),
        _envelope("user", sid2, "00000000-0000-4000-8000-000000002003", "00000000-0000-4000-8000-000000002002",
                  message={"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_tc_01", "content": "1 passed"}]},
                  toolUseResult={"stdout": "1 passed"},
                  timestamp="2026-10-01T12:00:03.000Z"),
        _envelope("assistant", sid2, "00000000-0000-4000-8000-000000002004", "00000000-0000-4000-8000-000000002003",
                  requestId="req_tc_02b",
                  message={
                      "id": "msg_tc_02b",
                      "type": "message",
                      "role": "assistant",
                      "model": MODEL,
                      "content": [{"type": "text", "text": "All tests passed."}],
                      "stop_reason": "end_turn",
                      "usage": {"input_tokens": 120, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                  },
                  timestamp="2026-10-01T12:00:04.000Z"),
    ]
    _write_jsonl(HERE / "sessions" / "claude_code" / f"{sid2}.jsonl", s2_records)

    # File under sessions/cursor/ that must be ignored
    cursor_dir = HERE / "sessions" / "cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    cursor_file = cursor_dir / "cursor-ignored.jsonl"
    with cursor_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"cursor_session": "cur_01", "content": "This cursor file must be ignored."}) + "\n")

    print(f"Trace Commons fixtures written to {HERE}")


if __name__ == "__main__":
    make_fixtures()
