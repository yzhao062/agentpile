"""Write the synthetic SWE-Gym OpenHands fixture: three tiny Parquet shards in the pinned layout.

Run once from anywhere: ``python make_fixture.py``. Every value is hand-written and
invented here; none of it comes from the dataset. The Arrow schema mirrors the
pinned files (struct members absent from a Python dict are stored as null, which
is how the real files pad messages, tool calls and tool schemas).
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent

STR = pa.string()
CALL = pa.struct([("function", pa.struct([("arguments", STR), ("name", STR)])), ("id", STR),
                  ("index", pa.int64()), ("type", STR)])
MESSAGE = pa.struct([("content", STR), ("function_call", pa.null()), ("name", STR), ("role", STR),
                     ("tool_call_id", STR), ("tool_calls", pa.list_(CALL))])


def _prop(*extra):
    return pa.struct([("description", STR)] + list(extra) + [("type", STR)])


PROPS = pa.struct([
    ("command", _prop(("enum", pa.list_(STR)))),
    ("file_text", _prop()), ("insert_line", _prop()), ("new_str", _prop()), ("old_str", _prop()), ("path", _prop()),
    ("view_range", pa.struct([("description", STR), ("items", pa.struct([("type", STR)])), ("type", STR)])),
])
TOOL = pa.struct([
    ("function", pa.struct([("description", STR), ("name", STR),
                            ("parameters", pa.struct([("properties", PROPS), ("required", pa.list_(STR)),
                                                      ("type", STR)]))])),
    ("type", STR),
])
REPORT = pa.struct([("empty_generation", pa.bool_()), ("error_eval", pa.bool_()), ("failed_apply_patch", pa.bool_()),
                    ("resolved", pa.bool_()), ("test_timeout", pa.bool_())])
TEST_RESULT = pa.struct([("apply_patch_output", STR), ("git_patch", STR), ("report", REPORT), ("test_output", STR)])
SCHEMA = pa.schema([("instance_id", STR), ("run_id", STR), ("resolved", pa.bool_()), ("messages", pa.list_(MESSAGE)),
                    ("tools", pa.list_(TOOL)), ("test_result", TEST_RESULT)])

TOOLS = [
    {"function": {"description": "Run a shell command (synthetic).", "name": "execute_bash",
                  "parameters": {"properties": {"command": {"description": "The command.", "type": "string"}},
                                 "required": ["command"], "type": "object"}},
     "type": "function"},
    {"function": {"description": "Finish the task (synthetic).", "name": "finish"}, "type": "function"},
    {"function": {"description": "View or edit files (synthetic).", "name": "str_replace_editor",
                  "parameters": {"properties": {
                      "command": {"description": "The edit command.", "enum": ["view", "create", "str_replace"],
                                  "type": "string"},
                      "path": {"description": "The path.", "type": "string"},
                      "view_range": {"description": "Lines to show.", "items": {"type": "integer"},
                                     "type": "array"}},
                      "required": ["command", "path"], "type": "object"}},
     "type": "function"},
]


def call(cid, name, arguments, index=None):
    return {"function": {"arguments": arguments, "name": name}, "id": cid, "index": index, "type": "function"}


def msg(role, content, calls=None, tool_call_id=None, name=None):
    out = {"content": content, "role": role}
    if calls is not None:
        out["tool_calls"] = calls
    if tool_call_id is not None:
        out["tool_call_id"] = tool_call_id
    if name is not None:
        out["name"] = name
    return out


def report(resolved, empty_generation=False):
    return {"empty_generation": empty_generation, "error_eval": False, "failed_apply_patch": False,
            "resolved": resolved, "test_timeout": False}


SYSTEM = msg("system", "You are a synthetic assistant.")
TASK = msg("user", "<uploaded_files>\n/workspace/synth__repo__1.0\n</uploaded_files>\nSynthetic task text.")

# Row 0: id reuse, a three-call turn, empty and null contents, a continue prompt, an unanswered final call.
ROW_PLAIN = {
    "instance_id": "synth__repo-1",
    "run_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t0-run_1",
    "resolved": False,
    "messages": [
        SYSTEM, TASK,
        msg("assistant", "", [call("call_A1", "str_replace_editor", '{"command":"view","path":"/workspace/x"}')]),
        msg("tool", "OBSERVATION:\nlisting", tool_call_id="call_A1", name="str_replace_editor"),
        msg("assistant", "Plan: three checks.", [
            call("call_B1", "execute_bash", '{"command":"ls"}'),
            call("call_B2", "execute_bash", '{"command": "pwd"}'),
            call("call_B3", "str_replace_editor", '{"command":"view","path":"/workspace/y"}')]),
        msg("tool", "OBSERVATION:\nfirst", tool_call_id="call_B1", name="execute_bash"),
        msg("tool", "OBSERVATION:\nsecond", tool_call_id="call_B2", name="execute_bash"),
        msg("tool", "OBSERVATION:\nthird", tool_call_id="call_B3", name="str_replace_editor"),
        msg("assistant", None, [call("call_A1", "execute_bash", '{"command":"echo reuse"}')]),
        msg("tool", "OBSERVATION:\nreused", tool_call_id="call_A1", name="execute_bash"),
        msg("assistant", "Thinking aloud without a tool call."),
        msg("user", "Please continue working on the task."),
        msg("assistant", None, [call("call_F1", "finish", "{}")]),
    ],
    "tools": TOOLS,
    "test_result": {"apply_patch_output": "synthetic apply log", "git_patch": "", "report": report(False, True),
                    "test_output": None},
}

# Row 1: tool-use ids with an index, arguments that are not JSON, and a call to an undefined function.
ROW_CLAUDE = {
    "instance_id": "synth__repo-2",
    "run_id": "claude-3-5-sonnet-20241022_maxiter_50_N_v2.1-no-hint-run_1",
    "resolved": True,
    "messages": [
        SYSTEM, TASK,
        msg("assistant", "Looking around.", [
            call("toolu_01A", "str_replace_editor", '{"command":"view","path":"/w"}', index=0),
            call("toolu_01B", "execute_bash", '{"command":"ls"}', index=1)]),
        msg("tool", "OBSERVATION:\none", tool_call_id="toolu_01A", name="str_replace_editor"),
        msg("tool", "OBSERVATION:\ntwo", tool_call_id="toolu_01B", name="execute_bash"),
        msg("assistant", "", [call("toolu_01C", "str_replace_editor", '{"command":"create","path":"/w/f.py","file', 0)]),
        msg("tool", "OBSERVATION:\nbad arguments", tool_call_id="toolu_01C", name="str_replace_editor"),
        msg("assistant", None, [call("toolu_01D", "view", '{"path":"/w"}', 1)]),
    ],
    "tools": TOOLS,
    "test_result": {"apply_patch_output": "ok", "git_patch": "diff --git a/f b/f", "report": report(True),
                    "test_output": "synthetic test log"},
}

# Row 2: no iteration tag in the run id, no resolved flag, no tools and no evaluation result.
ROW_BARE = {
    "instance_id": "synth__repo-3",
    "run_id": "custom-run",
    "resolved": None,
    "messages": [SYSTEM, TASK, msg("assistant", "Done without any tool.")],
    "tools": None,
    "test_result": None,
}

# Row 3: an empty message list.
ROW_EMPTY = {
    "instance_id": "synth__repo-4",
    "run_id": "gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t02-run_1",
    "resolved": False,
    "messages": [],
    "tools": TOOLS,
    "test_result": {"apply_patch_output": None, "git_patch": None, "report": report(False), "test_output": None},
}

SHARDS = [[ROW_PLAIN, ROW_CLAUDE], [ROW_BARE], [ROW_EMPTY]]

if __name__ == "__main__":
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for i, rows in enumerate(SHARDS):
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out / f"train.raw-{i:05d}-of-00003.parquet")
    print("wrote", sorted(p.name for p in out.glob("*.parquet")))
