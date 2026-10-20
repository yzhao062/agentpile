"""Write the synthetic copy of the SWE-rebench OpenHands file layout (hand-written rows, no dataset content).

Run: python make_fixture.py   (needs pyarrow). It writes trajectories.parquet and tools.json next to itself.
The Parquet schema mirrors the real file: every message struct has all five members (null where the role has
none), and the file has several row groups.
"""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).parent

CALL = pa.struct([("function", pa.struct([("arguments", pa.string()), ("name", pa.string())])),
                  ("id", pa.string()), ("type", pa.string())])
MESSAGE = pa.struct([("content", pa.string()), ("name", pa.string()), ("role", pa.string()),
                     ("tool_call_id", pa.string()), ("tool_calls", pa.list_(CALL))])
PROP = pa.struct([("description", pa.string()), ("type", pa.string())])
TOOL = pa.struct([("function", pa.struct([
    ("description", pa.string()), ("name", pa.string()),
    ("parameters", pa.struct([("properties", pa.struct([("command", PROP), ("thought", PROP)])),
                              ("required", pa.list_(pa.string())), ("type", pa.string())]))])),
    ("type", pa.string())])
SCHEMA = pa.schema([("trajectory_id", pa.string()), ("instance_id", pa.string()), ("repo", pa.string()),
                    ("trajectory", pa.list_(MESSAGE)), ("tools", pa.list_(TOOL)), ("model_patch", pa.string()),
                    ("exit_status", pa.string()), ("resolved", pa.int64()),
                    ("gen_tests_correct", pa.float64()), ("pred_passes_gen_tests", pa.float64())])


def definition(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {p: {"type": "string", "description": f"The {p}."} for p in props},
        "required": required}}}


# tools.json: five synthetic function schemas with the real tool names.
TOOLS = [
    definition("execute_bash", "Run one synthetic shell command.", ["command"], ["command"]),
    definition("think", "Log a synthetic thought.", ["thought"], ["thought"]),
    definition("finish", "End the synthetic task.", ["message"], ["message"]),
    definition("task_tracker", "Track synthetic tasks.", ["command"], ["command"]),
    definition("str_replace_editor", "Edit a synthetic file.", ["command", "path"], ["command", "path"]),
]
# The row-level tools column keeps the Parquet union padding: members a definition lacks are null.
ROW_TOOLS = [
    {"type": "function", "function": {"name": "execute_bash", "description": "Run one synthetic shell command.",
                                       "parameters": {"type": "object", "required": ["command"],
                                                      "properties": {"command": {"type": "string", "description": "The command."},
                                                                     "thought": None}}}},
    {"type": "function", "function": {"name": "think", "description": "Log a synthetic thought.",
                                       "parameters": {"type": "object", "required": ["thought"],
                                                      "properties": {"command": None,
                                                                     "thought": {"type": "string", "description": "The thought."}}}}},
]


def msg(role, content, name=None, call_id=None, calls=None):
    return {"role": role, "content": content, "name": name, "tool_call_id": call_id, "tool_calls": calls}


def call(cid, name, arguments):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": arguments}}


def row(tid, instance, trajectory, resolved, exit_status, gen, pred, patch):
    return {"trajectory_id": tid, "instance_id": instance, "repo": instance.split("__")[0] + "/" + instance.split("__")[1].rsplit("-", 1)[0],
            "trajectory": trajectory, "tools": ROW_TOOLS, "model_patch": patch, "exit_status": exit_status,
            "resolved": resolved, "gen_tests_correct": gen, "pred_passes_gen_tests": pred}


ROWS = [
    # One call per turn, results on the same step, a final finish call with no result.
    row("syn-traj-1", "synorg__synrepo-1", [
        msg("system", "SYNTHETIC SYSTEM PROMPT"),
        msg("user", "SYNTHETIC TASK: make the widget total correct."),
        msg("assistant", "", calls=[call("syn-call-1a", "think", '{"thought": "plan the fix"}')]),
        msg("tool", "Your thought has been logged.", name="think", call_id="syn-call-1a"),
        msg("assistant", "Running the tests.", calls=[call("syn-call-1b", "execute_bash", '{"command": "pytest -q"}')]),
        msg("tool", "1 passed", name="execute_bash", call_id="syn-call-1b"),
        msg("assistant", "", calls=[call("syn-call-1c", "finish", '{"message": "done"}')]),
    ], 1, "submit", 1.0, 0.0, "diff --git a/w.py b/w.py\n+fixed\n"),
    # A two-call turn, an assistant turn without calls, a later user message, a function missing from
    # tools.json, an empty argument object, null test fields and an empty patch.
    row("syn-traj-2", "synorg__synrepo-2", [
        msg("system", "SYNTHETIC SYSTEM PROMPT"),
        msg("user", "SYNTHETIC TASK: rename the helper."),
        msg("assistant", "Two checks at once.", calls=[call("syn-call-2a", "execute_bash", '{"command": "ls"}'),
                                                       call("syn-call-2b", "str_replace_editor", '{"command": "view", "path": "/w/a.py"}')]),
        msg("tool", "a.py", name="execute_bash", call_id="syn-call-2a"),
        msg("tool", "file body", name="str_replace_editor", call_id="syn-call-2b"),
        msg("assistant", "No tool this turn."),
        msg("user", "SYNTHETIC continue prompt."),
        msg("assistant", "", calls=[call("syn-call-2c", "browser_goto", "{}")]),
        msg("tool", "no such tool", name="browser_goto", call_id="syn-call-2c"),
        msg("assistant", "", calls=[call("syn-call-2d", "finish", '{"message": "stop"}')]),
    ], 0, "RuntimeError: synthetic iteration limit", None, None, ""),
    # An empty message list is still a run; resolved is null here.
    row("syn-traj-3", "synorg__synrepo-3", [], None, "unknown", 0.0, 1.0, ""),
    # Arguments that are not a JSON object keep their source string.
    row("syn-traj-4", "synorg__synrepo-4", [
        msg("system", "SYNTHETIC SYSTEM PROMPT"),
        msg("user", "SYNTHETIC TASK: add a test."),
        msg("assistant", "", calls=[call("syn-call-4a", "think", "not json")]),
        msg("tool", "ok", name="think", call_id="syn-call-4a"),
        msg("assistant", "", calls=[call("syn-call-4b", "finish", '{"message": "x"}')]),
    ], 1, "submit", None, None, "diff --git a/t.py b/t.py\n+test\n"),
]


if __name__ == "__main__":
    table = pa.Table.from_pylist(ROWS, schema=SCHEMA)
    pq.write_table(table, OUT / "trajectories.parquet", row_group_size=3)  # two row groups: 3 + 1 rows
    (OUT / "tools.json").write_text(json.dumps(TOOLS, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(table.num_rows, "rows,", pq.ParquetFile(OUT / "trajectories.parquet").num_row_groups, "row groups")
