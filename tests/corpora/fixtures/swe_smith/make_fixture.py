"""Write the synthetic swe_smith fixture: two small Parquet shards in the layout of the pinned files.

Every string is invented here; no dataset content is copied. Run it from anywhere:
``python make_fixture.py`` rewrites ``data/tool-0000{0,1}-of-00008.parquet`` next to this file.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
ONE_LINER = "You are a synthetic assistant."


def part(text):
    return [{"type": "text", "text": text}]


def sysmsg(content=ONE_LINER, agent=True):
    msg = {"role": "system", "content": content}
    return dict(msg, agent="main", message_type="system_prompt") if agent else msg


def call(cid, name, args, index=None):
    tc = {"function": {"arguments": json.dumps(args), "name": name}, "id": cid, "type": "function"}
    return dict(tc, index=index) if index is not None else tc


def action(content, cid, name, args, index=None, final=False):
    tcs = None if final else [call(cid, name, args, index)]
    return {"role": "assistant", "content": content, "thought": content, "action": "" if final else name,
            "agent": "main", "tool_calls": tcs, "message_type": "action"}


def observation(content, ids, cache=False):
    msg = {"role": "tool", "content": content, "agent": "main", "message_type": "observation", "tool_call_ids": ids}
    return dict(msg, cache_control={"type": "ephemeral"}) if cache else msg


PROMPT = ("---- BEGIN FUNCTION #1: {a} ----\nDescription: first synthetic function.\n---- END FUNCTION #1 ----\n\n"
          "---- BEGIN FUNCTION #2: {b} ----\nDescription: second synthetic function.\n---- END FUNCTION #2 ----\n")

native_a = [
    sysmsg(),
    {"role": "user", "content": part("Fix the synthetic bug."), "agent": "main", "message_type": "observation"},
    action("Look at the file.", "toolu_synth_1", "str_replace_editor", {"command": "view", "path": "/testbed/a.py"}, 1),
    observation(part("OBSERVATION:\nfile a"), ["toolu_synth_1"], cache=True),
    action("", "toolu_synth_2", "bash", {"command": "ls"}, 0),
    observation("OBSERVATION:\nlisting", ["toolu_synth_2"]),
    action("Done.", "toolu_synth_3", "submit", {}, 1),
]
native_b = [
    sysmsg(),
    {"role": "user", "content": "Plain string task.", "agent": "main", "message_type": "observation"},
    action("Run it.", "call_synthB1", "bash", {"command": "ls"}),
    observation(part("OBSERVATION:\nnothing"), ["call_synthB1"]),
    action("I could not finish.", None, None, None, final=True),
]
text_c = [
    sysmsg(PROMPT.format(a="bash", b="submit") + "\nCall a function in the <function=NAME> format.", agent=False),
    {"role": "user", "content": "Synthetic task for a text row.", },
    {"role": "assistant", "content": "Listing.\n<function=bash>\n<parameter=command>ls</parameter>\n</function>"},
    {"role": "user", "content": "OBSERVATION:\nfile.py"},
    {"role": "assistant", "content": "<function=submit>\n</function>"},
]
text_d = [
    sysmsg(PROMPT.format(a="bash", b="bash"), agent=False),
    {"role": "user", "content": "Another synthetic task."},
    {"role": "assistant", "content": "I give up."},
]
native_g = [
    sysmsg(),
    {"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}],
     "agent": "main", "message_type": "observation"},
    action("First.", "toolu_synth_g1", "bash", {"command": "pwd"}, 1),
    observation(part("OBSERVATION:\ntwo ids"), ["toolu_synth_g1", "toolu_synth_g9"]),
    action("Second.", "toolu_synth_g2", "bash", {"command": "id"}, 1),
    observation(part("OBSERVATION:\nunknown id"), ["toolu_nomatch"]),
]

SCHEMA = pa.schema([("messages", pa.string()), ("instance_id", pa.string()), ("resolved", pa.bool_()),
                    ("model", pa.string()), ("traj_id", pa.string()), ("patch", pa.string())])


def row(messages, instance, traj, resolved, model, patch):
    text = messages if isinstance(messages, str) else json.dumps(messages)
    return {"messages": text, "instance_id": instance, "resolved": resolved, "model": model, "traj_id": traj,
            "patch": patch}


SHARDS = {
    "tool-00000-of-00008.parquet": [
        row(native_a, "synth__repo-a.0000000.pr_1", "synth__repo-a.0000000.pr_1.aaaa1111", True,
            "claude-3-7-sonnet-20250219", "diff --git a/x b/x\n"),
        row(native_b, "synth__repo-b.0000000.pr_2", "synth__repo-b.0000000.pr_2.bbbb2222", False,
            "gpt-4o-2024-08-06", ""),
        row(text_c, "synth__repo-c.0000000.pr_3", "synth__repo-c.0000000.pr_3.cccc3333", True,
            "claude-3-5-sonnet-20241022", ""),
    ],
    "tool-00001-of-00008.parquet": [
        row(text_d, "synth__repo-d.0000000.pr_4", "synth__repo-d.0000000.pr_4.dddd4444", False,
            "claude-3-7-sonnet-20250219", "diff --git a/y b/y\n"),
        row([], "synth__repo-e.0000000.pr_5", "synth__repo-e.0000000.pr_5.eeee5555", False, None, ""),
        row("this is not json", "synth__repo-f.0000000.pr_6", "synth__repo-f.0000000.pr_6.ffff6666", False,
            "claude-3-7-sonnet-20250219", ""),
        row(native_g, "synth__repo-g.0000000.pr_7", None, True, "claude-3-7-sonnet-20250219", ""),
    ],
}

if __name__ == "__main__":
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    for name, rows in SHARDS.items():
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out / name)
        print(out / name, len(rows))
