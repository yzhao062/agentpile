"""Write the synthetic swe_agent_nebius fixture: two small Parquet shards in the pinned layout.

Run once with pyarrow installed: ``python make_fixture.py``. Every string is invented here; no
dataset content is copied. The Arrow types match the pinned shards (a list of 5-field structs).
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).parent / "data"

TURN = pa.struct([("cutoff_date", pa.string()), ("mask", pa.bool_()), ("role", pa.string()),
                  ("system_prompt", pa.string()), ("text", pa.string())])
SCHEMA = pa.schema([("instance_id", pa.string()), ("model_name", pa.string()), ("target", pa.bool_()),
                    ("trajectory", pa.list_(TURN)), ("exit_status", pa.string()),
                    ("generated_patch", pa.string()), ("eval_logs", pa.string())])

FOOTER = "\n(Open file: n/a)\n(Current directory: /synth)\nbash-$"


def system(prompt="SYNTHETIC SETTING: you are a test agent.\nCOMMANDS:\nopen <path>\nsubmit"):
    return {"cutoff_date": "01.01.2023", "mask": False, "role": "system", "system_prompt": prompt, "text": None}


def user(text, **extra):
    turn = {"cutoff_date": None, "mask": False, "role": "user", "system_prompt": None, "text": text}
    turn.update(extra)
    return turn


def ai(text):
    return {"cutoff_date": None, "mask": True, "role": "ai", "system_prompt": None, "text": text}


ISSUE = "Synthetic issue: parse() drops the last field.\nISSUE:\nparse('a,b') returns ['a']." + FOOTER

rows0 = [
    {"instance_id": "synth__repo-1", "model_name": "swe-agent-llama-70b", "target": True,
     "trajectory": [
         system(), user(ISSUE),
         ai("Look at the file first.\n\n```\nopen parse.py\n```"),
         user("[File: parse.py (3 lines total)]\n1:def parse(s):\n2:    return s.split(',')[:-1]" + FOOTER),
         ai("Two commands here, but the source records no calls.\n\n```\nedit 2:2\n    return s.split(',')\nend_of_edit\n```\n\nThen:\n\n```\nsubmit\n```"),
         user("Your changes look good." + FOOTER),
         ai("Done.\n\n```\nsubmit\n```"),
     ],
     "exit_status": "submitted", "generated_patch": "diff --git a/parse.py b/parse.py\n", "eval_logs": "PASSED"},
    {"instance_id": "synth__repo-2", "model_name": "swe-agent-llama-8b", "target": False,
     "trajectory": [system(), user("Synthetic issue with no agent turn." + FOOTER)],
     "exit_status": "early_exit", "generated_patch": None, "eval_logs": None},
]

rows1 = [
    {"instance_id": "synth__repo-3", "model_name": "swe-agent-llama-405b", "target": False,
     "trajectory": [], "exit_status": None, "generated_patch": "", "eval_logs": None},
    {"instance_id": "synth__repo-1", "model_name": "swe-agent-llama-70b", "target": False,
     "trajectory": [
         system(None), user(ISSUE),
         user("A second user turn before any agent turn." + FOOTER),
         ai(None),
         user("Observation after a null agent turn." + FOOTER, cutoff_date="02.02.2023", system_prompt="stray"),
         ai("Last turn."),
     ],
     "exit_status": "submitted (exit_context)", "generated_patch": None, "eval_logs": None},
]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train-00000-of-00012.parquet", rows0), ("train-00001-of-00012.parquet", rows1)):
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), OUT / name)
        print(OUT / name, len(rows))
