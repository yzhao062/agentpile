"""SWE-rebench OpenHands trajectories (nebius/SWE-rebench-openhands-trajectories).

One run per row of ``trajectories.parquet``: one OpenHands rollout of
Qwen3-Coder-480B-A35B-Instruct on one SWE-rebench issue (67,074 rows, a 2 GB
file read in small batches). The ``trajectory`` column is an OpenAI-style chat
list and goes through the shared message parser:

- ``system`` and ``user`` messages become system and user steps. The 306 later
  user messages (OpenHands "Please continue ..." prompts and tool-argument
  error feedback) stay user steps, as the source labels them.
- Each ``assistant`` message is one agent step with its tool calls, and the
  ``tool`` messages that answer it become the observation of that same step.
  Calls and results are paired by ``tool_call_id`` only. The last assistant
  message of a run (mostly ``finish``) has no result, so its observation is
  absent; four turns carry two calls with two results in call order.
- Parquet pads every message struct with the union of all members, so a
  ``name`` on an assistant message or ``tool_calls`` on a tool message arrives
  as null. A null member the dataset card does not define for that role is
  dropped before parsing; a role's own members (an assistant turn without
  calls has ``tool_calls`` null) are kept as read.

Dataset-level statements, true of every row and not stored per row: the card
(``README.md``) names the scaffold OpenHands v0.54.0 and the model
Qwen/Qwen3-Coder-480B-A35B-Instruct, so the ATIF ``agent`` block carries them
and ``dataset["agent_from"]`` says where they came from. The sibling file
``tools.json`` (five OpenAI function schemas; the card never mentions it)
gives ``agent.tool_definitions`` and the run's tool entries
(``openhands.<function name>``, permission unknown). The row-level ``tools``
column equals ``tools.json`` once the null padding is dropped, in all 67,074
rows at the pinned revision, so it is not read again per row and is listed in
``lost_fields``. ``config.toml`` is a generic OpenHands example configuration
that lists other model groups and a browsing switch no defined tool uses; it
is pinned with the download but is not read as a per-run setting.

- A call carries ``tool_id`` when its function name matches a definition, and
  its ``type`` (always ``function``) in ``source_fields``.
- ``resolved`` gives ``outcome.success``; the card does not say how it was
  evaluated. ``resolved``, ``exit_status``, ``gen_tests_correct`` and
  ``pred_passes_gen_tests`` are run-level labels with the values as read
  (the two test fields are 0.0 or 1.0 or null, although the card calls them
  counts). ``model_patch`` is an artifact of the run, so it stays in ``dataset``.
- The issue text exists only inside the first user message, wrapped in the
  OpenHands task template. ``task.text`` stays null rather than cutting the
  template out; ``task.id`` is ``instance_id``.
- No agent is named by any field, so ``agents`` is empty. There are no
  timestamps, token counts or costs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from ..formats.messages import messages_to_trajectory
from ..record import EXT, TRAJECTORY, Run
from ._base import Dataset

TRAJECTORIES = "trajectories.parquet"
TOOLS = "tools.json"

AGENT_NAME = "OpenHands"
AGENT_VERSION = "0.54.0"
MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

# Row columns read; the row-level ``tools`` column is replaced by tools.json.
COLUMNS = ("trajectory_id", "instance_id", "repo", "trajectory", "model_patch", "exit_status", "resolved",
           "gen_tests_correct", "pred_passes_gen_tests")
RUN_LABELS = ("resolved", "exit_status", "gen_tests_correct", "pred_passes_gen_tests")

# The members the dataset card gives each role; any other member read as null is Parquet padding.
ROLE_MEMBERS = {
    "system": ("role", "content"),
    "user": ("role", "content"),
    "assistant": ("role", "content", "tool_calls"),
    "tool": ("role", "content", "name", "tool_call_id"),
}


def _unpad(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Drop members that are null and not defined for the message's role (the union-schema padding)."""
    members = ROLE_MEMBERS.get(msg.get("role"))
    if members is None:
        return msg
    return {k: v for k, v in msg.items() if v is not None or k in members}


def _tool_entries(defs: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """(tool id -> extension tool entry, function name -> tool id) for the tools.json definitions."""
    tools: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, str] = {}
    for i, definition in enumerate(defs):
        fn = definition.get("function") if isinstance(definition, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or name in by_name:
            raise ValueError(f"{TOOLS}: definition {i} has no unique function name ({name!r})")
        tid = f"openhands.{name}"
        by_name[name] = tid
        tools[tid] = {"name": name, "agent_or_server": None, "type": definition.get("type"),
                      "definition": definition, "definition_format": "openai", "atif_index": i,
                      "permission": "unknown", "permission_from": "unknown"}
    return tools, by_name


class SweRebenchOpenHands(Dataset):
    name = "swe_rebench_openhands"
    format = "swe_rebench_openhands-parquet"
    incremental = True  # one Parquet file: a limit saves README.md and config.toml
    upfront = (TOOLS,)  # the tool definitions are shared by every run
    batch_rows = 64  # rows per Parquet batch: one row is about 250 KB uncompressed, a row group up to 1 GB

    def records(self, root: Path, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"swe_rebench_openhands takes no options, got {sorted(options)}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("loading swe_rebench_openhands needs the hf extra: pip install 'agentpile[hf]'") from exc
        defs = json.loads((root / TOOLS).read_text(encoding="utf-8"))
        row = 0
        with pq.ParquetFile(self.ready(root / TRAJECTORIES)) as pf:
            for batch in pf.iter_batches(batch_size=self.batch_rows, columns=list(COLUMNS)):
                for rec in batch.to_pylist():
                    yield rec["trajectory_id"], rec, {"file": TRAJECTORIES, "config": "default", "split": "train",
                                                      "row": row, "tool_definitions": defs}
                    row += 1

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        messages = [_unpad(m) for m in (raw.get("trajectory") or [])]
        defs = copy.deepcopy(where.get("tool_definitions") or [])
        tools, by_name = _tool_entries(defs)

        source = self.source(key, where)
        source["row_index"] = where.get("row")
        source["lost_fields"] = ["tools"]  # replaced by tools.json, equal in every row at the pinned revision

        resolved = raw.get("resolved")
        labels = [{"name": field, "target": {"kind": "run", "address": {"trajectory": key}}, "value": raw.get(field),
                   "label_source": self.name, "source_ref": {"field": field}} for field in RUN_LABELS]
        ext: Dict[str, Any] = {
            "source": source,
            "task": {"id": raw.get("instance_id"), "text": None},
            "outcome": {"success": bool(resolved) if resolved in (0, 1) else None, "score": None,
                        "raw": {"resolved": resolved}},
            "labels": labels,
            "agents": [],
            "tools": tools,
            "dataset": {"instance_id": raw.get("instance_id"), "repo": raw.get("repo"),
                        "model_patch": raw.get("model_patch"), "agent_from": "README.md",
                        "tool_definitions_from": TOOLS},
        }
        traj = messages_to_trajectory(messages, trajectory_id=key, agent_name=AGENT_NAME,
                                      agent_version=AGENT_VERSION, model_name=MODEL_NAME,
                                      tool_definitions=defs, ext=ext)

        # Tag each call with its tool definition and the call type; steps keep the message index in source_ref.
        for step in traj["steps"]:
            calls = step.get("tool_calls")
            if not calls:
                continue
            sx = step["extra"][EXT]
            raw_calls = messages[sx["source_ref"]].get("tool_calls") or []
            known = sx.get("calls") or {}
            for call, tc in zip(calls, raw_calls):
                cx = dict(known.get(call["tool_call_id"]) or {})
                if call["function_name"] in by_name:
                    cx["tool_id"] = by_name[call["function_name"]]
                if tc.get("type") is not None:
                    cx["source_fields"] = dict(cx.get("source_fields") or {}, type=tc["type"])
                if cx:
                    known[call["tool_call_id"]] = cx
            if known:
                sx["calls"] = known
        return Run(traj, TRAJECTORY)


dataset = SweRebenchOpenHands()
