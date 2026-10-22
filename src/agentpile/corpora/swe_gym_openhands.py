"""SWE-Gym OpenHands (SWE-Gym/OpenHands-Sampled-Trajectories): OpenHands rollouts on SWE-Gym instances, with the resolved flag.

One run per row of the three ``data/train.raw-*.parquet`` shards (the only split
is named ``train.raw``), keyed on the global row index in shard order, because
neither ``instance_id`` nor ``instance_id`` plus ``run_id`` is unique. The row
is an OpenAI-style chat (``messages``) and is read by the shared message parser:

- Calls and results pair by ``tool_call_id`` only, on the step of the assistant
  message that issued the call. A call id is not unique within a run (an id is
  reused after its call was answered), so the parser binds a result to the most
  recent call with that id; in this data that is always a call of the assistant
  message right before the tool message. Several calls in one assistant turn
  stay in source order on one step, and the source does not say whether they
  ran in parallel. The last assistant message of most runs has calls with no
  result; their steps carry no observation.
- Tool result content stays verbatim, including its leading ``OBSERVATION:``.
- An assistant message with tool calls often has an empty string (57,458) or a
  null (4,360) content. The empty string is the message, and the null is marked
  ``message_state: null`` on the step.
- The run belongs to the OpenHands scaffold (the dataset name); no message names
  an agent, so no step has an actor. The model name is the ``run_id`` prefix
  before ``_maxiter_`` and the whole ``run_id`` stays in ``dataset``.
- ``tools`` is the same three tool schemas in every row. They become the ATIF
  tool definitions and the extension tools (id = function name, permission
  unknown). A call whose function name is one of them carries that ``tool_id``.
  A call naming an undefined function (8 calls, all unanswered) carries none.
- The Parquet struct reader reports an absent member as null. Members that are
  null that way (``function_call``, ``name``, ``tool_call_id``, ``tool_calls``,
  a call's ``index``, null members of a tool schema) are dropped, since the file
  cannot tell absent from null, and ``source.null_members_dropped`` names them.
  ``content`` is kept: a null content is marked, never dropped.
- ``resolved`` is the outcome and a run label. It equals
  ``test_result.report.resolved`` in every row; the card defines neither. The
  four other report flags are run labels too. The remaining row fields
  (``instance_id``, ``run_id``, ``test_result`` with the patch and the logs,
  which can reach hundreds of thousands of characters) stay in ``dataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..formats.messages import messages_to_trajectory
from ..record import TRAJECTORY, Run
from ._base import Dataset

SHARDS = "data/train.raw-*.parquet"
AGENT = "OpenHands"
BATCH_ROWS = 64  # rows per Arrow batch; one row can hold a few hundred thousand characters of test output

# Members the Parquet union schema pads with null when a message or call does not have them.
_PADDED_MEMBERS = ("function_call", "name", "tool_call_id", "tool_calls")
_REPORT_FLAGS = ("empty_generation", "error_eval", "failed_apply_patch", "test_timeout")


def _clean_call(call: Any, dropped: Set[str]) -> Any:
    if isinstance(call, dict) and "index" in call and call["index"] is None:
        dropped.add("messages[].tool_calls[].index")
        return {k: v for k, v in call.items() if k != "index"}
    return call


def _clean_message(msg: Any, dropped: Set[str]) -> Any:
    if not isinstance(msg, dict):
        return msg
    out: Dict[str, Any] = {}
    for key, value in msg.items():
        if value is None and key in _PADDED_MEMBERS:
            dropped.add(f"messages[].{key}")
            continue
        if key == "tool_calls" and isinstance(value, list):
            value = [_clean_call(c, dropped) for c in value]
        out[key] = value
    return out


def _drop_nulls(value: Any, hit: List[bool]) -> Any:
    """A tool schema without the nulls the struct reader pads in; null list items stay."""
    if isinstance(value, dict):
        kept = {k: _drop_nulls(v, hit) for k, v in value.items() if v is not None}
        if len(kept) != len(value):
            hit.append(True)
        return kept
    if isinstance(value, list):
        return [_drop_nulls(v, hit) for v in value]
    return value


def _model_name(run_id: Any) -> Optional[str]:
    if isinstance(run_id, str) and "_maxiter_" in run_id:
        return run_id.split("_maxiter_", 1)[0] or None
    return None


def _first_user_text(messages: List[Any]) -> Optional[str]:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else None
    return None


class SweGymOpenHands(Dataset):
    name = "swe_gym_openhands"
    format = "swe_gym_openhands-parquet"
    incremental = True

    def records(self, root: Path, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"swe_gym_openhands takes no options, got {sorted(options)}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("reading swe_gym_openhands needs the hf extra: pip install 'agentpile[hf]'") from exc
        shards = self.paths(root, SHARDS)
        if not shards:
            raise FileNotFoundError(f"swe_gym_openhands: no {SHARDS} under {root}")
        index = 0
        for path in shards:
            rel = path.relative_to(root).as_posix()
            row_in_file = 0
            with pq.ParquetFile(self.ready(path)) as pf:
                for batch in pf.iter_batches(batch_size=BATCH_ROWS):
                    for row in batch.to_pylist():
                        yield str(index), row, {"file": rel, "config": "default", "split": "train.raw",
                                                "row_in_file": row_in_file}
                        index += 1
                        row_in_file += 1

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        dropped: Set[str] = set()
        messages = [_clean_message(m, dropped) for m in raw.get("messages") or []]
        hit: List[bool] = []
        defs = None if raw.get("tools") is None else [_drop_nulls(t, hit) for t in raw["tools"]]
        if hit:
            dropped.add("tools[].*")

        tools: Dict[str, Dict[str, Any]] = {}
        by_name: Dict[str, Optional[str]] = {}  # function name -> tool id, None for a name the row defines twice
        for i, d in enumerate(defs or []):
            fn = d.get("function") if isinstance(d, dict) else None
            fname = fn.get("name") if isinstance(fn, dict) else None
            tid = fname if isinstance(fname, str) and fname and fname not in tools else f"tool-{i}"
            tools[tid] = {"name": fname if isinstance(fname, str) else "", "agent_or_server": None,
                          "type": d.get("type") if isinstance(d, dict) else None, "definition": d,
                          "definition_format": "openai-function", "atif_index": i,
                          "permission": "unknown", "permission_from": "unknown"}
            if isinstance(fname, str):
                by_name[fname] = None if fname in by_name else tid

        source = self.source(key, where)
        source.update(row_in_file=where.get("row_in_file"), instance_id=raw.get("instance_id"),
                      run_id=raw.get("run_id"))
        if dropped:
            source["null_members_dropped"] = sorted(dropped)

        resolved = raw.get("resolved")
        test_result = raw.get("test_result")
        report = test_result.get("report") if isinstance(test_result, dict) else None
        labels: List[Dict[str, Any]] = []
        target = {"kind": "run", "address": {"trajectory": key}}
        if isinstance(resolved, bool):
            labels.append({"name": "resolved", "target": target, "value": resolved, "label_source": "swe_gym_eval",
                           "source_ref": {"field": "resolved"}})
        for flag in _REPORT_FLAGS:
            value = report.get(flag) if isinstance(report, dict) else None
            if isinstance(value, bool):
                labels.append({"name": flag, "target": target, "value": value, "label_source": "swe_gym_eval",
                               "source_ref": {"field": f"test_result.report.{flag}"}})

        ext = {
            "source": source,
            "task": {"id": raw.get("instance_id"), "text": _first_user_text(messages)},
            "outcome": {"success": resolved if isinstance(resolved, bool) else None, "score": None,
                        "raw": {"resolved": resolved}},
            "labels": labels,
            "agents": [],
            "tools": tools,
            "dataset": {k: v for k, v in raw.items() if k not in ("messages", "tools")},
        }
        traj = messages_to_trajectory(messages, trajectory_id=key, agent_name=AGENT,
                                      model_name=_model_name(raw.get("run_id")), tool_definitions=defs, ext=ext)
        if by_name:
            for step in traj["steps"]:
                sx = (step.get("extra") or {}).get("agentpile")
                for call in step.get("tool_calls") or []:
                    tid = by_name.get(call["function_name"])
                    if tid is not None:
                        sx.setdefault("calls", {}).setdefault(call["tool_call_id"], {})["tool_id"] = tid
        return Run(traj, TRAJECTORY)


dataset = SweGymOpenHands()
