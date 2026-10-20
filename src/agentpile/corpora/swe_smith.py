"""SWE-smith trajectories (SWE-bench/SWE-smith-trajectories), tool split: SWE-agent runs on SWE-smith task instances.

One run per row of the eight ``data/tool-*.parquet`` shards (24,100 rows; the
xml and ticks splits and the unlisted ``data/train-*`` shards are not read).
``messages`` is a JSON-encoded string, decoded per row and mapped by the shared
OpenAI-style parser, so calls and results pair by id and nothing else.

The split holds two message formats, told apart by what the messages contain
(recorded as ``dataset["message_format"]``):

- ``native`` (9,116 rows): assistant messages carry ``tool_calls`` (one call per
  message, an id on each) and a ``tool`` message answers it through
  ``tool_call_ids``. That list holds exactly one id in every tool message, so it
  is read as the parser's ``tool_call_id``; the list itself stays in
  ``result_fields`` (a list of any other length leaves the result unpaired). The
  ``thought``, ``action``, ``agent`` and ``message_type`` keys and each call's
  ``index`` stay in ``source_fields``. The messages record ``agent: main``, which
  becomes the step actor.
- ``text`` (14,984 rows): every action is written inside the assistant text as a
  ``<function=NAME>`` block and every observation is a plain user message. They
  load as plain messages exactly as the source labels them: no call is parsed
  out of the text, the user turns stay user steps, and nothing is paired.
  Whether to keep these rows is open (``message_format=`` selects one format).

Other mapping choices:

- A content list of one text part (the native user and tool messages) becomes
  that part's string, marked ``content_form: text_parts`` in ``source_fields`` or
  ``result_fields``. Any other list is left as the source gave it.
- ``record_key`` and the run id are ``traj_id`` (unique in the split);
  ``instance_id`` is the task id and is not unique (up to 21 runs each). The row
  number inside the shard is ``source["row"]``. The task text is the first user
  message, which in all 24,100 rows is the prompt that opens with
  ``<uploaded_files>`` (23,706 of them hold a ``<pr_description>`` block).
- ``resolved`` is the outcome and a run-level label. The card does not define
  it; by name it is the verdict of the task's tests. ``patch`` does not match
  the row's run in most rows (stage-0 heuristic), so it is kept in ``dataset``
  and never read as an outcome or output.
- The agent is SWE-agent as the card states, and ``model`` goes to
  ``agent.model_name``. The card names no version.
- Tools are the ``---- BEGIN FUNCTION #n: name ----`` blocks of a system prompt
  (text rows); the id is the name and the definition text stays in the system
  step. Native rows record no tool definitions, so their ``tools`` is empty and
  the names appear only on the calls.
- A row whose messages are empty, or do not decode to a list of objects, still
  becomes a run; an undecodable string is kept as ``dataset["messages_raw"]``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..formats.messages import messages_to_trajectory
from ..record import TRAJECTORY, Run
from ._base import Dataset

FORMATS = ("native", "text")
BATCH = 16  # rows decoded at a time; the longest messages string is 660 kB
_BLOCK = re.compile(r"^---- BEGIN FUNCTION #(\d+): ([^\n]+?) ----[ \t]*$.*?^---- END FUNCTION #\1 ----[ \t]*$",
                    re.M | re.S)
PATCH_NOTE = ("Stage-0 heuristic: in most rows the patch names no file that the row's messages mention. "
              "It is not this run's output or outcome.")


def _pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("reading Parquet needs the hf extra: pip install 'agentpile[hf]'") from exc
    return pq


def _decode(text: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """(messages, ok): the decoded message objects, or ``([], False)`` when ``text`` is not a JSON list of objects."""
    try:
        messages = json.loads(text)
    except (TypeError, ValueError):
        return [], False
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        return [], False
    return messages, True


def _message_format(messages: List[Dict[str, Any]]) -> Optional[str]:
    if not messages:
        return None
    return "native" if any(m.get("role") == "tool" or m.get("tool_calls") for m in messages) else "text"


def _prepare(msg: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of one message in the shape the shared parser reads."""
    out = dict(msg)
    ids = msg.get("tool_call_ids")
    if msg.get("role") == "tool" and "tool_call_id" not in msg and isinstance(ids, list) and len(ids) == 1 \
            and isinstance(ids[0], str):
        out["tool_call_id"] = ids[0]
    content = msg.get("content")
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict) \
            and set(content[0]) == {"type", "text"} and content[0]["type"] == "text" \
            and isinstance(content[0]["text"], str):
        out["content"] = content[0]["text"]
        out["content_form"] = "text_parts"
    return out


def _tools(system_text: Any) -> Dict[str, Dict[str, Any]]:
    """Tools declared by the delimited function blocks of a system prompt, keyed by name."""
    tools: Dict[str, Dict[str, Any]] = {}
    if not isinstance(system_text, str):
        return tools
    for block in _BLOCK.finditer(system_text):
        number, name = block.group(1), block.group(2)
        tid = name if name not in tools else f"{name}#{number}"
        tools[tid] = {"name": name, "agent_or_server": None, "type": "function", "definition": None,
                      "definition_format": None, "atif_index": None, "permission": "unknown",
                      "permission_from": "unknown"}
    return tools


class SweSmith(Dataset):
    name = "swe_smith"
    format = "swe_smith-parquet"

    def records(self, root: Path, message_format: Optional[str] = None,
                **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"swe_smith takes message_format= only, got {sorted(options)}")
        if message_format is not None and message_format not in FORMATS:
            raise ValueError(f"message_format must be one of {list(FORMATS)}, got {message_format!r}")
        pq = _pyarrow()
        for path in sorted((root / "data").glob("tool-*.parquet")):
            rel = path.relative_to(root).as_posix()
            row_no = 0
            with pq.ParquetFile(path) as pf:
                for batch in pf.iter_batches(batch_size=BATCH):
                    for raw in batch.to_pylist():
                        where = {"file": rel, "config": "default", "split": "tool", "row": row_no}
                        row_no += 1
                        if message_format is not None and _message_format(_decode(raw.get("messages"))[0]) != message_format:
                            continue
                        key = raw.get("traj_id")
                        yield (key if isinstance(key, str) and key else f"{rel}#{where['row']}"), raw, where

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        messages, ok = _decode(raw.get("messages"))
        prepared = [_prepare(m) for m in messages]
        system = next((m for m in prepared if m.get("role") == "system"), None)
        system_text = system.get("content") if system else None
        first_user = next((m for m in prepared if m.get("role") == "user"), None)
        task_text = first_user.get("content") if first_user and isinstance(first_user.get("content"), str) else None

        actors: List[str] = []
        for m in prepared:
            who = m.get("agent")
            if m.get("role") == "assistant" and isinstance(who, str) and who not in actors:
                actors.append(who)
        agents = [{"id": a, "name": a, "instructions": None, "model": None, "tools": [], "handoffs": []}
                  for a in actors]

        resolved = raw.get("resolved")
        labels: List[Dict[str, Any]] = []
        if resolved is not None:
            labels.append({"name": "resolved", "target": {"kind": "run", "address": {"trajectory": key}},
                           "value": resolved, "label_source": "swe_smith", "source_ref": {"field": "resolved"}})

        dataset = {k: v for k, v in raw.items() if k != "messages"}
        dataset["message_format"] = _message_format(messages)
        dataset["system_prompt_sha256"] = (hashlib.sha256(system_text.encode("utf-8")).hexdigest()
                                           if isinstance(system_text, str) else None)
        dataset["patch_note"] = PATCH_NOTE
        if not ok:
            dataset["messages_raw"] = raw.get("messages")

        source = self.source(key, where)
        source["row"] = where.get("row")
        ext = {
            "source": source,
            "task": {"id": raw.get("instance_id"), "text": task_text},
            "outcome": {"success": resolved if isinstance(resolved, bool) else None, "score": None,
                        "raw": {"resolved": resolved} if "resolved" in raw else None},
            "labels": labels,
            "agents": agents,
            "tools": _tools(system_text),
            "dataset": dataset,
        }
        model = raw.get("model")
        traj = messages_to_trajectory(prepared, trajectory_id=key, agent_name="SWE-agent",
                                      model_name=model if isinstance(model, str) else None, ext=ext,
                                      actor=lambda m: m.get("agent") if isinstance(m.get("agent"), str) else None)
        return Run(traj, TRAJECTORY)


dataset = SweSmith()
