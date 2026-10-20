"""AgentDojo trajectories (sungjuncho/agentdojo-trajectories): tool-using runs under prompt injection.

One run per JSON file ``<model>/<suite>/<user_task>/<attack>/<injection_task>.json``
(15,095 files; the repository has no card and no Parquet export). Messages
go through the shared parser, with three source-specific steps first:

- A message's content is a list holding one ``{"type": "text", "content": ...}``
  block (not ATIF's text-part shape), so it is flattened to its string and
  ``content_form: "blocks"`` records the original form.
- Each tool message repeats the call it answers as ``tool_call``; the copy is
  dropped when it equals the call (``tool_call_copy: "identical"`` marks
  this) and kept otherwise. ``placeholder_args``, always null, and a null
  ``error`` are dropped; a non-null tool ``error`` is kept and reported as
  the result's error.
- Calls and results pair by tool call id.

``utility`` and ``security`` are run-level labels from the AgentDojo
harness. When a run has no injection task the harness returns ``security``
True as a placeholder (276 runs), so those labels carry
``source_ref.placeholder: true``. 13 runs have neither label. Where in the
trace an injection appears is not recorded, so ``injections`` (vector id to
text) stays in ``dataset`` and no step is labelled from a text search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from ..formats.messages import messages_to_trajectory
from ..record import TRAJECTORY, Run
from ._base import Dataset


def _flatten(content: Any):
    """(content for the parser, True if it was a list of text blocks)."""
    if (isinstance(content, list) and content
            and all(isinstance(b, dict) and b.get("type") == "text" and set(b) == {"type", "content"} for b in content)):
        return "".join(b["content"] for b in content), True
    return content, False


def _prepare(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: Dict[str, Dict[str, Any]] = {}
    out = []
    for msg in messages:
        m = dict(msg)
        if "content" in m:
            m["content"], blocks = _flatten(m["content"])
            if blocks:
                m["content_form"] = "blocks"
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                calls[tc.get("id")] = tc
                tc = dict(tc)
                if tc.get("placeholder_args", 0) is None:
                    del tc["placeholder_args"]
                tcs.append(tc)
            m["tool_calls"] = tcs
        if m.get("role") == "tool":
            if "tool_call" in m and m["tool_call"] == calls.get(m.get("tool_call_id")):
                del m["tool_call"]
                m["tool_call_copy"] = "identical"
            if m.get("error", 0) is None:
                del m["error"]
        out.append(m)
    return out


class AgentDojo(Dataset):
    name = "agentdojo"
    format = "agentdojo-json"

    def records(self, root: Path, suite: str = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"agentdojo takes suite= only, got {sorted(options)}")
        paths = sorted(p for p in root.glob("*/*/*/*/*.json") if not p.relative_to(root).parts[0].startswith("."))
        for path in paths:
            rel = path.relative_to(root).as_posix()
            model, suite_name = rel.split("/")[:2]
            if suite is not None and suite_name != suite:
                continue
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            yield rel[: -len(".json")], raw, {"file": rel, "config": f"{model}/{suite_name}", "split": None}

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        labels = []
        no_attack = raw.get("injection_task_id") is None
        for field in ("utility", "security"):
            if field in raw:
                ref: Dict[str, Any] = {"field": field}
                if field == "security" and no_attack:
                    ref["placeholder"] = True  # the harness returns True when there is no injection task
                labels.append({"name": field, "target": {"kind": "run", "address": {"trajectory": key}},
                               "value": raw[field], "label_source": "agentdojo_harness", "source_ref": ref})
        for field in ("injection_task_id", "attack_type"):
            labels.append({"name": field, "target": {"kind": "run", "address": {"trajectory": key}},
                           "value": raw.get(field), "label_source": "agentdojo", "source_ref": {"field": field}})
        utility = raw.get("utility")
        ext = {
            "source": self.source(key, where),
            "task": {"id": f"{raw.get('suite_name')}/{raw.get('user_task_id')}", "text": None},
            "outcome": {"success": utility if isinstance(utility, bool) else None, "score": None,
                        "raw": {k: raw[k] for k in ("utility", "security") if k in raw}},
            "labels": labels,
            "agents": [],
            "tools": {},
            "dataset": {k: v for k, v in raw.items() if k != "messages"},
        }
        traj = messages_to_trajectory(_prepare(raw.get("messages") or []), trajectory_id=key,
                                      agent_name="agentdojo", agent_version=str(raw.get("agentdojo_package_version") or "unknown"),
                                      model_name=raw.get("pipeline_name"), ext=ext)
        return Run(traj, TRAJECTORY)


dataset = AgentDojo()
