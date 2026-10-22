"""Who&When (Kevin355/Who_and_When): failed multi-agent runs with the decisive error step and agent.

One run per JSON file ``Who&When/<subset>/N.json`` (the Parquet config files
drop fields and hold three out-of-range ``mistake_step`` values, so they are
not read). Each ``history`` entry becomes one ATIF step, in order, with the
speaking agent in the extension. Nothing else is inferred:

- Algorithm-Generated (CaptainAgent): the ``role`` field (user or assistant)
  does not identify the speaker, so every entry is an agent step and the agent
  is ``name``. Computer_terminal entries are its own agent, not paired with an
  earlier code block, because the data records no call.
- Hand-Crafted (Magentic-One): ``history[0]`` is the human question and
  becomes a user step; the agent is ``role`` without its parenthetical
  ("Orchestrator (-> WebSurfer)" is Orchestrator), and the raw role is kept.
- ``mistake_step`` is a 0-based index over ``history``, so it labels ATIF step
  ``mistake_step + 1``. ``mistake_agent`` labels an agent, matched to the run's
  roster ignoring case ("Websurfer"); the raw strings stay in ``source_ref``.
- ``is_correct`` (Algorithm-Generated) and ``is_corrected`` (Hand-Crafted)
  both give the outcome. All 184 runs are failures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run
from ._base import Dataset

SUBSETS = {"Algorithm-Generated": "CaptainAgent", "Hand-Crafted": "Magentic-One"}
_PAREN = re.compile(r"\s*\(.*\)\s*$")
_RUN_LABELS = ("mistake_type", "labels")


def _number(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 1 << 30


def _actor(entry: Dict[str, Any], subset: str) -> Optional[str]:
    if subset == "Algorithm-Generated":
        return entry.get("name")
    role = entry.get("role")
    return _PAREN.sub("", role) if isinstance(role, str) else None


class WhoAndWhen(Dataset):
    name = "who_and_when"
    format = "who_and_when-json"
    incremental = True

    def records(self, root: Path, subset: Optional[str] = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"who_and_when takes subset= only, got {sorted(options)}")
        if subset is not None and subset not in SUBSETS:
            raise ValueError(f"subset must be one of {sorted(SUBSETS)}, got {subset!r}")
        for sub in SUBSETS:
            if subset is not None and sub != subset:
                continue
            for path in self.paths(root, f"Who&When/{sub}/*.json", key=_number):
                rel = path.relative_to(root).as_posix()
                with self.ready(path).open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                yield f"{sub}/{path.stem}", raw, {"file": rel, "config": sub, "split": "train"}

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        sub = where["config"]
        steps: List[Dict[str, Any]] = []
        roster: List[str] = []
        for i, entry in enumerate(raw.get("history") or []):
            human = sub == "Hand-Crafted" and entry.get("role") == "human"
            who = None if human else _actor(entry, sub)
            if who is not None and who not in roster:
                roster.append(who)
            sx: Dict[str, Any] = {"source_ref": i, "role": entry.get("role")}
            if who is not None:
                sx["actor"] = who
            extra_keys = {k: v for k, v in entry.items() if k not in ("content", "role", "name")}
            if extra_keys:
                sx["source_fields"] = extra_keys
            content = entry.get("content")
            step = {"step_id": i + 1, "source": "user" if human else "agent", "message": content}
            if content is None or "content" not in entry:
                step["message"] = ""
                sx["message_state"] = "null" if "content" in entry else "missing"
            step["extra"] = {EXT: sx}
            steps.append(step)

        prompts = raw.get("system_prompt") or {}
        agents = [{"id": a, "name": a, "instructions": prompts.get(a) if isinstance(prompts, dict) else None,
                   "model": None, "tools": [], "handoffs": []} for a in roster]

        labels: List[Dict[str, Any]] = []
        ms = raw.get("mistake_step")
        try:
            ms_int = int(ms)
        except (TypeError, ValueError):
            ms_int = None
        if ms_int is not None:
            in_range = 0 <= ms_int < len(steps)
            labels.append({"name": "mistake_step",
                           "target": {"kind": "step", "address": {"trajectory": key, "step": ms_int + 1}},
                           "value": True, "label_source": "who_and_when",
                           "source_ref": {"field": "mistake_step", "value": ms, "in_range": in_range}})
            if raw.get("mistake_reason") is not None:
                labels.append({"name": "mistake_reason",
                               "target": {"kind": "step", "address": {"trajectory": key, "step": ms_int + 1}},
                               "value": raw["mistake_reason"], "label_source": "who_and_when",
                               "source_ref": {"field": "mistake_reason"}})
        agent_raw = raw.get("mistake_agent")
        if agent_raw is not None:
            match = next((a for a in roster if a.lower() == str(agent_raw).lower()), None)
            labels.append({"name": "mistake_agent",
                           "target": {"kind": "agent", "address": {"trajectory": key, "agent": match or str(agent_raw)}},
                           "value": True, "label_source": "who_and_when",
                           "source_ref": {"field": "mistake_agent", "value": agent_raw, "resolved": match is not None}})
        for field in _RUN_LABELS:
            if raw.get(field) is not None:
                labels.append({"name": field, "target": {"kind": "run", "address": {"trajectory": key}},
                               "value": raw[field], "label_source": "who_and_when", "source_ref": {"field": field}})

        outcome_key = "is_correct" if "is_correct" in raw else "is_corrected" if "is_corrected" in raw else None
        success = raw.get(outcome_key) if outcome_key else None
        ext = {
            "version": EXT_VERSION,
            "source": self.source(key, where),
            "task": {"id": raw.get("question_ID"), "text": raw.get("question")},
            "outcome": {"success": success if isinstance(success, bool) else None, "score": None,
                        "raw": {outcome_key: success} if outcome_key else None},
            "labels": labels,
            "agents": agents,
            "tools": {},
            "dataset": {k: v for k, v in raw.items() if k != "history"},
        }
        traj = {"schema_version": ATIF_VERSION, "session_id": key, "trajectory_id": key,
                "agent": {"name": SUBSETS[sub], "version": "unknown"}, "steps": steps, "extra": {EXT: ext}}
        if not steps:
            traj["steps"] = [{"step_id": 1, "source": "system", "message": "",
                              "extra": {EXT: {"message_state": "missing", "empty_source": True}}}]
        return Run(traj, TRAJECTORY)


dataset = WhoAndWhen()
