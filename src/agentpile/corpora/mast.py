"""MAST (mcemri/MAST-Data): multi-agent system logs with 14 run-level failure-mode labels.

One run per element of ``MAD_full_dataset.json``. The trajectory is a single
log string written in one of seven framework grammars, and the labels carry no
location, so the whole log is one agent step marked ``raw_log`` in the
extension; it is not split into turns and no tool call is guessed from the
text. The framework (``mas_name``) and model family (``llm_name``) are true of
the whole run and go in the ATIF agent block. The labels come from an LLM
judge (dataset card), and a null label means the annotation is unavailable,
so it stays null. The run id is ``trace.key:trace.index``; ``trace_id`` is not
unique across keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run
from ._base import Dataset

FILE = "MAD_full_dataset.json"

FAILURE_MODES = {
    "1.1": "Disobey Task Specification",
    "1.2": "Disobey Role Specification",
    "1.3": "Step Repetition",
    "1.4": "Loss of Conversation History",
    "1.5": "Unaware of Termination Conditions",
    "2.1": "Conversation Reset",
    "2.2": "Fail to Ask for Clarification",
    "2.3": "Task Derailment",
    "2.4": "Information Withholding",
    "2.5": "Ignored Other Agent's Input",
    "2.6": "Reasoning-Action Mismatch",
    "3.1": "Premature Termination",
    "3.2": "No or Incomplete Verification",
    "3.3": "Incorrect Verification",
}


class Mast(Dataset):
    name = "mast"
    format = "mast-json"
    incremental = True  # one 200 MB array: a limit saves only the README

    def records(self, root: Path, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"mast takes no options, got {sorted(options)}")
        with self.ready(root / FILE).open("r", encoding="utf-8") as fh:
            rows = json.load(fh)  # one 200 MB array; the file has no record-level framing
        for row in rows:
            trace = row.get("trace") or {}
            yield f"{trace.get('key')}:{trace.get('index')}", row, {"file": FILE, "config": "default", "split": "train"}

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        trace = raw.get("trace") or {}
        log = trace.get("trajectory")
        sx: Dict[str, Any] = {"raw_log": True, "source_ref": "trace.trajectory"}
        message = log
        if not isinstance(log, str):
            message = "" if log is None else json.dumps(log, ensure_ascii=False)
            if log is None:
                sx["message_state"] = "null" if "trajectory" in trace else "missing"
            else:
                sx["source_fields"] = {"trajectory": log}
        annotation = raw.get("mast_annotation") or {}
        labels = [{"name": code, "target": {"kind": "run", "address": {"trajectory": key}},
                   "value": annotation.get(code), "label_source": "llm_judge",
                   "source_ref": {"field": f"mast_annotation.{code}", "mode": FAILURE_MODES.get(code)}}
                  for code in annotation]
        dataset = {k: v for k, v in raw.items() if k != "trace"}
        dataset["trace"] = {k: v for k, v in trace.items() if k != "trajectory"}
        ext = {
            "version": EXT_VERSION,
            "source": self.source(key, where),
            "task": {"id": None, "text": None},
            "outcome": {"success": None, "score": None, "raw": None},
            "labels": labels,
            "agents": [],
            "tools": {},
            "dataset": dataset,
        }
        agent = {"name": str(raw.get("mas_name") or "unknown"), "version": "unknown"}
        if raw.get("llm_name") is not None:
            agent["model_name"] = str(raw["llm_name"])
        traj = {"schema_version": ATIF_VERSION, "session_id": key, "trajectory_id": key, "agent": agent,
                "steps": [{"step_id": 1, "source": "agent", "message": message, "extra": {EXT: sx}}],
                "extra": {EXT: ext}}
        return Run(traj, TRAJECTORY)


dataset = Mast()
