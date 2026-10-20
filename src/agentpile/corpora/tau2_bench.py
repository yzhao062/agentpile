"""tau2-bench trajectories (AgentSuite/tau2-bench-trajectories): 30 agent models on 278 tasks each.

One run per JSON line of ``<model>.jsonl`` (30 files, 278 rows each: telecom
114, retail 114, airline 50). A row holds the full OpenAI-style message list,
``eval_result`` and ``meta``; the shared message parser reads the list. Nothing
the source does not say is inferred:

- Tool results carry no call id (no ``tool_call_id``, name or requestor), so
  the reader passes ``pair_by_position=True``: the k-th result in the block of
  ``tool`` messages that follows a call message answers that message's k-th
  call. This is the documented rule of this reader, not a stored link, and
  every such pairing is marked ``paired_by: "position"`` in the step's
  ``result_fields``. In the pinned data every block as long as its call
  message, except 167 call messages at the very end of ``max_steps`` runs; a
  call with no result stays without an observation, and a result that no open
  call can take would stay unpaired (``source_call_id`` null).
- Two actors call tools. An ``assistant`` message is an agent step with actor
  ``assistant`` (the model named by the file). A ``user`` message that carries
  ``tool_calls`` (the telecom user simulator acting on a device) is also an
  agent step, because ATIF allows calls on agent steps only; its actor is
  ``user`` and the original role stays in the step extension. A ``user``
  message without calls is an ordinary user step. Each call keeps its
  ``requestor`` in ``source_fields``. No call is marked concurrent: the
  source order inside a multi-call message is the only order there is.
- The source has no tool definitions, so ``tools`` is empty; tool names appear
  only in the calls (some are malformed or empty and are kept as given).
- Message 2 of every row is the string ``Hi! How can I help you today?``. The
  step carries ``scripted: true`` when that exact text sits at message index 1,
  and no step is given a ``model_name``, so no model is credited with it.
- Control tokens (``###STOP###``, ``###TRANSFER###``) and raw model markup stay
  in the text as given.
- The run is keyed ``<model file stem>/<task_name>/<id>``, with ``meta.id``
  stripped of a leading ``<task_name>_`` (ten files use the bare id; a row
  with no id would be keyed ``row<N>`` by its line number, with a null task
  id). ``task.id`` is ``<task_name>/<id>`` and joins the same task across models.
  The task has no single text, so ``task.text`` is null and the user scenario
  stays whole in ``dataset["meta"]``.
- ``outcome.score`` is ``eval_result.score``; ``success`` is true for 1.0,
  false for 0.0 and null otherwise. ``outcome.raw`` is the ``eval_result``
  dict. ``db_match`` has no DB component in telecom rows (use the score), and
  ``finish_reason`` ``max_steps`` marks truncated runs: both are run labels.
- Run-level timestamps, ``duration_seconds``, costs and the rest of ``meta``
  are kept in ``dataset``; the source has no per-step time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..formats.messages import messages_to_trajectory
from ..record import EXT, EXT_VERSION, TRAJECTORY, Run
from ._base import Dataset

LABEL_SOURCE = "tau2_bench"
GREETING = "Hi! How can I help you today?"


def _source_of(msg: Dict[str, Any]) -> Optional[str]:
    """A user message that calls tools must be an agent step; every other message keeps its role's source."""
    return "agent" if msg.get("role") == "user" and msg.get("tool_calls") else None


def _actor(msg: Dict[str, Any]) -> str:
    return "user" if msg.get("role") == "user" else "assistant"


def _task_key(row: Dict[str, Any]) -> Optional[str]:
    """``meta.id`` without a leading ``<task_name>_`` (ten files store the bare id), or None when absent."""
    ident = (row.get("meta") or {}).get("id")
    if not isinstance(ident, str) or not ident:
        return None
    prefix = f"{row.get('task_name')}_"
    return ident[len(prefix):] if ident.startswith(prefix) else ident


def _label(key: str, name: str, value: Any, field: str, **note: Any) -> Dict[str, Any]:
    return {"name": name, "target": {"kind": "run", "address": {"trajectory": key}}, "value": value,
            "label_source": LABEL_SOURCE, "source_ref": dict({"field": field}, **note)}


class Tau2Bench(Dataset):
    name = "tau2_bench"
    format = "tau2_bench-jsonl"

    def records(self, root: Path, model: Optional[str] = None, domain: Optional[str] = None,
                **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"tau2_bench takes model= and domain= only, got {sorted(options)}")
        for path in sorted(root.glob("*.jsonl")):
            if model is not None and path.stem != model:
                continue
            with path.open("r", encoding="utf-8") as fh:
                index = -1
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    index += 1
                    row = json.loads(line)
                    if domain is not None and row.get("task_name") != domain:
                        continue
                    key = f"{path.stem}/{row.get('task_name')}/{_task_key(row) or f'row{index}'}"
                    yield key, row, {"file": path.relative_to(root).as_posix(), "config": "default",
                                     "split": "train", "row": index}

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        meta = raw.get("meta") or {}
        evaluation = raw.get("eval_result") or {}
        messages: List[Dict[str, Any]] = raw.get("messages") or []
        model_path = raw.get("model_path")
        stem = Path(where["file"]).stem
        domain = raw.get("task_name")

        score = evaluation.get("score")
        numeric = isinstance(score, (int, float)) and not isinstance(score, bool)
        success = True if numeric and score == 1 else False if numeric and score == 0 else None

        labels: List[Dict[str, Any]] = []
        if "score" in evaluation:
            labels.append(_label(key, "score", score, "eval_result.score"))
        if "db_match" in evaluation:
            note = {"note": "no DB component in the telecom reward breakdown; use score"} if domain == "telecom" else {}
            labels.append(_label(key, "db_match", evaluation["db_match"], "eval_result.db_match", **note))
        if "finish_reason" in meta:
            labels.append(_label(key, "finish_reason", meta["finish_reason"], "meta.finish_reason"))
        breakdown = (meta.get("reward_details") or {}).get("reward_breakdown")
        for component, value in (breakdown.items() if isinstance(breakdown, dict) else ()):
            labels.append(_label(key, f"reward_breakdown.{component}", value,
                                 f"meta.reward_details.reward_breakdown.{component}"))

        source = self.source(key, where)
        source["row"] = where["row"]
        task_key = _task_key(raw)
        ext = {
            "version": EXT_VERSION,
            "source": source,
            "task": {"id": f"{domain}/{task_key}" if task_key else None, "text": None},
            "outcome": {"success": success, "score": float(score) if numeric else None, "raw": evaluation or None},
            "labels": labels,
            "agents": [
                {"id": "assistant", "name": stem, "instructions": None, "model": model_path, "tools": [], "handoffs": []},
                {"id": "user", "name": "user_simulator", "instructions": None,
                 "model": raw.get("user_model_path"), "tools": [], "handoffs": []},
            ],
            "tools": {},
            "dataset": {k: v for k, v in raw.items() if k != "messages"},
        }
        traj = messages_to_trajectory(
            messages, trajectory_id=key, agent_name=stem, model_name=model_path if isinstance(model_path, str) else None,
            ext=ext, actor=_actor, source_of=_source_of, pair_by_position=True)
        # the second message is the same string in every row of the pinned data: mark it, credit no model
        if len(traj["steps"]) > 1:
            step = traj["steps"][1]
            sx = (step.get("extra") or {}).get(EXT) or {}
            if sx.get("source_ref") == 1 and step["source"] == "agent" and step["message"] == GREETING \
                    and not step.get("tool_calls"):
                sx["scripted"] = True
        return Run(traj, TRAJECTORY)


dataset = Tau2Bench()
