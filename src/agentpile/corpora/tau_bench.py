"""tau-bench (AgentSuite/tau-bench-trajectories): 30 agent models on 165 airline and retail tasks.

One run per JSON line of ``<model>.jsonl`` (30 files x 165 rows, 4950 runs). A
row is one agent model, played against a simulated user, on one tau-bench task.
Rows are read in file order, files by name; the file stem is the model, because
``model_path`` is wrong in ``gpt-5-nano.jsonl`` (it says ``openai/gpt-5``). The
run key is ``<stem>/<task_name>/<id>`` where ``id`` is ``meta.id`` without its
``<task_name>_`` prefix (the three gemini files store a bare number). Messages go
through the shared OpenAI-style parser; nothing is inferred:

- ``system`` is the task policy and becomes a system step, ``user`` a user step
  (the simulated user), ``assistant`` an agent step whose actor is the model.
  Each assistant message holds at most one call, and the ``tool`` message
  answering it joins that same agent step, paired by ``tool_call_id`` only.
- The three gemini files have no ``tool`` messages: each call's result is the
  next message, labelled ``user``, without an id. Those results stay user steps
  as labelled and their calls stay unanswered; nothing is paired by adjacency.
- The three claude ``*-thinking-on-10k`` files have a user message, "The tool
  has finished running. Please continue your reasoning.", after every tool
  result. It is kept as the user step the source calls it.
- A row with no messages (94 of them) is still a run; the shared parser marks
  it ``empty_source``. Its labels are the source's placeholder values (score
  0.0, ``db_match`` false, empty task), and each label's ``source_ref`` says
  ``empty_run``.
- ``turn_idx`` and the call's ``index`` stay in the extension. There are no
  tool definitions, timestamps or reasoning fields in the source.
- ``eval_result.db_match`` is the outcome (``success``) and ``eval_result.score``
  is the score; they differ on 651 rows (score 1.0, ``db_match`` false, where
  ``db_reward`` is null). ``meta.is_correct`` equals ``db_match`` in every row.
  Both fields, the database reward and the finish reason are run labels, and
  the whole ``eval_result`` plus ``is_correct`` sit in ``outcome.raw``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from ..formats.messages import messages_to_trajectory
from ..record import TRAJECTORY, Run
from ._base import Dataset

LABEL_SOURCE = "tau_bench"


def _task_number(task_name: Any, meta_id: Any) -> Optional[str]:
    """``meta.id`` without a leading ``<task_name>_``; the gemini files store the bare number."""
    if meta_id is None:
        return None
    text = str(meta_id)
    prefix = f"{task_name}_"
    return text[len(prefix):] if text.startswith(prefix) else text


def _names(value: Union[None, str, Iterable[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    return [value] if isinstance(value, str) else list(value)


class TauBench(Dataset):
    name = "tau_bench"
    format = "tau_bench-jsonl"
    incremental = True

    def records(self, root: Path, model: Union[None, str, Iterable[str]] = None,
                task_name: Optional[str] = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"tau_bench takes model= and task_name= only, got {sorted(options)}")
        wanted = _names(model)
        paths = self.paths(root, "*.jsonl")
        if wanted is not None:
            missing = sorted(set(wanted) - {p.stem for p in paths})
            if missing:
                raise ValueError(f"unknown model file stem {missing}; the files are {[p.stem for p in paths]}")
        for path in paths:
            if wanted is not None and path.stem not in wanted:
                continue
            with self.ready(path).open("r", encoding="utf-8") as fh:
                for row, line in enumerate(fh):
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if task_name is not None and raw.get("task_name") != task_name:
                        continue
                    domain = raw.get("task_name")
                    number = _task_number(domain, (raw.get("meta") or {}).get("id"))
                    key = f"{path.stem}/{domain}/{number if number is not None else 'row' + str(row)}"
                    yield key, raw, {"file": path.name, "config": "default", "split": "train",
                                     "model": path.stem, "row": row}

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        stem = where["model"]
        meta = raw.get("meta") or {}
        ev = raw.get("eval_result") or {}
        description = meta.get("task_description") or {}
        messages = raw.get("messages")
        messages = messages if isinstance(messages, list) else []
        empty_run = not messages

        domain = raw.get("task_name")
        number = _task_number(domain, meta.get("id"))
        db_match, score = ev.get("db_match"), ev.get("score")
        run_target = {"kind": "run", "address": {"trajectory": key}}

        def label(name: str, value: Any, field: str) -> Dict[str, Any]:
            ref: Dict[str, Any] = {"field": field}
            if empty_run:
                ref["empty_run"] = True  # a placeholder row: the value is the source's, not a measured result
            return {"name": name, "target": run_target, "value": value, "label_source": LABEL_SOURCE,
                    "source_ref": ref}

        reward = (meta.get("reward_details") or {}).get("db_reward")
        labels = [label("db_match", db_match, "eval_result.db_match"),
                  label("score", score, "eval_result.score"),
                  label("db_reward", reward, "meta.reward_details.db_reward"),
                  label("finish_reason", meta.get("finish_reason"), "meta.finish_reason"),
                  label("gold_actions", description.get("actions"), "meta.task_description.actions")]

        source = self.source(key, where)
        source["row"] = where["row"]  # 0-based line number in the file
        ext = {
            "source": source,
            "task": {"id": f"{domain}/{number}" if number is not None else None,
                     "text": description.get("instruction")},
            "outcome": {"success": db_match if isinstance(db_match, bool) else None,
                        "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                        "raw": {"eval_result": dict(ev), "meta": {"is_correct": meta.get("is_correct")}}},
            "labels": labels,
            "agents": [{"id": stem, "name": stem, "instructions": None, "model": stem, "tools": [], "handoffs": []}],
            "tools": {},
            "dataset": {k: v for k, v in raw.items() if k != "messages"},
        }
        traj = messages_to_trajectory(messages, trajectory_id=key, model_name=stem, ext=ext,
                                      actor=lambda _msg: stem)
        return Run(traj, TRAJECTORY)


dataset = TauBench()
