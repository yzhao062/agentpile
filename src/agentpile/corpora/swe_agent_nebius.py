"""SWE-agent trajectories (nebius/SWE-agent-trajectories): 80,036 SWE-agent attempts on SWE-bench-style issues.

One run per Parquet row, read from the 12 ``data/train-*.parquet`` shards in
batches. A row holds one attempt by one model (``model_name``) on one issue
(``instance_id``) as a list of turns with the roles ``system``, ``user`` and
``ai``. The source has no tool-call field: an action is a fenced code block
inside the text of an ``ai`` turn, so no call is produced, none is guessed
from the text, and no id is invented. The reader's rules, all by position
because no field marks them:

- The ``system`` turn is a system step; its message is ``system_prompt`` (the
  command documentation travels in it as text, there is no tool schema).
- The first ``user`` turn is the issue prompt and becomes a user step. It is
  also the task text, kept whole (the issue sits inside SWE-agent's template).
- Each ``ai`` turn is an agent step whose message is the whole text. The
  original role string ``ai`` is kept as the step ``role``.
- Every later ``user`` turn is the environment's output in the SWE-agent
  format, not a user message. It becomes an observation result with
  ``source_call_id`` null on the preceding agent step, and its original role
  ``user`` is kept in ``result_fields``. A later ``user`` turn that has no
  earlier agent step to attach to stays a user step. The pinned data has none,
  so this branch exists only to keep the rule total.
- ``mask`` (true on every ``ai`` turn, false on every other turn; the card
  does not define it) and the one-valued ``cutoff_date`` of the system turn
  are kept as given, in ``source_fields`` of a step or ``result_fields`` of a
  result.

``target`` is the outcome (the card: the linked pull request's tests pass on
``generated_patch``); ``exit_status`` is a run-level label, not an outcome.
The row has no run id and ``(instance_id, model_name)`` repeats, so the run id
is ``<shard file stem>:<row index in that shard>``. The model is in the ATIF
agent block and the agent name is SWE-agent, from the card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from ..formats.messages import messages_to_trajectory
from ..record import EXT, TRAJECTORY, Run
from ._base import Dataset

BATCH_ROWS = 64  # rows hold up to ~800 turns and single turns of 370k characters
AGENT_NAME = "SWE-agent"
ROLES = ("system", "user", "ai")
_TURN_ROLE = "turn_role"  # carries the original role of an observation through the shared parser


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("reading this dataset needs pyarrow: pip install 'agentpile[hf]'") from exc
    return pq


def _message(turn: Dict[str, Any], role: str) -> Dict[str, Any]:
    """The shared parser's message for one turn; the turn's other fields ride along as extra keys."""
    own, other = ("system_prompt", "text") if turn.get("role") == "system" else ("text", "system_prompt")
    msg: Dict[str, Any] = {"role": role, "content": turn.get(own)}
    for key, value in turn.items():
        if key in ("role", "text", "system_prompt") or (key == "cutoff_date" and value is None):
            continue
        msg[key] = value
    if turn.get(other) is not None:  # never in the pinned data; kept if a turn fills both text fields
        msg[other] = turn[other]
    return msg


class SweAgentNebius(Dataset):
    name = "swe_agent_nebius"
    format = "swe_agent_nebius-parquet"
    incremental = True

    def records(self, root: Path, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"swe_agent_nebius takes no options, got {sorted(options)}")
        pq = _require_pyarrow()
        for path in self.paths(root, "data/train-*.parquet"):
            rel = path.relative_to(root).as_posix()
            where = {"file": rel, "config": "default", "split": "train"}
            row = 0
            for batch in pq.ParquetFile(self.ready(path)).iter_batches(batch_size=BATCH_ROWS):
                for raw in batch.to_pylist():
                    yield f"{path.stem}:{row}", raw, where
                    row += 1

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        messages: List[Dict[str, Any]] = []
        task_text = None
        seen_user = seen_agent = False
        for turn in raw.get("trajectory") or []:
            role = turn.get("role")
            if role not in ROLES:
                raise ValueError(f"{key}: unexpected turn role {role!r}, expected one of {list(ROLES)}")
            if role == "user" and seen_user and seen_agent:
                msg = _message(turn, "tool")  # the environment's output: a result, not a user message
                msg[_TURN_ROLE] = role
            else:
                msg = _message(turn, role)
                if role == "user" and not seen_user:
                    task_text = turn.get("text")
            seen_user = seen_user or role == "user"
            seen_agent = seen_agent or role == "ai"
            messages.append(msg)

        target = raw.get("target")
        labels = []
        if raw.get("exit_status") is not None:
            labels.append({"name": "exit_status", "target": {"kind": "run", "address": {"trajectory": key}},
                           "value": raw["exit_status"], "label_source": "swe_agent_nebius",
                           "source_ref": {"field": "exit_status"}})
        ext = {
            "source": self.source(key, where),
            "task": {"id": raw.get("instance_id"), "text": task_text},
            "outcome": {"success": target if isinstance(target, bool) else None, "score": None,
                        "raw": {"target": target} if "target" in raw else None},
            "labels": labels,
            "agents": [],
            "tools": {},
            "dataset": {k: v for k, v in raw.items() if k != "trajectory"},
        }
        traj = messages_to_trajectory(
            messages, trajectory_id=key, agent_name=AGENT_NAME, model_name=raw.get("model_name"), ext=ext,
            source_of=lambda m: "agent" if m["role"] == "ai" else None)
        for step in traj["steps"]:  # give each observation its original role under the key the record schema reads
            for fields in ((step.get("extra") or {}).get(EXT) or {}).get("result_fields") or []:
                if fields and _TURN_ROLE in fields:
                    fields["role"] = fields.pop(_TURN_ROLE)
        return Run(traj, TRAJECTORY)


dataset = SweAgentNebius()
