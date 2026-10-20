"""Compatibility projection from agentpile runs to GRADE steps (projection version ``grade-steps-v1``).

The tested path is fixed::

    load -> derive_deps (named, general rules) -> project (this module) -> compare with an independent expected output

The projection may only filter, reorder and rename events and remap edge
endpoints. It never computes a dependency: every ``deps`` list is built from
``derive_deps`` entries whose ``src`` and ``dst`` addresses are looked up in a
table of the steps the projection emitted. An entry with an endpoint that has no
step (a call the target tool did not turn into a step) is dropped and counted.
GRADE step order, names and kinds are properties of its converters, so they are
reproduced here from record fields; agentpile's own types are not shaped by them.

Run identity: an expected line carries ``run_key`` and ``source`` ``{file, row,
ids}``. ``identity`` rebuilds the same three values from a loaded run, and
``grade_run_key`` formats them the way the expected files do, which is the join
between an expected run and an agentpile record key (``run.id``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import quote

import agentpile as ap
from agentpile import deps as D

PROJECTION_VERSION = "grade-steps-v1"

#: rules per corpus: the dependency semantics of the corpus's GRADE converter, as agentpile rules
CORPUS_RULES: Dict[str, List[Any]] = {
    "who_and_when": ["full_context_v1"],
    "tau_bench": ["prior_reads_v1"],
    "tau2_bench": ["value_flow_v1"],
    "swe_rebench_openhands": ["resource_chain_v1"],
    "swe_gym_openhands": ["resource_chain_v1"],
    "swe_agent_nebius": [D.resource_chain(D.shell_commands_v1)],
    "agent_reward_bench": [D.resource_chain(D.web_urls_v1)],
    # no ScienceWorld reader exists yet; the rule and extractor are ready for it
    "scienceworld": [D.last_writer(D.world_location_v1)],
}

_TOOL_AGENTS = {"Computer_terminal"}
_BLOCK_RE = re.compile(r"```(?:bash|python)?\s*(.*?)```", re.DOTALL)


def _ext(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (((obj or {}).get("extra") or {}).get("agentpile")) or {}


def _decision(agent: str) -> Dict[str, Any]:
    return {"agent": agent, "kind": "decision", "deps": []}


def _tool(agent: str = "env") -> Dict[str, Any]:
    return {"agent": agent, "kind": "tool_call", "deps": []}


Node = Tuple[Any, Any]  # (step id, call id or None)


@dataclass
class Projection:
    steps: List[Dict[str, Any]]
    dropped_edges: int = 0
    edges: int = 0


def _assistant_steps(run: ap.Run) -> Iterator[Dict[str, Any]]:
    """Steps that came from an assistant message (a user message promoted to an agent step is not one)."""
    for step in run.steps:
        if step.get("source") == "agent" and _ext(step).get("role", "assistant") == "assistant":
            yield step


# -- step lists (filter, reorder, rename) ----------------------------------------------------------

def _decision_then_calls(run: ap.Run) -> Tuple[List[Dict[str, Any]], Dict[Node, int]]:
    """OpenHands and tau2: one decision per assistant message, then one tool step per call, at the call."""
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    for step in _assistant_steps(run):
        steps.append(dict(_decision("agent"), idx=len(steps)))
        for call in step.get("tool_calls") or []:
            nodes[(step["step_id"], call["tool_call_id"])] = len(steps)
            steps.append(dict(_tool(), idx=len(steps)))
    return steps, nodes


def _tau_bench_steps(run: ap.Run) -> Tuple[List[Dict[str, Any]], Dict[Node, int]]:
    """tau-bench: a decision per user and assistant message, and a tool step at each tool result (message order)."""
    model = run.dataset.get("model_path", "model")
    items: List[Tuple[Tuple[int, int, int], Dict[str, Any], Optional[Node]]] = []
    for pos, step in enumerate(run.steps):
        sx = _ext(step)
        ref = sx["source_ref"] if isinstance(sx.get("source_ref"), int) else pos
        source = step.get("source")
        if source == "user" and sx.get("role") in ("tool", "function"):  # a tool output before any agent turn
            items.append(((ref, 0, 0), _tool(), None))
        elif source == "user":
            items.append(((ref, 0, 0), _decision("user"), None))
        elif source == "agent":
            items.append(((ref, 0, 0), _decision(model), None))
            refs = sx.get("result_refs") or []
            for i, res in enumerate((step.get("observation") or {}).get("results") or []):
                at = refs[i] if i < len(refs) and isinstance(refs[i], int) else ref
                cid = res.get("source_call_id")
                items.append(((at, 1, i), _tool(), (step["step_id"], cid) if cid is not None else None))
    items.sort(key=lambda item: item[0])
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    for _key, proto, node in items:
        if node is not None:
            nodes[node] = len(steps)
        steps.append(dict(proto, idx=len(steps)))
    return steps, nodes


def _who_and_when_steps(run: ap.Run) -> Tuple[List[Dict[str, Any]], Dict[Node, int]]:
    """Who&When: one step per history entry; the speaker is the raw name, else the raw role."""
    named = run.source.get("config") == "Algorithm-Generated"  # only these entries carry a name
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    for pos, step in enumerate(run.steps):
        sx = _ext(step)
        name = sx.get("actor") if named else None
        agent = name or sx.get("role") or "unknown"
        text = str(step.get("message", "")).lower()
        tool = name in _TOOL_AGENTS or text.startswith("exitcode") or text.startswith("code output")
        steps.append({"idx": pos, "agent": agent, "kind": "tool_call" if tool else "decision", "deps": []})
        nodes[(step["step_id"], None)] = pos
    return steps, nodes


def _action_turn_steps(run: ap.Run, has_action) -> Tuple[List[Dict[str, Any]], Dict[Node, int]]:
    """Text-action corpora: a decision per agent turn, and a tool step when the turn holds an action."""
    steps: List[Dict[str, Any]] = []
    nodes: Dict[Node, int] = {}
    for step in run.steps:
        if step.get("source") != "agent":
            continue
        text = step.get("message") if isinstance(step.get("message"), str) else ""
        wants = has_action(step, text)
        if wants is None:
            continue  # the turn emits nothing
        steps.append(dict(_decision("agent"), idx=len(steps)))
        if wants:
            nodes[(step["step_id"], None)] = len(steps)
            for call in step.get("tool_calls") or []:
                nodes[(step["step_id"], call["tool_call_id"])] = len(steps)
            steps.append(dict(_tool(), idx=len(steps)))
    return steps, nodes


def _scienceworld_steps(run: ap.Run):
    return _action_turn_steps(run, lambda step, text: True if "action:" in text.lower() else None)


def _swe_agent_steps(run: ap.Run):
    def has_action(step: Dict[str, Any], text: str):
        blocks = _BLOCK_RE.findall(text)
        return bool(blocks and blocks[-1].strip() and blocks[-1].strip().splitlines()[0].strip())

    return _action_turn_steps(run, has_action)


def _agent_reward_bench_steps(run: ap.Run):
    return _action_turn_steps(run, lambda step, text: True)


STEPS = {
    "who_and_when": _who_and_when_steps,
    "tau_bench": _tau_bench_steps,
    "tau2_bench": _decision_then_calls,
    "swe_rebench_openhands": _decision_then_calls,
    "swe_gym_openhands": _decision_then_calls,
    "swe_agent_nebius": _swe_agent_steps,
    "agent_reward_bench": _agent_reward_bench_steps,
    "scienceworld": _scienceworld_steps,
}


# -- projection ------------------------------------------------------------------------------------

def _node(address: Dict[str, Any]) -> Node:
    return (address.get("step"), address.get("call"))


def project(corpus: str, run: ap.Run, deps: Sequence[Dict[str, Any]]) -> Projection:
    """GRADE steps of ``run``, with ``deps`` taken from the ``derive_deps`` entries and nowhere else."""
    steps, nodes = STEPS[corpus](run)
    out = Projection(steps)
    for entry in deps:
        src, dst = nodes.get(_node(entry["src"])), nodes.get(_node(entry["dst"]))
        if src is None or dst is None:
            out.dropped_edges += 1
            continue
        steps[dst]["deps"].append(src)
        out.edges += 1
    for step in steps:
        step["deps"].sort()
    return out


def project_run(corpus: str, run: ap.Run) -> Projection:
    """The whole tested path after ``load``: ``derive_deps`` with the corpus rules, then the projection."""
    return project(corpus, run, ap.derive_deps(run, rules=CORPUS_RULES[corpus]))


# -- identity and join -----------------------------------------------------------------------------

def identity(corpus: str, run: ap.Run) -> Tuple[str, int, List[Any]]:
    """(file, row, ids) of a loaded run, in the form of an expected line's ``source``."""
    src = run.source
    meta = run.dataset.get("meta") or {}
    if corpus == "who_and_when":
        return src["file"], 0, [run.task.get("id")]
    if corpus in ("tau_bench", "tau2_bench"):
        return src["file"], src["row"], [run.dataset.get("task_name"), meta.get("id")]
    if corpus == "swe_rebench_openhands":
        return src["file"], src["row_index"], [run.id, run.task.get("id")]
    if corpus == "swe_gym_openhands":
        return src["file"], src["row_in_file"], [src.get("instance_id"), src.get("run_id")]
    if corpus == "swe_agent_nebius":  # record key is "<shard stem>:<row in shard>"
        return src["file"], int(src["record_key"].rsplit(":", 1)[1]), [run.dataset.get("instance_id"), run.dataset.get("model_name")]
    if corpus == "agent_reward_bench":  # record key is "<benchmark>/<agent>/<task id>"; one run per file
        return src["file"], 0, [run.dataset.get("benchmark"), run.id.split("/")[1], run.task.get("id")]
    raise KeyError(f"no identity rule for {corpus!r}: its reader does not exist yet")


def grade_run_key(corpus: str, file: str, row: int, ids: Iterable[Any]) -> str:
    """``corpus:quote(file):row:quote(id)...`` as the expected files write it."""
    return ":".join([corpus, quote(str(file), safe=""), str(row)] + [quote(str(i), safe="") for i in ids])


def run_key(corpus: str, run: ap.Run) -> str:
    return grade_run_key(corpus, *identity(corpus, run))


def label_failed(corpus: str, run: ap.Run) -> Optional[int]:
    """The failure label as GRADE's loaders read it, from the record's outcome; None when GRADE counts none."""
    raw = run.outcome.get("raw") or {}
    if corpus == "swe_rebench_openhands":
        return 1 - int(raw["resolved"])
    if corpus == "swe_gym_openhands":
        return 0 if raw.get("resolved") else 1
    if corpus == "tau_bench":
        return 0 if (raw.get("eval_result") or {}).get("db_match") else 1
    if corpus == "tau2_bench":
        return 1 if float((raw or {})["score"]) == 0.0 else 0
    if corpus == "who_and_when":
        return (1 if str(raw["is_correct"]).lower() == "false" else 0) if "is_correct" in raw else None
    if corpus == "swe_agent_nebius":  # GRADE: target is resolved
        return 0 if run.outcome.get("success") else 1
    if corpus == "agent_reward_bench":  # GRADE: the first annotator's success starting with "unsuccess" is a failure
        return 1 if run.outcome.get("success") is False else 0
    raise KeyError(corpus)


# -- loading the runs an expected file names ---------------------------------------------------------

def load_expected(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_runs(corpus: str, root: Optional[str], expected: Sequence[Dict[str, Any]]) -> Iterator[ap.Run]:
    """Stream the loaded runs that can hold the expected ones; ``root`` is a local copy of the pinned files."""
    files = sorted({e["source"]["file"] for e in expected})
    if corpus == "who_and_when":
        return ap.load("who_and_when", root=root)
    if corpus == "tau_bench":
        return ap.load("tau_bench", root=root, model=[Path(f).stem for f in files])
    if corpus == "tau2_bench":
        if len(files) != 1:
            raise ValueError(f"the tau2 expectation is one model file, got {files}")
        return ap.load("tau2_bench", root=root, model=Path(files[0]).stem)
    if corpus == "swe_rebench_openhands":
        return ap.load("swe_rebench_openhands", root=root, limit=1 + max(e["source"]["row"] for e in expected))
    if corpus == "swe_gym_openhands":
        return ap.load("swe_gym_openhands", root=root)
    if corpus == "swe_agent_nebius":
        if files != ["data/train-00000-of-00012.parquet"]:
            raise ValueError(f"the SWE-agent expectation reads the first shard only, got {files}")
        return ap.load("swe_agent_nebius", root=root, limit=1 + max(e["source"]["row"] for e in expected))
    if corpus == "agent_reward_bench":
        benchmarks = sorted({f.split("/")[1] for f in files})
        return ap.load("agent_reward_bench", root=root, benchmark=benchmarks)
    raise KeyError(f"no loader for {corpus!r}: its reader does not exist yet")


# -- comparison ---------------------------------------------------------------------------------------

@dataclass
class Report:
    corpus: str
    runs_expected: int = 0
    runs_compared: int = 0
    steps_compared: int = 0
    edges_expected: int = 0
    edges_projected: int = 0
    dropped_edges: int = 0
    missing_runs: List[str] = field(default_factory=list)
    join_errors: List[str] = field(default_factory=list)
    differences: List[Dict[str, Any]] = field(default_factory=list)
    listed: List[Dict[str, Any]] = field(default_factory=list)
    unlisted: List[Dict[str, Any]] = field(default_factory=list)
    stale: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.corpus}: runs {self.runs_compared}/{self.runs_expected}, steps {self.steps_compared}, "
                f"edges expected {self.edges_expected} projected {self.edges_projected} dropped {self.dropped_edges}, "
                f"differences {len(self.differences)} (listed {len(self.listed)}, unlisted {len(self.unlisted)}), "
                f"stale exceptions {len(self.stale)}, missing runs {len(self.missing_runs)}, "
                f"join errors {len(self.join_errors)}")

    @property
    def clean(self) -> bool:
        return not (self.missing_runs or self.join_errors or self.unlisted or self.stale)


def _shape(step: Dict[str, Any]) -> Dict[str, Any]:
    return {k: step[k] for k in ("idx", "agent", "kind", "deps")}


def diff_steps(corpus: str, key: str, expected: Sequence[Dict[str, Any]], got: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Step-by-step differences in ``agent``, ``kind`` and ``deps``, plus a step-count difference.

    When the lists differ in length, the steps past the shorter list are reported once as a ``tail``
    difference at the index of the first of them, with the steps (``idx``, ``agent``, ``kind``, ``deps``)
    on each side, so that the edges of a step only one side has are never uncounted.
    """
    out: List[Dict[str, Any]] = []

    def add(step: Optional[int], fld: str, exp: Any, have: Any) -> None:
        out.append({"corpus": corpus, "run_key": key, "step": step, "field": fld, "expected": exp, "got": have})

    if len(expected) != len(got):
        add(None, "n_steps", len(expected), len(got))
        n = min(len(expected), len(got))
        add(n, "tail", [_shape(s) for s in expected[n:]], [_shape(s) for s in got[n:]])
    for i, (e, g) in enumerate(zip(expected, got)):
        if e["idx"] != g["idx"]:
            add(i, "idx", e["idx"], g["idx"])
        for fld in ("agent", "kind", "deps"):
            if e[fld] != g[fld]:
                add(e["idx"], fld, e[fld], g[fld])
    return out


def load_exceptions(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)["exceptions"]


def _same(entry: Dict[str, Any], diff: Dict[str, Any]) -> bool:
    return all(entry.get(k) == diff[k] for k in ("corpus", "run_key", "step", "field", "expected", "got"))


def compare_corpus(corpus: str, runs: Iterable[ap.Run], expected: Sequence[Dict[str, Any]],
                   exceptions: Sequence[Dict[str, Any]] = (), label: Optional[str] = None) -> Report:
    """Compare every expected run with the projection of the loaded run it names.

    Each difference must be in ``exceptions`` (matching run, step, field, expected and got values);
    an exception that matches no difference is stale. Loaded runs that no expected line names are ignored.
    ``label`` names the comparison in reports and exceptions when one corpus is compared with
    more than one expected file (default: the corpus).
    """
    name = label or corpus
    report = Report(name, runs_expected=len(expected))
    by_key = {e["run_key"]: e for e in expected}
    seen = set()
    for run in runs:
        key = run_key(corpus, run)
        want = by_key.get(key)
        if want is None:
            continue
        seen.add(key)
        file, row, ids = identity(corpus, run)
        src = want["source"]
        if (src["file"], src["row"], list(src["ids"].values())) != (file, row, ids):
            report.join_errors.append(f"{key}: source {src['file']!r} {src['row']} {list(src['ids'].values())} != "
                                      f"{file!r} {row} {ids}")
        if src.get("label_failed") != label_failed(corpus, run):
            report.join_errors.append(f"{key}: label_failed {src.get('label_failed')!r} != {label_failed(corpus, run)!r}")
        proj = project_run(corpus, run)
        report.runs_compared += 1
        report.steps_compared += len(want["steps"])
        report.edges_expected += sum(len(s["deps"]) for s in want["steps"])
        report.edges_projected += proj.edges
        report.dropped_edges += proj.dropped_edges
        report.differences.extend(diff_steps(name, key, want["steps"], proj.steps))
    report.missing_runs = sorted(set(by_key) - seen)
    used = set()
    for diff in report.differences:
        hit = next((i for i, e in enumerate(exceptions) if _same(e, diff)), None)
        if hit is None:
            report.unlisted.append(diff)
        else:
            used.add(hit)
            report.listed.append(dict(diff, exception=exceptions[hit].get("id")))
    report.stale = [e for i, e in enumerate(exceptions) if i not in used and e.get("corpus") == name]
    return report
