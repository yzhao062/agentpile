"""The run record and its read-only view.

A trajectory record is an ATIF v1.7 trajectory. Everything ATIF has no field for
(source, labels, permissions, the actor of an interleaved step, dependencies)
sits under ``extra["agentpile"]`` at the trajectory, step and call level. A
config record (an MCP tool list, a CatchBench PRE row) has no execution steps,
so it is stored as its own ``agentpile-config/0`` JSON instead of a trajectory
with invented steps. ``Run`` gives both one read-only view.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

EXT = "agentpile"
EXT_VERSION = "0"
ATIF_VERSION = "ATIF-v1.7"
CONFIG_SCHEMA = "agentpile-config/0"

TRAJECTORY = "trajectory"
CONFIG = "config"


class _Missing:
    """The source had no value at all, as opposed to null or an empty string."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self):
        return (_Missing, ())


MISSING = _Missing()


@dataclass(frozen=True)
class Event:
    """One message, tool call or tool result, in source order.

    ``address`` is the full record address of the event, the same form that
    labels and dependency entries use: ``{"trajectory", "step"}`` plus
    ``"call"`` (a tool call id) or ``"result"`` (an index into the step's
    observation results). ``source_ref`` points back into the source record
    (a message index, a span id), when the reader recorded one.
    """

    kind: str
    role: str
    actor: Optional[str]
    content: Any
    address: Dict[str, Any]
    name: Optional[str] = None
    call_id: Optional[str] = None
    error: Any = None
    source_ref: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _ext(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not obj:
        return {}
    extra = obj.get("extra") or {}
    return extra.get(EXT) or {}


def _is_position(ref: Any) -> bool:
    """A source position the event order can use: an integer (a message or item index), never a bool."""
    return isinstance(ref, int) and not isinstance(ref, bool)


def _placed(pos: Any, ref: Any) -> Tuple[Optional[int], bool]:
    """(the integer an event sorts by or None, whether the order can use what the reader stored).

    A stored sort position (``source_pos`` of a step, an item of ``result_pos``) wins over the
    reference beside it; without one, an integer reference is the position. No reference at all
    leaves the event beside its step; a reference that is not an integer, with no sort position
    next to it, is one the order cannot use.
    """
    if _is_position(pos):
        return pos, True
    if _is_position(ref):
        return ref, True
    return None, pos is None and ref is None


class Run:
    """Read-only view over one trajectory record or one config record."""

    __slots__ = ("record", "kind")

    def __init__(self, record: Dict[str, Any], kind: Optional[str] = None):
        if kind is None:
            kind = CONFIG if record.get("schema") == CONFIG_SCHEMA else TRAJECTORY
        if kind not in (TRAJECTORY, CONFIG):
            raise ValueError(f"kind must be {TRAJECTORY!r} or {CONFIG!r}, got {kind!r}")
        self.record = record
        self.kind = kind

    # -- shared by both kinds ------------------------------------------------

    @property
    def ext(self) -> Dict[str, Any]:
        """The agentpile fields: ``extra["agentpile"]`` of a trajectory, or the config record itself."""
        return self.record if self.kind == CONFIG else _ext(self.record)

    @property
    def id(self) -> Optional[str]:
        if self.kind == CONFIG:
            return (self.ext.get("source") or {}).get("record_key")
        return self.record.get("trajectory_id") or self.record.get("session_id")

    @property
    def source(self) -> Dict[str, Any]:
        return self.ext.get("source") or {}

    @property
    def task(self) -> Dict[str, Any]:
        return self.ext.get("task") or {}

    @property
    def outcome(self) -> Dict[str, Any]:
        return self.ext.get("outcome") or {}

    @property
    def labels(self) -> List[Dict[str, Any]]:
        return list(self.ext.get("labels") or [])

    @property
    def agents(self) -> List[Dict[str, Any]]:
        return list(self.ext.get("agents") or [])

    @property
    def tools(self) -> Dict[str, Dict[str, Any]]:
        """Tool id -> tool entry. Ids are unique within a record, so two servers may expose the same name.

        A native ATIF trajectory, with no agentpile extension, gets an entry for
        each of its ``agent.tool_definitions``, built as the message reader builds them.
        """
        tools = dict(self.ext.get("tools") or {})
        if self.kind == TRAJECTORY and EXT not in (self.record.get("extra") or {}):
            from .formats.messages import _tool_entries

            _tool_entries((self.record.get("agent") or {}).get("tool_definitions"), tools)
        return tools

    @property
    def spans(self) -> Dict[str, Dict[str, Any]]:
        """Span id -> span the source recorded (OTLP or span-structured datasets), with the address it maps to."""
        return dict(self.ext.get("spans") or {})

    @property
    def dataset(self) -> Dict[str, Any]:
        """Dataset-specific raw fields, kept as the source gave them and not interpreted by the core."""
        return self.ext.get("dataset") or {}

    # -- trajectory only -----------------------------------------------------

    @property
    def steps(self) -> List[Dict[str, Any]]:
        if self.kind == CONFIG:
            return []
        return list(self.record.get("steps") or [])

    @property
    def subagents(self) -> List["Run"]:
        if self.kind == CONFIG:
            return []
        return [Run(t, TRAJECTORY) for t in self.record.get("subagent_trajectories") or []]

    def events(self) -> Iterator[Event]:
        """Messages, calls and results in recorded source order; empty for a config record.

        ATIF stores a result on the step that issued its call, which can sit
        before turns the source recorded in between. The events follow the
        source instead: a message, its calls and every later result come out at
        the source position the reader stored. A reader whose references are
        integers (a message index) needs nothing more: ``source_ref`` of the
        step and ``result_refs`` of the results are the positions. A reader
        whose references are not integers (a span address) stores an integer
        sort position beside each: ``source_pos`` on the step extension and
        ``result_pos``, parallel to ``result_refs``. A stored sort position wins
        over the reference. A result keeps its association with its call through
        ``call_id``, and every event keeps its address, so the order of this view
        never changes an address. Where a record has no usable positions, or an
        event has a reference that is not an integer and no sort position, the
        order is the order of the ATIF steps.

        Where ATIF forced a stand-in value (a non-null message or result, a
        dict of arguments), the event carries the source value: ``None`` for
        null, ``MISSING`` for absent, the raw value for arguments that were not
        a JSON object. A call whose source had no ``arguments`` member gives
        ``MISSING``, one whose arguments were null gives ``None``, and an
        explicit empty object gives ``{}``.
        A step whose source had no message at all yields no message event.
        """
        for _step, event in self._timeline():
            if event is not None:
                yield event

    def _timeline(self) -> List[Tuple[Dict[str, Any], Optional[Event]]]:
        """``(step, event)`` pairs in recorded source order, with ``(step, None)`` marking where each step starts.

        The step marker lets a rule see an agent step that has no message event
        (a step whose source had no message). Ordering is by the integer
        positions the reader stored (``source_pos``, ``result_pos`` or integer
        references); see ``events``.
        """
        if self.kind == CONFIG:
            return []
        tid = self.id
        items: List[Tuple[Any, Dict[str, Any], Optional[Event]]] = []
        positional = True
        position = -1  # the position of the latest step; a step with no stored position stays beside its predecessor
        for step in self.record.get("steps") or []:
            sx = _ext(step)
            sid = step.get("step_id")
            role = step.get("source")
            actor = sx.get("actor")
            ref = sx.get("source_ref")
            placed, usable = _placed(sx.get("source_pos"), ref)
            if placed is not None:
                position = placed
            if not usable:
                positional = False
            items.append((position, step, None))
            state = sx.get("message_state")
            if state != "missing":
                if state == "null":
                    content = None
                elif "message_raw" in sx:
                    content = sx["message_raw"]
                else:
                    content = step.get("message")
                items.append((position, step, Event("message", role, actor, content,
                                                    {"trajectory": tid, "step": sid}, source_ref=ref)))
            calls = sx.get("calls") or {}
            for call in step.get("tool_calls") or []:
                cid = call.get("tool_call_id")
                cx = calls.get(cid) or {}
                if cx.get("arguments_state") == "missing":
                    args: Any = MISSING
                elif "arguments_raw" in cx:
                    args = cx["arguments_raw"]
                else:
                    args = call.get("arguments")
                items.append((position, step, Event(
                    "call", role, actor, args, {"trajectory": tid, "step": sid, "call": cid},
                    name=call.get("function_name"), call_id=cid, error=cx.get("error"),
                    source_ref=cx.get("source_ref"), extra={"tool_id": cx.get("tool_id")})))
            refs = sx.get("result_refs") or []
            rposs = sx.get("result_pos") or []
            rfields = sx.get("result_fields") or []
            results = (step.get("observation") or {}).get("results") or []
            for i, res in enumerate(results):
                cid = res.get("source_call_id")
                cx = (calls.get(cid) or {}) if cid else {}
                rf = (rfields[i] if i < len(rfields) else None) or {}
                state = rf.get("content_state")
                content = None if state == "null" else MISSING if state == "missing" else rf.get("content_raw", res.get("content"))
                rref = refs[i] if i < len(refs) else None
                placed, usable = _placed(rposs[i] if i < len(rposs) else None, rref)
                rpos = position if placed is None else placed  # no stored position: beside the step's own events
                if not usable:
                    positional = False
                items.append((rpos, step, Event(
                    "result", "tool", actor, content, {"trajectory": tid, "step": sid, "result": i},
                    name=rf.get("name"), call_id=cid, error=cx.get("error", rf.get("error")), source_ref=rref)))
        if positional:
            items = sorted(items, key=lambda item: item[0])  # stable: ties keep the ATIF order
        return [(step, event) for _pos, step, event in items]

    def with_deps(self, rules: Any = ("tool_call_id_v1",)) -> "Run":
        """A copy of this run whose steps carry the dependencies ``derive_deps(self, rules)`` finds.

        Each entry goes under ``deps`` in the extension of its destination step,
        where ``to_otlp()`` writes it as a span link. Every entry keeps the rule
        and version that produced it, so an inferred dependency stays apart from
        what was loaded. Entries a step already holds are kept, and an equal
        entry is not added twice. This run is not changed; a config record comes
        back as an equal copy.
        """
        from .deps import derive_deps

        record = self.to_dict()
        if self.kind == CONFIG:
            return Run(record, CONFIG)
        steps: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        todo = [record]
        while todo:
            traj = todo.pop()
            tid = traj.get("trajectory_id") or traj.get("session_id")
            for step in traj.get("steps") or []:
                steps.setdefault((tid, step.get("step_id")), step)
            todo.extend(t for t in traj.get("subagent_trajectories") or [] if isinstance(t, dict))
        for dep in derive_deps(self, rules):
            dst = dep.get("dst") or {}
            step = steps.get((dst.get("trajectory"), dst.get("step")))
            if step is None:
                raise ValueError(f"dependency {dep!r} names a destination step this run does not have")
            held = step.setdefault("extra", {}).setdefault(EXT, {}).setdefault("deps", [])
            if dep not in held:
                held.append(copy.deepcopy(dep))
        return Run(record, TRAJECTORY)

    # -- output --------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.record)

    def to_json(self, dest: Union[str, Path, None] = None, indent: Optional[int] = None) -> Optional[str]:
        """Return the record as JSON, or write it to ``dest`` (ATIF for a trajectory, agentpile-config for a config)."""
        text = json.dumps(self.record, ensure_ascii=False, indent=indent)
        if dest is None:
            return text
        Path(dest).write_text(text + "\n", encoding="utf-8")
        return None

    def to_otlp(self, dest: Union[str, Path, None] = None, indent: Optional[int] = None) -> Optional[str]:
        """Return the trajectory as OTLP JSON, or write it to ``dest``; the ``otlp-json`` writer does the work."""
        from .registry import writers

        writer = writers.get("otlp-json")
        if dest is None:
            return writer.dumps(self, indent=indent)
        writer.write(self, dest, indent=indent)
        return None

    def __repr__(self) -> str:
        src = self.source
        where = src.get("dataset") or src.get("format") or "?"
        if self.kind == CONFIG:
            return f"<Run config {self.id!r} from {where} tools={len(self.tools)}>"
        return f"<Run trajectory {self.id!r} from {where} steps={len(self.record.get('steps') or [])}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Run) and self.kind == other.kind and self.record == other.record

    __hash__ = None  # records are mutable dicts
