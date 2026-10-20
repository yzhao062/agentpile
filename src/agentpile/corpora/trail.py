"""TRAIL (PatronusAI/TRAIL): span-structured smolagents traces with span-level error annotations.

One run per trace: 117 GAIA traces (a CodeAgent manager that may delegate to a
ToolCallingAgent) and 31 SWE-Bench Lite traces (a single CodeAgent), each an
OpenTelemetry span tree with OpenInference attributes.

Source choice. The pinned repository holds the same data three ways: Parquet
rows ``data/<split>-*.parquet`` (two string columns, ``trace`` and ``labels``,
the JSON text of one trace file and of its annotation file), per-trace
``GAIA/*.json`` and ``SWE Bench/*.json`` files, and ``processed_annotations_*``
files. Compared byte for byte, all 148 ``trace`` cells equal the per-trace
files and all 148 ``labels`` cells equal the annotation files (and both equal
the authors' GitHub copy), so no field is lost either way. The loader reads the
Parquet: it is the dataset card's own configuration, pairs every trace with its
labels inside one row (the per-trace route pairs by file name, and 5 GAIA label
files carry no ``trace_id``), and downloads 55 MB in two files instead of 186 MB
in 296. It needs ``pyarrow`` (the ``hf`` extra). The repository is gated: accept
its terms on the hub and log in before the first download.

Spans. Every span of the trace, nested ``child_spans`` included, goes into the
extension's ``spans`` map under its ``span_id``: OpenInference kind, parent,
``trace_id``, start and end in Unix nanoseconds (the end is the start plus the
recorded ISO 8601 ``duration``), status, the flat ``span_attributes`` dict,
events and links. Source span fields with no schema slot (the raw ``timestamp``
and ``duration``, OTel ``span_kind``, service, scope, resource attributes,
``logs``) stay under ``source_fields``. Event times carry no zone in the source
and are read as UTC (all 437 fall inside their span). A span that became a step
has ``address`` ``{"trajectory", "step"}``. CHAIN, TOOL, AGENT and kind-less
spans are not steps and keep ``address`` null: a TOOL span carries no call id,
so it is not tied to a call (its Step CHAIN parent also holds the LLM span).
A repeated span id is kept once (the repeat is recorded in ``dataset``), and a
parent id absent from the file is kept as given. The ``logs`` of the harness
spans hold the task id (GAIA ``task_id``, SWE-Bench ``instance_id``), which
becomes ``task.id``, and also the GAIA reference answer and the SWE-Bench gold
patch and tests, so strip them before showing spans to an agent under test.

Steps. The span tree is walked depth first in start order. An agent run is the
AGENT span above an LLM span, with only CHAIN spans between them. An LLM span
below any other span forms a run of its own, keyed by that span: the
final-answer call under ``answer_single_question``, and the 8 model calls made
inside a ``TextInspectorTool`` span (their context is the tool's, so they stay
out of the agent's history). A run without an AGENT span has no ``actor``; so
does a trace whose AGENT span is missing from the file (its LLM spans share one
run keyed by the missing parent id). Every LLM span re-sends its whole history,
so a message is read only the first time its text appears in its run (a
repeated identical text counts once per occurrence; the role label is not part
of the identity, because a trimmed window drops the role of its first
message). The final-answer call re-renders the memory of the runs before it
with every role set to user, so the messages of a run that has no AGENT span
are compared with all messages read so far in the trace. A first-seen message
becomes:

- ``system`` or ``user``: a system or user step.
- ``assistant`` (or role-less): smolagents renders each earlier model output
  back into the history, wrapped (``<end_code>``, ``[FACTS LIST]:``,
  ``[PLAN]:``). When the text of a not yet claimed earlier output of the same
  run occurs in it (whitespace collapsed, a trailing ``<end_plan>`` ignored),
  the message is that output's copy: it makes no new step and is noted on the
  output's step as ``history_copy``. Any other assistant message becomes an
  agent step marked ``history_only``, a role-less one a user step with
  ``role_state`` ``missing``. On the pinned data every assistant message finds
  its output.
- ``tool-call``: its text is ``Calling tools:`` plus a Python-repr list of
  recorded call objects (``id``, ``function.name``, ``function.arguments``),
  read with ``ast.literal_eval`` (no code is parsed). A call whose id the run
  already holds (the structured call of the output that produced it) only gets
  a ``history_refs`` note on that call. Any other becomes a tool call on the
  latest agent step of the run, marked ``call_from`` ``history_message`` and
  ``paired_by`` ``latest_agent_step``. That is the one pairing by position
  here: the history places the call right after the rendering of the output
  that produced it, and the call's code occurs in that step's message (561 of
  564 CodeAgent calls; the other 3 join several code blocks of the message).
- ``tool-response``: an observation result on the step holding the call named by
  its ``Call id: <id>`` prefix. A call can have two responses (its logs, then
  an error). A response without a known id is kept unpaired on the latest
  agent step of the run, as in the messages reader.
- anything else: a user step that keeps the role.

An LLM span's output becomes one agent step (timestamp, model, token counts as
ATIF metrics). Structured calls (``llm.output_messages.N.message.tool_calls``,
GAIA ToolCallingAgent only) become tool calls with their ids. Code written in a
message stays message text; its call exists only through the next ``tool-call``
history message. A failed LLM span with no output message makes no step. Each
step and call extension carries ``span_id`` and a ``source_ref`` with the
message index, to go back to the span. ``actor`` and ``agent`` are the AGENT
span id of the owning run, listed in ``agents``.

Labels. Each entry of ``errors`` is a label named ``error`` on the span named by
its ``location``, with the entry as the value (category, evidence, description,
impact and any other key, raw, with no category normalization). A location that
names no span of the trace (the sentinel ``Span ID not found for this shard``,
twice) stays a run-level label with ``resolved`` false in ``source_ref``, never
a span target that does not exist. Each key of ``scores[0]`` (score and
reasoning per dimension, ``overall``) is a run-level label. One annotation file
is not strict JSON (a trailing comma) and is read with a comma-tolerant parser,
recorded as ``labels_parse`` ``tolerant`` in ``dataset``. The source gives no
task outcome, so ``outcome`` stays empty.
"""

from __future__ import annotations

import ast
import calendar
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run
from ._base import Dataset

SPLITS = ("gaia", "swe_bench")
INPUT = re.compile(r"llm\.input_messages\.(\d+)\.message\.(.+)")
OUTPUT = re.compile(r"llm\.output_messages\.(\d+)\.message\.(.+)")
OUTPUT_CALL = re.compile(r"tool_calls\.(\d+)\.tool_call\.(.+)")
TOOL_DEF = re.compile(r"llm\.tools\.(\d+)\.tool\.(json_schema)")
KIND = "openinference.span.kind"
CALLING = "Calling tools:\n"
CALL_ID = re.compile(r"Call id: (\S+)\n")
TIMESTAMP = re.compile(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)?")
DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)(?:\.(\d+))?S)?")
SPAN_KEYS = {"span_id", "span_name", "parent_span_id", "trace_id", "status_code", "status_message", "span_attributes",
             "events", "links", "child_spans"}
_ABSENT = object()


def _iso_ns(text: Any) -> Optional[int]:
    """Unix nanoseconds of an ISO 8601 timestamp; no zone is read as UTC."""
    m = TIMESTAMP.fullmatch(text) if isinstance(text, str) else None
    if not m:
        return None
    y, mo, d, h, mi, s, frac, zone = m.groups()
    seconds = calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s)))
    if zone and zone != "Z":
        offset = (int(zone[1:3]) * 60 + int(zone[4:6])) * 60
        seconds -= offset if zone[0] == "+" else -offset
    return seconds * 10**9 + int((frac or "0")[:9].ljust(9, "0"))


def _duration_ns(text: Any) -> Optional[int]:
    m = DURATION.fullmatch(text) if isinstance(text, str) else None
    if not m or not any(m.groups()):
        return None
    h, mi, s, frac = m.groups()
    return ((int(h or 0) * 60 + int(mi or 0)) * 60 + int(s or 0)) * 10**9 + int((frac or "0")[:9].ljust(9, "0"))


def _loads_tolerant(text: str) -> Tuple[Any, str]:
    """Parse JSON; on failure drop commas that precede a closing bracket (outside strings) and parse again."""
    try:
        return json.loads(text), "strict"
    except ValueError:
        pass
    out: List[str] = []
    in_string = escaped = False
    n = len(text)
    for i, ch in enumerate(text):
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                continue
            out.append(ch)
        else:
            out.append(ch)
    return json.loads("".join(out)), "tolerant"


def _indexed(attrs: Dict[str, Any], pattern: "re.Pattern[str]") -> List[Tuple[int, Dict[str, Any]]]:
    """Flattened ``prefix.N.message.<field>`` attributes as ``(N, {field: value})`` in index order."""
    found: Dict[int, Dict[str, Any]] = {}
    for k, v in attrs.items():
        m = pattern.fullmatch(k)
        if m:
            found.setdefault(int(m.group(1)), {})[m.group(2)] = v
    return [(i, found[i]) for i in sorted(found)]


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _output_key(text: str) -> str:
    """The text of a model output as it can occur inside the history rendering of that output."""
    text = _collapse(text)
    return text[: -len("<end_plan>")].rstrip() if text.endswith("<end_plan>") else text


def _parse_arguments(raw: Any):
    """ATIF arguments must be an object: (arguments, raw value or _ABSENT when the raw value was that object)."""
    if isinstance(raw, dict):
        return raw, _ABSENT
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}, raw
        if isinstance(parsed, dict):
            return parsed, _ABSENT
    return {}, raw


class _Group:
    """State of one agent run while its LLM spans are read in time order."""

    __slots__ = ("agent", "seen", "last", "outputs", "calls")

    def __init__(self, agent: Optional[str]):
        self.agent = agent
        self.seen: Counter = Counter()  # message text -> occurrences already read
        self.last: Optional[int] = None  # index in steps of the latest agent step
        self.outputs: List[Dict[str, Any]] = []  # model outputs that no history message has claimed yet
        self.calls: Dict[str, int] = {}  # call id -> index in steps of the step holding the call


class _Conversion:
    """One trace -> ATIF steps, the span map, agents, tools and labels."""

    def __init__(self, key: str, trace: Dict[str, Any]):
        self.key = key
        self.trace = trace
        self.steps: List[Dict[str, Any]] = []
        self.spans: Dict[str, Dict[str, Any]] = {}  # span id -> span record, in walk order
        self.duplicates: Dict[str, Dict[str, Any]] = {}
        self.agents: List[Dict[str, Any]] = []
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.groups: Dict[Tuple[str, str], _Group] = {}
        self.seen: Counter = Counter()  # message text -> occurrences read in any run of the trace

    # -- span tree ---------------------------------------------------------

    def walk(self) -> None:
        def visit(spans: List[Dict[str, Any]]) -> None:
            for span in sorted(spans, key=lambda s: s.get("timestamp") or ""):
                sid = span.get("span_id")
                if not isinstance(sid, str):
                    sid = f"ap-span-{len(self.spans) + len(self.duplicates)}"
                rec = self.span_record(span)
                if sid in self.spans:  # the same id again: kept once, the repeat is recorded
                    seen = self.duplicates.setdefault(sid, {"count": 1, "identical": True})
                    seen["count"] += 1
                    seen["identical"] = seen["identical"] and rec == self.spans[sid]
                else:
                    self.spans[sid] = rec
                visit(span.get("child_spans") or [])

        visit(self.trace.get("spans") or [])

    @staticmethod
    def span_record(span: Dict[str, Any]) -> Dict[str, Any]:
        attrs = span.get("span_attributes") or {}
        start = _iso_ns(span.get("timestamp"))
        length = _duration_ns(span.get("duration"))
        events = []
        for ev in span.get("events") or []:
            ev = ev if isinstance(ev, dict) else {"value": ev}
            events.append({"name": ev.get("Name"), "time_unix_nano": _iso_ns(ev.get("Timestamp")),
                           "attributes": ev.get("Attributes") or {}, "timestamp": ev.get("Timestamp"),
                           **{k: v for k, v in ev.items() if k not in ("Name", "Timestamp", "Attributes")}})
        return {
            "name": span.get("span_name"),
            "kind": attrs.get(KIND),
            "parent": span.get("parent_span_id"),
            "trace_id": span.get("trace_id"),
            "start_time_unix_nano": start,
            "end_time_unix_nano": start + length if start is not None and length is not None else None,
            "status": {"code": span.get("status_code"), "message": span.get("status_message")},
            "attributes": attrs,
            "events": events,
            "links": span.get("links") or [],
            "address": None,
            "source_fields": {k: v for k, v in span.items() if k not in SPAN_KEYS},
        }

    def owner(self, sid: str) -> Tuple[Tuple[str, str], Optional[str]]:
        """Group key and AGENT span id of an LLM span: the nearest AGENT ancestor, else the nearest non-CHAIN ancestor."""
        cur = sid
        seen = {sid}
        while True:
            parent = self.spans[cur]["parent"]
            if parent is None:
                return ("root", cur), None
            if parent not in self.spans or parent in seen:
                return ("missing", parent), None
            kind = self.spans[parent]["kind"]
            if kind == "AGENT":
                return ("agent", parent), parent
            if kind not in ("CHAIN", "LLM"):
                return ("wrapper", parent), None
            seen.add(parent)
            cur = parent

    # -- steps -------------------------------------------------------------

    def add(self, source: str, message: Any, sx: Dict[str, Any], **fields: Any) -> int:
        step = {"step_id": len(self.steps) + 1, "source": source, "message": message}
        step.update(fields)
        step["extra"] = {EXT: sx}
        self.steps.append(step)
        return len(self.steps) - 1

    @staticmethod
    def text(msg: Dict[str, Any], sx: Dict[str, Any]) -> str:
        content = msg.get("content", _ABSENT)
        if content is _ABSENT or content is None:
            sx["message_state"] = "missing" if content is _ABSENT else "null"
            return ""
        if not isinstance(content, str):
            sx["source_fields"] = dict(sx.get("source_fields") or {}, content=content)
            return json.dumps(content, ensure_ascii=False)
        return content

    def base(self, group: _Group, sid: str, part: str, index: int) -> Dict[str, Any]:
        sx: Dict[str, Any] = {"span_id": sid, "source_ref": {"span_id": sid, "part": part, "index": index}}
        if group.agent is not None:
            sx["agent"] = group.agent
        return sx

    def read_llm(self, sid: str) -> None:
        span = self.spans[sid]
        attrs = span["attributes"]
        gkey, agent = self.owner(sid)
        group = self.groups.setdefault(gkey, _Group(agent))
        # An LLM span outside any AGENT span (the final-answer call) re-renders the memory of the runs before
        # it, so its messages are compared with every message read so far in the trace.
        known = group.seen if group.agent is not None else self.seen
        seen_here: Counter = Counter()
        for n, msg in _indexed(attrs, INPUT):
            ident = msg.get("content")
            if ident is not None and not isinstance(ident, str):
                ident = repr(ident)  # attribute values are strings in the data; this keeps any other value hashable
            seen_here[ident] += 1
            if seen_here[ident] <= known[ident]:
                continue
            known[ident] += 1
            if known is not self.seen:
                self.seen[ident] += 1
            self.read_message(group, sid, n, msg)
        self.read_output(group, sid, span)

    def read_message(self, group: _Group, sid: str, n: int, msg: Dict[str, Any]) -> None:
        role = msg.get("role", _ABSENT)
        rest = {k: v for k, v in msg.items() if k not in ("role", "content")}
        if role in ("system", "user"):
            sx = self.base(group, sid, "input_message", n)
            if rest:
                sx["source_fields"] = rest
            self.add(role, self.text(msg, sx), sx)
        elif role in ("assistant", _ABSENT):
            self.read_rendering(group, sid, n, msg, role, rest)
        elif role == "tool-call":
            self.read_call_message(group, sid, n, msg, rest)
        elif role == "tool-response":
            self.read_response(group, sid, n, msg, rest)
        else:
            self.read_other(group, sid, n, msg, role, rest)

    def read_other(self, group: _Group, sid: str, n: int, msg: Dict[str, Any], role: Any, rest: Dict[str, Any]) -> int:
        sx = self.base(group, sid, "input_message", n)
        sx["role"] = None if role is _ABSENT else role
        if role is _ABSENT:
            sx["role_state"] = "missing"
        if rest:
            sx["source_fields"] = rest
        return self.add("user", self.text(msg, sx), sx)

    def read_rendering(self, group: _Group, sid: str, n: int, msg: Dict[str, Any], role: Any, rest: Dict[str, Any]) -> None:
        content = msg.get("content")
        text = _collapse(content) if isinstance(content, str) else ""
        for i, out in enumerate(group.outputs):
            if out["key"] and out["key"] in text:
                del group.outputs[i]
                notes = self.steps[out["step"]]["extra"][EXT].setdefault("history_copy", [])
                notes.append({"span_id": sid, "index": n})
                return
        if role is _ABSENT:
            self.read_other(group, sid, n, msg, role, rest)
            return
        sx = self.base(group, sid, "input_message", n)
        sx["history_only"] = True
        if group.agent is not None:
            sx["actor"] = group.agent
        if rest:
            sx["source_fields"] = rest
        group.last = self.add("agent", self.text(msg, sx), sx)

    def read_call_message(self, group: _Group, sid: str, n: int, msg: Dict[str, Any], rest: Dict[str, Any]) -> None:
        content = msg.get("content")
        calls = None
        if isinstance(content, str) and content.startswith(CALLING):
            try:
                calls = ast.literal_eval(content[len(CALLING):])
            except Exception:  # noqa: BLE001 - any parse failure falls back to a plain step
                calls = None
        if not (isinstance(calls, list) and calls and all(isinstance(c, dict) for c in calls)):
            self.read_other(group, sid, n, msg, "tool-call", rest)
            return
        ref = {"span_id": sid, "index": n}
        for tc in calls:
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            cid = tc.get("id")
            if isinstance(cid, str) and cid in group.calls:
                cx = self.steps[group.calls[cid]]["extra"][EXT]["calls"][cid]
                cx.setdefault("history_refs", []).append(ref)
                continue
            if group.last is None:
                sx = self.base(group, sid, "input_message", n)
                sx["history_only"] = True
                sx["message_state"] = "missing"
                group.last = self.add("agent", "", sx)
            step = self.steps[group.last]
            sx = step["extra"][EXT]
            used = [c["tool_call_id"] for c in step.get("tool_calls") or []]
            cx: Dict[str, Any] = {"span_id": sx.get("span_id"), "parent": self.spans[sx["span_id"]]["parent"],
                                  "call_from": "history_message", "paired_by": "latest_agent_step",
                                  "source_ref": dict(ref, part="input_message")}
            if not isinstance(cid, str) or not cid or cid in used:
                if cid is not None:
                    cx["source_id"] = cid
                cid = f"ap-{step['step_id']}-{len(used)}"
                cx["id_generated"] = True
            args, raw = _parse_arguments(fn.get("arguments"))
            if raw is not _ABSENT:
                cx["arguments_raw"] = raw
            name = fn.get("name")
            if not isinstance(name, str):
                cx["function_name_raw"] = name
            if name in self.tools:
                cx["tool_id"] = name
            extra = {k: v for k, v in tc.items() if k not in ("id", "type", "function")}
            if extra:
                cx["source_fields"] = extra
            step.setdefault("tool_calls", []).append(
                {"tool_call_id": cid, "function_name": name if isinstance(name, str) else "", "arguments": args})
            sx.setdefault("calls", {})[cid] = cx
            group.calls[cid] = group.last

    def read_response(self, group: _Group, sid: str, n: int, msg: Dict[str, Any], rest: Dict[str, Any]) -> None:
        content = msg.get("content")
        m = CALL_ID.match(content) if isinstance(content, str) else None
        cid = m.group(1) if m else None
        target = group.calls.get(cid) if cid is not None else None
        fields: Dict[str, Any] = {"role": "tool-response", **rest}
        if target is None:
            target = group.last
            if cid is not None:
                fields["tool_call_id"] = cid  # kept, though it matches no call of this run
        if target is None:  # no agent turn yet: keep the response as its own step
            self.read_other(group, sid, n, msg, "tool-response", rest)
            return
        step = self.steps[target]
        result = content
        if content is None or "content" not in msg:
            fields["content_state"] = "missing" if "content" not in msg else "null"
            result = ""
        elif not isinstance(content, (str, list)):
            fields["content_raw"] = content
            result = json.dumps(content, ensure_ascii=False)
        paired = cid is not None and group.calls.get(cid) == target
        step.setdefault("observation", {"results": []})["results"].append(
            {"source_call_id": cid if paired else None, "content": result})
        sx = step["extra"][EXT]
        sx.setdefault("result_refs", []).append({"span_id": sid, "index": n})
        sx.setdefault("result_fields", []).append(fields)

    def read_output(self, group: _Group, sid: str, span: Dict[str, Any]) -> None:
        attrs = span["attributes"]
        outputs: Dict[int, Dict[str, Any]] = {}
        for k, v in attrs.items():
            m = OUTPUT.fullmatch(k)
            if m:
                outputs.setdefault(int(m.group(1)), {})[m.group(2)] = v
        first = None
        for n in sorted(outputs):
            out = outputs[n]
            sx = self.base(group, sid, "output_message", n)
            if group.agent is not None:
                sx["actor"] = group.agent
            content = out.get("content", _ABSENT)
            message = self.text({"content": content} if content is not _ABSENT else {}, sx)
            calls, cx_all = [], {}
            raw_calls: Dict[int, Dict[str, Any]] = {}
            for k, v in out.items():
                m = OUTPUT_CALL.fullmatch(k)
                if m:
                    raw_calls.setdefault(int(m.group(1)), {})[m.group(2)] = v
            rest = {k: v for k, v in out.items() if k not in ("content",) and not OUTPUT_CALL.fullmatch(k)}
            role = rest.pop("role", None)
            if role not in (None, "assistant"):
                sx["role"] = role
            if rest:
                sx["source_fields"] = rest
            for k in sorted(raw_calls):
                tc = raw_calls[k]
                cid = tc.get("id")
                cx: Dict[str, Any] = {"span_id": sid, "parent": span["parent"],
                                      "source_ref": {"span_id": sid, "part": "output_message", "index": n, "call": k}}
                if not isinstance(cid, str) or not cid or cid in [c["tool_call_id"] for c in calls]:
                    if cid is not None:
                        cx["source_id"] = cid
                    cid = f"ap-{len(self.steps) + 1}-{k}"
                    cx["id_generated"] = True
                args, raw = _parse_arguments(tc.get("function.arguments"))
                if raw is not _ABSENT:
                    cx["arguments_raw"] = raw
                name = tc.get("function.name")
                if not isinstance(name, str):
                    cx["function_name_raw"] = name
                if name in self.tools:
                    cx["tool_id"] = name
                extra = {kk: vv for kk, vv in tc.items() if kk not in ("id", "function.name", "function.arguments")}
                if extra:
                    cx["source_fields"] = extra
                calls.append({"tool_call_id": cid, "function_name": name if isinstance(name, str) else "",
                              "arguments": args})
                cx_all[cid] = cx
            if cx_all:
                sx["calls"] = cx_all
            fields: Dict[str, Any] = {}
            if span["start_time_unix_nano"] is not None:
                fields["timestamp"] = span["source_fields"]["timestamp"]
            if isinstance(attrs.get("llm.model_name"), str):
                fields["model_name"] = attrs["llm.model_name"]
            metrics = self.metrics(attrs, sx)
            if metrics:
                fields["metrics"] = metrics
            if calls:
                fields["tool_calls"] = calls
            index = self.add("agent", message, sx, **fields)
            for cid in cx_all:
                group.calls[cid] = index
            group.last = index
            if message:
                group.outputs.append({"key": _output_key(message), "step": index})
            if first is None:
                first = index
        if first is not None:
            span["address"] = {"trajectory": self.key, "step": self.steps[first]["step_id"]}

    @staticmethod
    def metrics(attrs: Dict[str, Any], sx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        bad: Dict[str, Any] = {}
        for field in ("prompt", "completion", "total"):
            raw = attrs.get(f"llm.token_count.{field}")
            if raw is None:
                continue
            try:
                counts[field] = int(raw)
            except (TypeError, ValueError):
                bad[f"llm.token_count.{field}"] = raw
        if bad:
            sx["source_fields"] = dict(sx.get("source_fields") or {}, **bad)
        metrics: Dict[str, Any] = {}
        if "prompt" in counts:
            metrics["prompt_tokens"] = counts["prompt"]
        if "completion" in counts:
            metrics["completion_tokens"] = counts["completion"]
        if "total" in counts:
            metrics["extra"] = {"total_tokens": counts["total"]}
        return metrics or None

    # -- agents and tools ---------------------------------------------------

    def collect_tools(self) -> None:
        """Tools by name: the OpenAI-style definitions on LLM spans, else the TOOL span's own description."""
        for span in self.spans.values():
            attrs = span["attributes"]
            if span["kind"] == "LLM":
                for _, entry in _indexed(attrs, TOOL_DEF):
                    try:
                        definition = json.loads(entry["json_schema"])
                        name = definition["function"]["name"]
                    except (ValueError, KeyError, TypeError):
                        continue
                    if isinstance(name, str):
                        self.tools[name] = self.tool(name, definition, "openai-function-json-schema")
        for span in self.spans.values():
            attrs = span["attributes"]
            name = attrs.get("tool.name")
            lacking = name not in self.tools or self.tools[name]["definition"] is None
            if span["kind"] == "TOOL" and isinstance(name, str) and lacking:
                definition = {"name": name, "description": attrs.get("tool.description"),
                              "parameters": _parse_arguments(attrs.get("tool.parameters"))[0]
                              or attrs.get("tool.parameters")}
                self.tools[name] = self.tool(name, definition, "smolagents-tool-span")
        for span in self.spans.values():
            if span["kind"] == "AGENT":
                for name in self.tool_names(span):
                    self.tools.setdefault(name, self.tool(name, None, None))

    @staticmethod
    def tool(name: str, definition: Optional[Dict[str, Any]], fmt: Optional[str]) -> Dict[str, Any]:
        return {"name": name, "agent_or_server": None, "type": "function", "definition": definition,
                "definition_format": fmt, "atif_index": None, "permission": "unknown", "permission_from": "unknown"}

    @staticmethod
    def tool_names(span: Dict[str, Any]) -> List[str]:
        try:
            names = json.loads(span["attributes"].get("smolagents.tools_names"))
        except (TypeError, ValueError):
            return []
        return [n for n in names if isinstance(n, str)] if isinstance(names, list) else []

    def collect_agents(self) -> None:
        for sid, span in self.spans.items():
            if span["kind"] == "AGENT":
                self.agents.append({"id": sid, "name": span["name"], "instructions": None, "model": None,
                                    "tools": self.tool_names(span), "handoffs": []})

    def task_id(self) -> Optional[str]:
        """The GAIA ``task_id`` or SWE-Bench ``instance_id`` in the log of the harness span that called the agent."""
        for span in self.spans.values():
            for log in span["source_fields"].get("logs") or []:
                body = log.get("body") if isinstance(log, dict) else None
                args = body.get("function.arguments") if isinstance(body, dict) else None
                if not isinstance(args, dict):
                    continue
                for holder, field in (("example", "task_id"), ("item", "instance_id")):
                    item = args.get(holder)
                    if isinstance(item, dict) and isinstance(item.get(field), str):
                        return item[field]
        return None

    def task_text(self) -> Optional[str]:
        """The ``task`` argument of the outermost AGENT span (the first in walk order has no AGENT ancestor)."""
        for span in self.spans.values():
            if span["kind"] == "AGENT":
                attrs = span["attributes"]
                given = _parse_arguments(attrs.get("input.value"))[0].get("task")
                if isinstance(given, str):
                    return given
                return attrs["smolagents.task"] if isinstance(attrs.get("smolagents.task"), str) else None
        return None

    def run(self) -> None:
        self.walk()
        self.collect_tools()
        self.collect_agents()
        for sid, span in list(self.spans.items()):
            if span["kind"] == "LLM":
                self.read_llm(sid)


class Trail(Dataset):
    name = "trail"
    format = "trail-parquet"

    def records(self, root: Path, split: Optional[str] = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"trail takes split= only, got {sorted(options)}")
        if split is not None and split not in SPLITS:
            raise ValueError(f"split must be one of {list(SPLITS)}, got {split!r}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("reading TRAIL needs pyarrow: pip install 'agentpile[hf]'") from exc
        for sp in SPLITS:
            if split is not None and sp != split:
                continue
            for path in sorted((root / "data").glob(f"{sp}-*.parquet")):
                rel = path.relative_to(root).as_posix()
                row = 0
                for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=1, columns=["trace", "labels"]):
                    cells = batch.to_pydict()
                    text = cells["trace"][0]
                    try:
                        trace = json.loads(text)
                    except (TypeError, ValueError):
                        trace = None
                    tid = trace.get("trace_id") if isinstance(trace, dict) else None
                    key = f"{sp}/{tid}" if isinstance(tid, str) else f"{sp}/row-{row}"
                    raw = {"trace": trace, "trace_text": None if isinstance(trace, dict) else text,
                           "labels": cells["labels"][0], "row": row}
                    yield key, raw, {"file": rel, "config": "default", "split": sp}
                    row += 1

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        trace = raw["trace"] if isinstance(raw.get("trace"), dict) else {}
        conv = _Conversion(key, trace)
        conv.run()
        spans = conv.spans
        labels, label_info = self.labels(key, raw.get("labels"), spans)
        data: Dict[str, Any] = {"split": where.get("split"), "row": raw.get("row"), "trace_id": trace.get("trace_id"),
                                "span_count": len(spans), "labels_parse": label_info["parse"]}
        if raw.get("trace_text") is not None:
            data["trace_text"] = raw["trace_text"]
        data.update({k: v for k, v in trace.items() if k not in ("trace_id", "spans")})
        data.update(label_info["rest"])
        if conv.duplicates:
            data["duplicate_spans"] = conv.duplicates
        orphans = [sid for sid, s in spans.items() if s["parent"] is not None and s["parent"] not in spans]
        if orphans:
            data["spans_with_missing_parent"] = {sid: spans[sid]["parent"] for sid in orphans}
        scopes = sorted({(s["source_fields"].get("scope_name"), s["source_fields"].get("scope_version"))
                         for s in spans.values() if s["kind"] is not None}, key=str)
        data["instrumentation"] = [{"scope_name": a, "scope_version": b} for a, b in scopes]

        models = sorted({s["attributes"]["llm.model_name"] for s in spans.values()
                         if s["kind"] == "LLM" and isinstance(s["attributes"].get("llm.model_name"), str)})
        agent = {"name": "unknown", "version": "unknown"}
        scope_names = {a for a, _ in scopes}
        if len(scope_names) == 1 and isinstance(next(iter(scope_names)), str):
            agent["name"] = next(iter(scope_names)).rsplit(".", 1)[-1]
        if len(models) == 1:
            agent["model_name"] = models[0]

        steps = conv.steps
        if not steps:
            steps = [{"step_id": 1, "source": "system", "message": "",
                      "extra": {EXT: {"message_state": "missing", "empty_source": True}}}]
        ext = {
            "version": EXT_VERSION,
            "source": self.source(key, where),
            "task": {"id": conv.task_id(), "text": conv.task_text()},
            "outcome": {"success": None, "score": None, "raw": None},
            "labels": labels,
            "agents": conv.agents,
            "tools": conv.tools,
            "spans": spans,
            "dataset": data,
        }
        traj = {"schema_version": ATIF_VERSION, "session_id": key, "trajectory_id": key, "agent": agent,
                "steps": steps, "extra": {EXT: ext}}
        return Run(traj, TRAJECTORY)

    @staticmethod
    def labels(key: str, text: Any, spans: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        info: Dict[str, Any] = {"parse": "missing", "rest": {}}
        if not isinstance(text, str):
            return [], info
        try:
            doc, info["parse"] = _loads_tolerant(text)
        except ValueError:
            info["parse"] = "failed"
            info["rest"]["labels_text"] = text
            return [], info
        if not isinstance(doc, dict):
            info["parse"] = "failed"
            info["rest"]["labels_text"] = text
            return [], info
        if "trace_id" in doc:
            info["rest"]["labels_trace_id"] = doc["trace_id"]
        info["rest"].update({k: v for k, v in doc.items() if k not in ("trace_id", "errors", "scores")})
        run = {"kind": "run", "address": {"trajectory": key}}
        labels: List[Dict[str, Any]] = []
        for j, err in enumerate(doc.get("errors") or []):
            where = err.get("location") if isinstance(err, dict) else None
            ref = {"field": "errors", "index": j, "location": where, "resolved": isinstance(where, str) and where in spans}
            target = ({"kind": "span", "address": {"trajectory": key, "span": where}} if ref["resolved"] else run)
            labels.append({"name": "error", "target": target, "value": err, "label_source": "trail", "source_ref": ref})
        for j, score in enumerate(doc.get("scores") or []):
            if not isinstance(score, dict):
                labels.append({"name": "scores", "target": run, "value": score, "label_source": "trail",
                               "source_ref": {"field": "scores", "index": j}})
                continue
            for name, value in score.items():
                labels.append({"name": name, "target": run, "value": value, "label_source": "trail",
                               "source_ref": {"field": f"scores[{j}].{name}"}})
        return labels, info


dataset = Trail()
