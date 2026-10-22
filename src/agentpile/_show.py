"""A readable terminal view of one run: what agentpile recovered from a source.

``show(run)`` prints a header (where the run came from, its task, agents, outcome, counts and run-level
labels), then the events in source order with step, role, actor and kind. Labels, dependencies and the marks
the record carries sit beside the event they belong to. A mark says where the record stands in for a value the
source did not have (a null message, absent arguments, a generated call id) or where a reader inferred a
pairing. Every mark comes from the record.

The plain text form uses the standard library only, is deterministic, and is what the tests check. When the
``rich`` library is installed (extra ``view``) the same content prints with colour and tables.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import IO, Any, Dict, Iterable, List, Optional, Tuple, Union

from .deps import derive_deps
from .record import CONFIG, EXT, MISSING, Event, Run

Segment = Tuple[str, str]  # (text, style name)
Line = List[Segment]

DEFAULT_WIDTH = 100
MIN_CONTENT = 20  # the widest column never shrinks below this, so a narrow terminal overflows instead of failing
SHORT_CELL = 24  # actor names in a narrow column are cut to this
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MEDIA_PARTS = ("image", "image_url", "input_image", "audio", "input_audio", "file", "input_file", "video")

SUBSTITUTED = "stand-in: content substituted in ATIF, source value kept"


# -- text helpers ----------------------------------------------------------------------

def _flat(text: str) -> str:
    """One line: whitespace runs become a space, and a control character (an ANSI escape) is spelled out."""
    return _CONTROL.sub(lambda m: "\\x%02x" % ord(m.group()), " ".join(text.split()))


def _clip(text: str, limit: Optional[int]) -> str:
    if limit is not None and len(text) > limit:
        return _flat(text[:limit]) + "... (+%d chars)" % (len(text) - limit)
    return _flat(text)


def _short(text: str, size: int = SHORT_CELL) -> str:
    return text if len(text) <= size else text[:size - 3] + "..."


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(value)


def _part_text(part: Any) -> str:
    """Text of one content part: a text part or bare string as is, a named media part as a placeholder."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text") and isinstance(part.get("text"), str):
            return part["text"]
        if kind in MEDIA_PARTS:
            return "[%s part]" % kind
    return _json(part)


def _content_text(value: Any) -> str:
    """Text of a message or result content: a string as is, a list of parts joined, anything else as JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return " ".join(_part_text(part) for part in value)
    return _json(value)


def _ext(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return ((obj or {}).get("extra") or {}).get(EXT) or {}


def _join(*groups: Line) -> Line:
    """The segments of several groups on one line, two spaces between groups."""
    out: Line = []
    for group in groups:
        if group:
            if out:
                out.append(("  ", ""))
            out.extend(group)
    return out


def _flatten(line: Line) -> str:
    return "".join(text for text, _style in line)


def _mark(text: str) -> Line:
    return [("[%s]" % text, "mark")]


# -- the view model ----------------------------------------------------------------------

@dataclass
class _Row:
    cells: List[Line]  # one per column; the last column is the wide one
    notes: List[Line] = field(default_factory=list)  # lines under the last column


@dataclass
class _Section:
    heading: str
    columns: List[Tuple[str, str]] = field(default_factory=list)  # (name, "l" or "r")
    rows: List[_Row] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)


@dataclass
class _View:
    title: str
    header: List[Tuple[str, List[Line]]]
    sections: List[_Section]


# -- labels and dependencies as notes --------------------------------------------------------

def _label_line(label: Dict[str, Any], limit: Optional[int], arrow: bool = True) -> Line:
    """``<- kind label name = value (source)``, the form a label takes beside the thing it targets."""
    target = label.get("target") or {}
    kind = target.get("kind") or "?"
    out: Line = [(("<- " if arrow else "") + "%s label " % kind, "label"),
                 (str(label.get("name") or "(unnamed)"), "name"), (" = " + _clip(_json(label.get("value")), limit), "")]
    if label.get("label_source") is not None:
        out.append((" (%s)" % label["label_source"], "dim"))
    if kind == "span":
        out.append((", span %s" % (target.get("address") or {}).get("span"), "dim"))
    return out


def _address_text(address: Dict[str, Any]) -> str:
    parts = ["step %s" % address.get("step")]
    if address.get("call") is not None:
        parts.append(str(address["call"]))
    if address.get("result") is not None:
        parts.append("result %s" % address["result"])
    return " ".join(parts)


def _ranges(numbers: List[int]) -> str:
    out, i = [], 0
    nums = sorted(set(numbers))
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if i == j else "%d-%d" % (nums[i], nums[j]))
        i = j + 1
    return ", ".join(out)


def _dep_groups(deps: Iterable[Dict[str, Any]], limit: Optional[int]) -> List[Tuple[Dict[str, Any], Line]]:
    """``(destination address, note line)`` for each destination and rule, with its sources listed together."""
    groups: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], str, Any, List[Dict[str, Any]]]] = {}
    for dep in deps:
        dst, src = dep.get("dst") or {}, dep.get("src") or {}
        resource = dep.get("resource")
        key = (dst.get("step"), dst.get("call"), dst.get("result"), dep.get("rule"), _json(resource))
        groups.setdefault(key, (dst, dep.get("rule"), resource, []))[3].append(src)
    out = []
    for dst, rule, resource, sources in groups.values():
        plain = [s for s in sources if s.get("call") is None and s.get("result") is None
                 and isinstance(s.get("step"), int)]
        named = [s for s in sources if s not in plain]
        pieces = []
        if plain:
            pieces.append(("step %s" if len({s["step"] for s in plain}) == 1 else "steps %s") % _ranges(
                [s["step"] for s in plain]))
        pieces.extend(_address_text(s) for s in named)
        tail = str(rule)
        if resource not in (None, [], ""):
            tail += ", " + _clip(", ".join(map(str, resource)) if isinstance(resource, list) else str(resource), limit)
        out.append((dst, [("<- depends on %s (%s)" % (", ".join(pieces), tail), "dep")]))
    return out


# -- the trajectory body --------------------------------------------------------------------

def _agent_names(run: Run) -> Dict[Any, str]:
    """Agent id -> name, for the ids whose name is set and not shared with another agent."""
    named = {a.get("id"): a.get("name") for a in run.agents if a.get("id") is not None}
    counts = Counter(n for n in named.values() if n)
    return {i: n for i, n in named.items() if n and counts[n] == 1}


def _message_body(event: Event, sx: Dict[str, Any], limit: Optional[int]) -> Line:
    if event.content is None:
        main: Line = _mark("stand-in: source was null")
    else:
        text = _clip(_content_text(event.content), limit)
        main = [(text, "")] if text else [("(empty)", "dim")]
    marks: List[Line] = []
    if "message_raw" in sx:
        marks.append(_mark(SUBSTITUTED))
    if sx.get("role_state") == "missing":
        marks.append(_mark("stand-in: source had no role"))
    if sx.get("history_only"):
        marks.append(_mark("history only"))
    return _join(main, *marks)


def _call_body(event: Event, cx: Dict[str, Any], limit: Optional[int]) -> Line:
    name = event.name or ""
    main: Line = [(name or "(unnamed)", "name")]
    args = event.content
    marks: List[Line] = []
    if args is MISSING:
        marks.append(_mark("stand-in: source had no arguments"))
    elif args is None:
        marks.append(_mark("stand-in: source arguments were null"))
    else:
        main.append((" " + _clip(args if isinstance(args, str) else _json(args), limit), ""))
    if cx.get("arguments_state") == "unparsed":
        marks.append(_mark("stand-in: arguments were not a JSON object, ATIF holds {}"))
    if "arguments_stand_in" in cx:
        marks.append(_mark("stand-in: arguments rebuilt as {}: %s" % _clip(str(cx["arguments_stand_in"]), limit)))
    if cx.get("id_generated"):
        was = cx.get("source_id")
        marks.append(_mark("stand-in: call id generated" + ("" if was is None else ", source id was %s" % _json(was))))
    if "function_name_raw" in cx:
        marks.append(_mark("stand-in: source function name was %s" % _clip(_json(cx["function_name_raw"]), limit)))
    if cx.get("call_from"):
        marks.append(_mark("call from %s" % str(cx["call_from"]).replace("_", " ")))
    if cx.get("paired_by"):
        marks.append(_mark("paired by %s" % cx["paired_by"]))
    ident: Line = [("id=%s" % event.call_id, "dim")] if event.call_id is not None else []
    return _join(main, ident, *marks)


def _result_body(event: Event, rf: Dict[str, Any], limit: Optional[int]) -> Line:
    marks: List[Line] = []
    if event.content is MISSING:
        main: Line = _mark("stand-in: source had no content")
    elif event.content is None:
        main = _mark("stand-in: source content was null")
    else:
        text = _clip(_content_text(event.content), limit)
        main = [(text, "")] if text else [("(empty)", "dim")]
    if "content_raw" in rf:
        marks.append(_mark(SUBSTITUTED))
    if rf.get("paired_by"):
        marks.append(_mark("paired by %s" % rf["paired_by"]))
    ident: Line = [("answers %s" % event.call_id, "dim")] if event.call_id is not None else []
    return _join(main, ident, *marks)


def _trajectory_rows(run: Run, events: List[Event], labels: List[Dict[str, Any]],
                     deps: Optional[List[Dict[str, Any]]], limit: Optional[int],
                     ) -> Tuple[_Section, List[Tuple[Dict[str, Any], str]], List[Line]]:
    """The events of one trajectory as a table, with the labels and dependencies placed on them.

    Also returns the labels (with the reason) and the dependencies that name nothing the events hold.
    Run-level labels are not placed here; the header shows them.
    """
    steps = {s.get("step_id"): s for s in run.steps}
    names = _agent_names(run)
    spans = run.spans
    answered = {(e.address.get("step"), e.call_id) for e in events if e.kind == "result" and e.call_id is not None}

    anchor: Dict[Any, int] = {}  # step id -> the event that stands for the step (its message, else its first event)
    call_at: Dict[Tuple[Any, Any], int] = {}
    for i, e in enumerate(events):
        step = e.address.get("step")
        if step not in anchor or (e.kind == "message" and events[anchor[step]].kind != "message"):
            anchor[step] = i
        if e.kind == "call":
            call_at[(step, e.call_id)] = i

    def locate(kind: Any, addr: Dict[str, Any]) -> Tuple[List[int], str]:
        if kind == "step":
            i = anchor.get(addr.get("step"))
            return ([] if i is None else [i]), "step %s has no events" % addr.get("step")
        if kind == "call":
            if addr.get("step") is not None:
                i = call_at.get((addr.get("step"), addr.get("call")))
                return ([] if i is None else [i]), "step %s has no call %s" % (addr.get("step"), addr.get("call"))
            hits = [i for (_s, c), i in call_at.items() if c == addr.get("call")]
            return hits, "no call %s" % addr.get("call")
        if kind == "agent":
            return ([i for i, e in enumerate(events) if e.actor == addr.get("agent")],
                    "no event by agent %s" % addr.get("agent"))
        if kind == "tool":
            return ([i for i, e in enumerate(events) if e.kind == "call" and e.extra.get("tool_id") == addr.get("tool")],
                    "no call of tool %s" % addr.get("tool"))
        if kind == "span":
            sid = addr.get("span")
            span = spans.get(sid)
            if span is None:
                return [], "span %s is not in this record" % sid
            where = span.get("address") or {}
            if where.get("call") is not None:
                return locate("call", where)
            if where.get("step") is not None:
                return locate("step", where)
            what = " ".join(str(x) for x in (span.get("kind"), span.get("name")) if x)
            return [], "span %s%s maps to no step or call" % (sid, " (%s)" % what if what else "")
        return [], "a %s target is not one this view places" % kind

    notes: List[List[Line]] = [[] for _ in events]
    for i, e in enumerate(events):
        if e.error and (e.kind == "result" or (e.address.get("step"), e.call_id) not in answered):
            text = e.error if isinstance(e.error, str) else _json(e.error)
            notes[i].append([("error: " + _clip(text, limit), "error")])

    unplaced_labels: List[Tuple[Dict[str, Any], str]] = []
    for label in labels:
        target = label.get("target") or {}
        if target.get("kind") == "run":
            continue
        hits, why = locate(target.get("kind"), target.get("address") or {})
        if not hits:
            unplaced_labels.append((label, why))
        for i in hits:
            notes[i].append(_label_line(label, limit))

    unplaced_deps: List[Line] = []
    for dst, line in _dep_groups(deps or [], limit):
        hits, why = locate("call" if dst.get("call") is not None else "step", dst)
        if not hits:
            unplaced_deps.append(line + [(" -- at %s" % _address_text(dst), "dim")])
        for i in hits:
            notes[i].append(line)

    show_actor = any(e.actor is not None for e in events)
    columns = [("step", "r"), ("role", "l")] + ([("actor", "l")] if show_actor else []) + [("kind", "l"),
                                                                                       ("content", "l")]
    rows = []
    for i, e in enumerate(events):
        step = steps.get(e.address.get("step")) or {}
        sx = _ext(step)
        cx = (sx.get("calls") or {}).get(e.call_id) or {}
        if e.kind == "message":
            body = _message_body(e, sx, limit)
        elif e.kind == "call":
            body = _call_body(e, cx, limit)
        else:
            rfs = sx.get("result_fields") or []
            n = e.address.get("result")
            body = _result_body(e, (rfs[n] if isinstance(n, int) and n < len(rfs) else None) or {}, limit)
        cells = [[(str(e.address.get("step")), "dim")], [(str(e.role or "-"), "role")]]
        if show_actor:
            actor = "-" if e.actor is None else _short(str(names.get(e.actor, e.actor)))
            cells.append([(actor, "")])
        cells.append([(e.kind, "kind_" + e.kind)])
        cells.append(body)
        rows.append(_Row(cells, notes[i]))
    section = _Section("events", columns, rows)
    if not events:
        placeholder = any(_ext(s).get("empty_source") for s in run.steps)
        section.lines.append([("no events: " + (
            "the source held none, and ATIF needs one step, so the record keeps an empty placeholder (empty_source)"
            if placeholder else "the steps hold no messages, calls or results"), "dim")])
    return section, unplaced_labels, unplaced_deps


# -- the header ------------------------------------------------------------------------------

def _source_lines(src: Dict[str, Any]) -> List[Line]:
    """Where the run came from: the dataset and its short revision, else the format, then the file."""
    if src.get("dataset"):
        first = str(src["dataset"]) + (" @ " + str(src["revision"])[:7] if src.get("revision") else "")
    else:
        first = "format %s" % src["format"] if src.get("format") else "unknown source"
    lines: List[Line] = [[(first, "")]]
    if src.get("file"):
        lines.append([("file %s" % src["file"], "dim")])
    return lines


def _roster(run: Run) -> List[Line]:
    agents = run.agents
    if agents:
        return [[(str(a.get("name") or a.get("id")), "name")] + ([(" (%s)" % a["model"], "dim")] if a.get("model") else [])
                for a in agents]
    atif = run.record.get("agent") or {}
    if run.kind == "trajectory" and (atif.get("model_name") or atif.get("name") not in (None, "unknown")):
        return [[(str(atif.get("name")), "name")] + ([(" (%s)" % atif["model_name"], "dim")]
                                                      if atif.get("model_name") else [])]
    return []


def _outcome(run: Run, limit: Optional[int]) -> str:
    outcome = run.outcome
    bits = []
    for key in ("success", "score"):
        if outcome.get(key) is not None:
            bits.append("%s=%s" % (key, _json(outcome[key])))
    if outcome.get("raw") is not None:
        bits.append("raw=" + _clip(_json(outcome["raw"]), limit))
    return ", ".join(bits)


def _header(run: Run, events: List[Event], limit: Optional[int]) -> List[Tuple[str, List[Line]]]:
    src = run.source
    config = run.kind == CONFIG
    out: List[Tuple[str, List[Line]]] = [
        ("record", [[("config (agentpile-config/0)" if config else
                      "trajectory (%s)" % (run.record.get("schema_version") or "no schema_version"), "")]]),
        ("from", _source_lines(src))]
    text = run.task.get("text")
    if text:
        out.append(("task", [[(_clip(str(text), limit), "")]]))
    elif run.task.get("id") not in (None, run.id):
        out.append(("task", [[("id %s (no task text recorded)" % run.task["id"], "dim")]]))
    else:
        instructions = next((a for a in run.agents if a.get("instructions")), None)
        if instructions is not None:
            out.append(("instruction", [[(_clip(str(instructions["instructions"]), limit), "")]]))
    roster = _roster(run)
    if roster and not config:  # a config record lists its agents in a table of their own
        out.append(("agents", roster))
    outcome = _outcome(run, limit)
    if outcome:
        out.append(("outcome", [[(outcome, "")]]))
    if config:
        out.append(("steps", [[("none: a configuration with no execution, so there are no events", "")]]))
        out.append(("counts", [[("agents %d, tools %d" % (len(run.agents), len(run.tools)), "")]]))
    else:
        count = Counter(e.kind for e in events)
        steps = sum(1 for s in run.steps if not _ext(s).get("empty_source"))
        line = "steps %d, messages %d, calls %d, results %d" % (steps, count["message"], count["call"], count["result"])
        subs = len(run.subagents)
        out.append(("counts", [[(line + (", subagents %d" % subs if subs else ""), "")]]))
    # A run label addressed to a subagent's trajectory belongs to that subagent's section, not to this header.
    run_labels = [lab for lab in run.labels if (lab.get("target") or {}).get("kind") == "run"
                  and ((lab.get("target") or {}).get("address") or {}).get("trajectory") in (None, run.id)]
    if run_labels:
        out.append(("labels", [[(str(lab.get("name") or "(unnamed)"), "name"),
                                (" = " + _clip(_json(lab.get("value")), limit), "")]
                               + ([(" (%s)" % lab["label_source"], "dim")] if lab.get("label_source") is not None else [])
                               for lab in run_labels]))
    caveats = []
    if src.get("lost_fields"):
        caveats.append("source fields the reader did not keep: %s" % ", ".join(map(str, src["lost_fields"])))
    if src.get("dialect_tested") is False:
        caveats.append("read from an instrumentation version that was not tested")
    if caveats:
        out.append(("notes", [[(c, "mark")] for c in caveats]))
    return out


# -- the config body ----------------------------------------------------------------------

def _config_sections(run: Run, limit: Optional[int]) -> Tuple[List[_Section], List[Tuple[Dict[str, Any], str]]]:
    labels = [lab for lab in run.labels if (lab.get("target") or {}).get("kind") != "run"]
    by_target: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    tool_ids = set(run.tools)
    names = Counter(t.get("name") for t in run.tools.values())
    agent_ids = {a.get("id") for a in run.agents}
    unplaced: List[Tuple[Dict[str, Any], str]] = []
    for lab in labels:
        target = lab.get("target") or {}
        kind, addr = target.get("kind"), target.get("address") or {}
        if kind == "tool":
            want = addr.get("tool")
            key = want if want in tool_ids else next(
                (i for i, t in run.tools.items() if t.get("name") == want and names[want] == 1), None)
            if key is None:
                unplaced.append((lab, "no tool %s in this record" % want))
            else:
                by_target.setdefault(("tool", key), []).append(lab)
        elif kind == "agent":
            if addr.get("agent") in agent_ids:
                by_target.setdefault(("agent", addr.get("agent")), []).append(lab)
            else:
                unplaced.append((lab, "no agent %s in this record" % addr.get("agent")))
        else:
            unplaced.append((lab, "a %s target is not one this view places" % kind))

    sections: List[_Section] = []
    if run.agents:
        rows = []
        for a in run.agents:
            cells = [[(str(a.get("name") or a.get("id")), "name")], [(str(a.get("model") or "-"), "")],
                     [(str(len(a.get("tools") or [])), "")],
                     [(_clip(str(a["instructions"]), limit), "")] if a.get("instructions") else [("-", "dim")]]
            rows.append(_Row(cells, [_label_line(lab, limit) for lab in by_target.get(("agent", a.get("id")), [])]))
        sections.append(_Section("agents", [("agent", "l"), ("model", "l"), ("tools", "r"), ("instructions", "l")], rows))
    if run.tools:
        rows = []
        for tid, t in run.tools.items():
            shown = str(t.get("name") or tid)
            details = ", ".join(x for x in (
                "owner %s" % t["agent_or_server"] if t.get("agent_or_server") else "",
                "type %s" % t["type"] if t.get("type") else "",
                "%s definition" % t["definition_format"] if t.get("definition_format") else "") if x)
            cells = [[(_short(shown if tid == shown else "%s (%s)" % (shown, tid), 40), "name")],
                     [(str(t.get("permission") or "unknown"), "")],
                     [(str(t.get("permission_from") or "unknown"), "dim")],
                     [(details or "-", "dim")]]
            rows.append(_Row(cells, [_label_line(lab, limit) for lab in by_target.get(("tool", tid), [])]))
        sections.append(_Section("tools", [("tool", "l"), ("permission", "l"), ("from", "l"), ("details", "l")], rows))
    return sections, unplaced


# -- building the whole view ----------------------------------------------------------------

def _rule_args(deps: Any) -> Optional[Dict[str, Any]]:
    if deps is None or deps is False:
        return None
    if deps is True:
        return {}
    return {"rules": [deps] if isinstance(deps, str) or callable(deps) else list(deps)}


def _trajectories(run: Run) -> List[Run]:
    """The run and its subagent trajectories, each followed by its own subagents."""
    out = [run]
    for sub in run.subagents:
        out.extend(_trajectories(sub))
    return out


def _build(run: Run, deps: Any, limit: Optional[int]) -> _View:
    rules = _rule_args(deps)
    title = "agentpile %s: %s" % (run.kind, run.id if run.id is not None else "(no id)")
    lost_deps: List[Line] = []
    events: List[Event] = []
    if run.kind == CONFIG:
        sections, unplaced = _config_sections(run, limit)
    else:
        events = list(run.events())
        trajectories = _trajectories(run)
        pool: Dict[int, List[Dict[str, Any]]] = {id(t): [] for t in trajectories}
        unplaced = []
        for label in run.labels:  # a label belongs to the trajectory its address names, else to the run holding it
            named = ((label.get("target") or {}).get("address") or {}).get("trajectory")
            owner = trajectories[0] if named is None else next((t for t in trajectories if t.id == named), None)
            if owner is None:
                unplaced.append((label, "addresses trajectory %s, which this record does not hold" % named))
            else:
                pool[id(owner)].append(label)
        sections = []
        for t in trajectories:
            labels = pool[id(t)] + (t.labels if t is not run else [])
            section, lost_labels, lost = _trajectory_rows(
                t, events if t is run else list(t.events()), labels,
                derive_deps(t, **rules) if rules is not None else None, limit)
            if t is not run:
                section.heading = "subagent %s" % t.id
                section.lines += [_label_line(lab, limit, arrow=False) for lab in labels
                                  if (lab.get("target") or {}).get("kind") == "run"]
            sections.append(section)
            unplaced += lost_labels
            lost_deps += lost
    if unplaced:
        sections.append(_Section("labels with nothing to sit beside", lines=[
            _label_line(lab, limit, arrow=False) + [(" -- %s" % why, "dim")] for lab, why in unplaced]))
    if lost_deps:
        sections.append(_Section("dependencies with nothing to sit beside", lines=lost_deps))
    return _View(title, _header(run, events, limit), sections)


# -- rendering --------------------------------------------------------------------------------

def _wrap(text: str, width: int, hang: str = "") -> List[str]:
    return textwrap.wrap(text, width=max(width, 10), subsequent_indent=hang, break_long_words=True,
                         break_on_hyphens=False) or [""]


def _plain_table(section: _Section, width: int) -> List[str]:
    columns = section.columns
    widths = [max([len(name)] + [len(_flatten(r.cells[c])) for r in section.rows]) for c, (name, _j) in
              enumerate(columns[:-1])]
    indent = 2
    wide = max(width - indent - sum(w + 2 for w in widths), MIN_CONTENT)
    pad = " " * (indent + sum(w + 2 for w in widths))

    def put(cells: List[str]) -> str:
        return " " * indent + "".join(
            (c.rjust(w) if columns[i][1] == "r" else c.ljust(w)) + "  " for i, (c, w) in enumerate(zip(cells, widths)))

    out = [(put([name for name, _j in columns[:-1]]) + columns[-1][0]).rstrip(),
           (" " * indent + "  ".join("-" * w for w in widths) + "  " + "-" * len(columns[-1][0])).rstrip()]
    for row in section.rows:
        first = put([_flatten(c) for c in row.cells[:-1]])
        for k, piece in enumerate(_wrap(_flatten(row.cells[-1]), wide)):
            out.append(((first if k == 0 else pad) + piece).rstrip())
        for note in row.notes:
            for piece in _wrap(_flatten(note), wide, hang="   "):
                out.append((pad + piece).rstrip())
    return out


def _render_plain(view: _View, width: int) -> str:
    out = [view.title]
    key_width = max(len(k) for k, _lines in view.header)
    value_at = 2 + key_width + 2
    for key, lines in view.header:
        for n, line in enumerate(lines):
            lead = "  " + (key if n == 0 else "").ljust(key_width) + "  "
            for k, piece in enumerate(_wrap(_flatten(line), width - value_at, hang="  ")):
                out.append(((lead if k == 0 else " " * value_at) + piece).rstrip())
    for section in view.sections:
        out += ["", section.heading]
        if section.rows:
            out += _plain_table(section, width)
        for line in section.lines:
            out += ["  " + piece for piece in _wrap(_flatten(line), width - 2, hang="  ")]
    return "\n".join(out) + "\n"


# Labels, the reason most people open a run, get the strongest color.
_RICH_STYLES = {"dim": "dim", "name": "bold", "role": "bold", "mark": "yellow", "label": "bold #ED8D5A",
                "dep": "#4098AC", "error": "red", "kind_call": "yellow", "kind_result": "green", "kind_message": "blue"}


def _rich_text(line: Line):
    from rich.text import Text

    return Text.assemble(*[(text, _RICH_STYLES.get(style, "")) for text, style in line])


def _render_rich(view: _View, console: Any) -> None:
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    for key, lines in view.header:
        grid.add_row(Text(key), Text("\n").join([_rich_text(line) for line in lines]))
    console.print(Panel(grid, title=Text(view.title, style="bold"), title_align="left", box=box.ROUNDED,
                        padding=(0, 1)))
    for section in view.sections:
        console.print()
        console.print(Text(section.heading, style="bold"))
        if section.rows:
            table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold dim")
            for i, (name, just) in enumerate(section.columns):
                last = i == len(section.columns) - 1
                table.add_column(Text(name), justify="right" if just == "r" else "left", no_wrap=not last,
                                 overflow="fold", ratio=1 if last else None)
            blanks = [Text("")] * (len(section.columns) - 1)
            for row in section.rows:
                table.add_row(*[_rich_text(c) for c in row.cells])
                for note in row.notes:
                    table.add_row(*blanks, _rich_text(note))
            console.print(table)
        for line in section.lines:
            console.print(Text("  ") + _rich_text(line))


# -- the public function ------------------------------------------------------------------------

def show(run: Run, *, deps: Union[None, bool, str, Iterable[Any]] = None, width: Optional[int] = None,
         max_chars: Optional[int] = 300, file: Optional[IO[str]] = None, plain: Optional[bool] = None) -> None:
    """Print a readable view of one run: header, events in source order, and the labels and marks on them.

    ``deps`` adds the dependencies ``derive_deps`` infers at their destination events: ``True`` for its
    default rules, or a rule name or list of rules (names or callables) passed through. ``max_chars`` cuts
    each text (content, arguments, label value) and says how much was left out; ``None`` keeps all of it.
    ``file`` is a text stream (default ``sys.stdout``) or a rich ``Console``. The view uses ``rich`` when it
    is installed (``pip install 'agentpile[view]'``); ``plain=True`` prints deterministic, ASCII-only layout
    text instead (the source text itself stays as it is). ``plain=False`` requires rich.
    """
    if not isinstance(run, Run):
        raise TypeError("show() takes a Run, got %s" % type(run).__name__)
    if max_chars is not None and max_chars < 1:
        raise ValueError("max_chars must be at least 1 or None, got %r" % (max_chars,))
    if width is not None and width < 1:
        raise ValueError("width must be positive, got %r" % (width,))
    console = None
    if plain is not True:
        try:
            from rich.console import Console
        except ImportError as exc:
            if plain is False:
                raise ImportError("a rich view needs the view extra: pip install 'agentpile[view]'") from exc
        else:
            console = file if isinstance(file, Console) else Console(file=file, width=width, highlight=False)
    view = _build(run, deps, max_chars)
    if console is not None:
        _render_rich(view, console)
        return
    out = sys.stdout if file is None else file if hasattr(file, "write") else file.file  # a Console wraps its stream
    if width is None:
        try:
            tty = bool(out.isatty())
        except (AttributeError, ValueError):
            tty = False
        width = shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns if tty else DEFAULT_WIDTH
    out.write(_render_plain(view, width))
