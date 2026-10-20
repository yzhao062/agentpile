"""Dependency rules: inference the caller chooses, kept apart from what ``load()`` observed.

``derive_deps(run, rules=[...])`` returns dependency entries
``{src, dst, rule, rule_version, resource}``, where ``src`` and ``dst`` are full
record addresses and ``dst`` depends on ``src``. Every entry names the rule and
version that produced it, so results from different rules, or from two versions
of one rule, never mix silently. A rule that lacks the fields it needs produces
no edge; it never guesses.

Rules that need to know what a tool call touches read it through an
*extractor*, which is named and versioned like a rule. An extractor turns a run
into ``Access`` values, ``(address, resource, mode, tool)``, one per event it
recognizes. A resource the data does not give is ``None`` and never produces an
edge. Entries of the four resource rules carry one extra key, ``extractor``,
the id of the extractor that supplied the resource.

Rules:

- ``tool_call_id_v1``: the next agent step after a result depends on the call that produced it.
- ``full_context_v1``: every step depends on every earlier step.
- ``resource_chain_v1``: an access depends on the previous access to the same resource, whatever the mode.
- ``last_writer_v1``: an access depends on the last write to its resource.
- ``prior_reads_v1``: a write depends on every earlier read, for tools with no resource id.
- ``value_flow_v1``: an id-like value in a result feeds the later calls whose arguments contain it.

Extractors (one per tool family; each may read action text, which is allowed
here and never happens in ``load()``):

- ``file_editor_paths_v1``: file editor tools and the file path in their arguments.
- ``shell_commands_v1``: the command line of a SWE-agent style turn (``open``, ``edit``, ``cat``, ...).
- ``web_urls_v1``: the page URL of a browser action.
- ``db_tool_names_v1``: database tools classified by name prefix, with no resource id.
- ``world_location_v1``: the agent location in a text world.

``value_flow_v1`` finds its own ids (``id_like_values_v1``, the pattern in ``ID_LIKE``), which is the extractor
name its entries carry.

A caller picks an extractor with the rule factories, for example
``derive_deps(run, rules=[resource_chain(web_urls_v1)])``; the registered rule
names use the default extractor of their family.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

from .record import CONFIG, Event, Run

Dep = Dict[str, Any]
Rule = Callable[[Run], List[Dep]]


def _dep(src: Dict[str, Any], dst: Dict[str, Any], rule: str, version: str, resource: Any = None,
         extractor: Optional[str] = None) -> Dep:
    entry = {"src": src, "dst": dst, "rule": rule, "rule_version": version, "resource": resource}
    if extractor is not None:
        entry["extractor"] = extractor
    return entry


def tool_call_id_v1(run: Run) -> List[Dep]:
    """The next agent step after a tool result depends on the call that produced it.

    Calls and results are paired by tool call id only. A result without a
    ``source_call_id`` yields no edge, and steps by the user or the system in
    between do not consume the pending results. The rule walks the run in
    recorded source order (``Run.events``), not in ATIF step order: a result
    becomes available only when its own event is reached, so an agent step that
    the source recorded before the result never depends on it, however early
    ATIF stores that result.
    """
    tid = run.id
    deps: List[Dep] = []
    pending: Dict[Tuple[Any, Any], Dict[str, Any]] = {}  # (step, call id) -> call address, in result order
    for step, event in run._timeline():
        sid = step.get("step_id")
        if event is None:  # the step starts here
            if step.get("source") == "agent" and pending:
                for src in pending.values():
                    deps.append(_dep(src, {"trajectory": tid, "step": sid}, "tool_call_id", "1"))
                pending = {}
        elif event.kind == "result" and event.call_id is not None:
            if event.call_id in {c.get("tool_call_id") for c in step.get("tool_calls") or []}:
                pending.setdefault((sid, event.call_id), {"trajectory": tid, "step": sid, "call": event.call_id})
    return deps


def full_context_v1(run: Run) -> List[Dep]:
    """Every step depends on every earlier step (the assumption GRADE uses for Who&When)."""
    tid = run.id
    ids = [step.get("step_id") for step in run.steps]
    return [_dep({"trajectory": tid, "step": ids[j]}, {"trajectory": tid, "step": ids[i]}, "full_context", "1")
            for i in range(len(ids)) for j in range(i)]


# -- accesses and extractors ----------------------------------------------------

READ = "read"
WRITE = "write"
RUN = "run"


class Access(NamedTuple):
    """What one event touches, as an extractor read it.

    ``address`` is the full address of the event (a call, or a step for a
    text action). ``resource`` is a family-qualified id such as
    ``file:/repo/a.py``, or ``None`` when the data gives none. ``mode`` is
    ``"read"``, ``"write"``, ``"run"`` or ``None`` (the event touches no tracked
    state); only ``"write"`` has a special meaning in the rules. ``tool`` is the
    tool name or command head, kept for the audit trail.
    """

    address: Dict[str, Any]
    resource: Optional[str]
    mode: Optional[str]
    tool: Optional[str]


class Extractor:
    """A named, versioned function ``run -> [Access]``, in event order."""

    def __init__(self, name: str, version: str, fn: Callable[[Run], List[Access]]):
        self.name, self.version, self._fn = name, version, fn
        self.__doc__ = fn.__doc__

    @property
    def id(self) -> str:
        return f"{self.name}_v{self.version}"

    def __call__(self, run: Run) -> List[Access]:
        return self._fn(run)

    def __repr__(self) -> str:
        return f"<Extractor {self.id}>"


def _ext(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return (((obj or {}).get("extra") or {}).get("agentpile")) or {}


def _step_address(run: Run, step: Dict[str, Any]) -> Dict[str, Any]:
    return {"trajectory": run.id, "step": step.get("step_id")}


def _arguments(ev: Event) -> Dict[str, Any]:
    """The arguments of a call event as an object; anything else is no arguments."""
    value = ev.content
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


EDITOR_TOOLS = ("str_replace_editor", "str_replace_based_edit_tool", "edit_file")
EDITOR_WRITE_COMMANDS = ("create", "str_replace", "insert", "append", "write")
SHELL_TOOLS = ("bash", "execute_bash", "run")
END_TOOLS = ("finish", "submit")


def _file_editor_paths(run: Run) -> List[Access]:
    """File editor tools and the path in their arguments (the OpenHands tool family).

    The resource is ``file:<path>`` with the path exactly as written, from the
    ``path`` argument or else ``file``; a value that is not a non-empty string is
    no path, and arguments that are not an object give none. The commands ``create``, ``str_replace``,
    ``insert``, ``append`` and ``write`` are writes; every other editor command,
    a missing one included, is a read. Shell tools are runs with no resource,
    because the command line is not parsed. ``finish`` and ``submit`` are writes
    with no resource, and any other tool touches nothing.
    """
    out: List[Access] = []
    for ev in run.events():
        if ev.kind != "call":
            continue
        name = ev.name
        if name in EDITOR_TOOLS:
            args = _arguments(ev)
            path = args.get("path") or args.get("file")
            command = args.get("command")
            mode = WRITE if isinstance(command, str) and command.lower() in EDITOR_WRITE_COMMANDS else READ
            out.append(Access(ev.address, f"file:{path}" if isinstance(path, str) and path else None, mode, name))
        elif name in SHELL_TOOLS:
            out.append(Access(ev.address, None, RUN, name))
        elif name in END_TOOLS:
            out.append(Access(ev.address, None, WRITE, name))
        else:
            out.append(Access(ev.address, None, None, name))
    return out


_OPEN_FILE_RE = re.compile(r"\(Open file:\s*(.+?)\)")
_BLOCK_RE = re.compile(r"```(?:bash|python)?\s*(.*?)```", re.DOTALL)


def _open_file_footer(text: str) -> Optional[str]:
    """Path of the open file named by an observation footer: ``""`` for none, ``None`` for no footer."""
    m = _OPEN_FILE_RE.search(text or "")
    if not m:
        return None
    value = m.group(1).strip()
    return "" if value in ("n/a", "") else value


def _shell_action(text: str, open_file: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(command, file, mode) of the action in a turn: the first line of its last fenced block."""
    blocks = _BLOCK_RE.findall(text or "")
    if not blocks or not blocks[-1].strip():
        return None, None, None
    line = blocks[-1].strip().splitlines()[0].strip()
    if not line:
        return None, None, None
    try:
        tok = shlex.split(line)
    except ValueError:
        tok = line.split()
    if not tok:
        return None, None, None
    c = tok[0]
    if c in ("open", "create"):
        return c, tok[1] if len(tok) > 1 else None, WRITE if c == "create" else READ
    if c in ("edit", "insert", "append"):
        return c, open_file, WRITE
    if c in ("goto", "scroll_up", "scroll_down"):
        return c, open_file, READ
    if c == "search_file":
        return c, tok[2] if len(tok) > 2 else open_file, READ
    if c == "cat":
        return c, tok[1] if len(tok) > 1 else None, READ
    if c == "grep":
        args = [a for a in tok[1:] if not a.startswith("-")]
        recursive = any(a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
                        for a in tok)
        f = args[1] if len(args) >= 2 else None
        if f and (recursive or f in (".", "./") or f.endswith("/")):
            f = None
        return c, f, READ
    if c in ("find_file", "search_dir", "ls"):
        return c, None, READ
    if c == "submit":
        return c, None, WRITE
    return c, None, RUN


def _shell_commands(run: Run) -> List[Access]:
    """The command of each agent turn in the SWE-agent command language, with the file it touches.

    Turns are agent steps whose message holds the action in its last fenced
    block. The environment's output after a turn carries an ``(Open file: ...)``
    footer that sets the file ``edit``, ``insert``, ``append``, ``goto``,
    ``scroll_*`` and ``search_file`` act on; ``open`` and ``create`` set it too.
    The output is read from the results of the agent step it follows (where a
    reader puts an observation) and from user steps (where a flat message list
    puts it); a footer of ``n/a`` clears the file, and an output with no footer
    leaves it. A command with no single file (``ls``, ``find_file``,
    ``search_dir``, a recursive or directory ``grep``, a test run) has no
    resource. File paths are not normalized.
    """
    out: List[Access] = []
    open_file: Optional[str] = None

    def observe(text: str) -> None:
        nonlocal open_file
        footer = _open_file_footer(text)
        if footer is not None:
            open_file = footer or None

    for step in run.steps:
        source = step.get("source")
        if source == "user":
            observe(_text(step.get("message")))
            continue
        if source != "agent":
            continue
        cmd, f, mode = _shell_action(_text(step.get("message")), open_file)
        if cmd is not None:
            out.append(Access(_step_address(run, step), f"file:{f}" if f else None, mode, cmd))
            if f and cmd in ("open", "create"):
                open_file = f
        for result in (step.get("observation") or {}).get("results") or []:
            content = result.get("content")
            observe(content if isinstance(content, str) else "")
    return out


WEB_WRITE_ACTIONS = ("fill", "click", "select_option", "press", "check", "clear", "dblclick", "hover",
                     "upload_file", "goto", "go_back", "go_forward")


def _record_urls(run: Run) -> Dict[int, str]:
    """Page URL of each source record, by record index, from where the reader keeps the observation.

    The observation of record ``j`` sits in the result of the call before it
    (``result_refs`` gives ``j``, ``result_fields`` the url); the observation of
    record 0 sits in the ``observation`` field of the user step made from the
    goal; an observation with no call before it is in the run's
    ``unpaired_observations`` with its record index.
    """
    urls: Dict[int, str] = {}
    for step in run.steps:
        sx = _ext(step)
        fields = sx.get("source_fields") or {}
        if step.get("source") == "user" and sx.get("source_ref") == "goal" and isinstance(fields.get("observation"), dict):
            urls[0] = _text(fields["observation"].get("url"))
        refs = sx.get("result_refs") or []
        for i, rf in enumerate(sx.get("result_fields") or []):
            if i < len(refs) and isinstance(refs[i], int) and not isinstance(refs[i], bool):
                urls[refs[i]] = _text((rf or {}).get("url"))
    for item in run.dataset.get("unpaired_observations") or []:
        if isinstance(item, dict) and isinstance(item.get("record"), int):
            urls[item["record"]] = _text(item.get("url"))
    return urls


def _web_action(step: Dict[str, Any]) -> str:
    """The action string of a browser step: the whole string of its call, else the action the step kept."""
    calls = step.get("tool_calls") or []
    if calls:
        cx = (_ext(step).get("calls") or {}).get(calls[0].get("tool_call_id")) or {}
        return _text(cx["arguments_raw"]) if "arguments_raw" in cx else _text(calls[0].get("function_name"))
    return _text((_ext(step).get("source_fields") or {}).get("action"))


def _web_urls(run: Run) -> List[Access]:
    """The page URL of each browser step, with the action type as the mode.

    A browser step is an agent step made from a source record; the page is the
    URL of that record, the page the agent saw when it chose the action. The
    resource is ``url:<url>``, the string as given, query and fragment included.
    The action type is the text of the action string before the first ``(``; a
    write action is one that interacts with or navigates the page, any other
    action (or none, which is the tool ``noop``) is a read. A step with no URL has no resource.
    """
    urls = _record_urls(run)
    out: List[Access] = []
    for step in run.steps:
        if step.get("source") != "agent":
            continue
        ref = _ext(step).get("source_ref")
        url = urls.get(ref, "") if isinstance(ref, int) and not isinstance(ref, bool) else ""
        action = _web_action(step).strip()
        atype = action.split("(", 1)[0].strip() if action else ""
        address = _step_address(run, step)
        calls = step.get("tool_calls") or []
        if calls:
            address = dict(address, call=calls[0].get("tool_call_id"))
        out.append(Access(address, f"url:{url}" if url else None, WRITE if atype in WEB_WRITE_ACTIONS else READ,
                          atype or "noop"))
    return out


DB_WRITE_PREFIXES = ("book_", "cancel_", "modify_", "update_", "return_", "exchange_", "send_")
DB_NON_DB_TOOLS = ("calculate", "think", "transfer_to_human_agents", "list_all_airports")


def _db_tool_names(run: Run) -> List[Access]:
    """Database tools classified by name; they carry no resource id.

    A name that starts with ``book_``, ``cancel_``, ``modify_``, ``update_``,
    ``return_``, ``exchange_`` or ``send_`` is a write. Any other name is a read,
    except local computation, a no-op thought, a human handoff and a static
    reference lookup (``calculate``, ``think``, ``transfer_to_human_agents``,
    ``list_all_airports``), which touch no database state. The lists are the
    retail and airline tool sets of tau-bench.
    """
    out: List[Access] = []
    for ev in run.events():
        if ev.kind != "call":
            continue
        name = ev.name
        if name and name.startswith(DB_WRITE_PREFIXES):
            mode: Optional[str] = WRITE
        elif name and name not in DB_NON_DB_TOOLS:
            mode = READ
        else:
            mode = None
        out.append(Access(ev.address, None, mode, name))
    return out


LOCATION_MOVES = ("go to", "teleport to")
_ACTION_LINE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)


def _world_location(run: Run) -> List[Access]:
    """The agent location in a text world: one resource, ``world:location``.

    Every agent turn with an ``Action:`` in its text is an access to the
    location. ``go to`` and ``teleport to`` change it (writes); every other action
    is valid only in the current room (a read). The verb is the first word of the
    action line, lowercased, or ``noop`` when the line is empty.
    """
    out: List[Access] = []
    for step in run.steps:
        if step.get("source") != "agent":
            continue
        text = _text(step.get("message"))
        if "action:" not in text.lower():
            continue
        am = _ACTION_LINE.search(text)
        action = (am.group(1).splitlines()[0].strip() if am else "").lower()
        verb = action.split()[0] if action else "noop"
        move = any(action.startswith(w) for w in LOCATION_MOVES)
        out.append(Access(_step_address(run, step), "world:location", WRITE if move else READ, verb))
    return out


file_editor_paths_v1 = Extractor("file_editor_paths", "1", _file_editor_paths)
shell_commands_v1 = Extractor("shell_commands", "1", _shell_commands)
web_urls_v1 = Extractor("web_urls", "1", _web_urls)
db_tool_names_v1 = Extractor("db_tool_names", "1", _db_tool_names)
world_location_v1 = Extractor("world_location", "1", _world_location)

EXTRACTORS: Dict[str, Extractor] = {e.id: e for e in (
    file_editor_paths_v1, shell_commands_v1, web_urls_v1, db_tool_names_v1, world_location_v1)}


# -- resource rules ---------------------------------------------------------------

def _named(rule: Rule, name: str, version: str, doc: str) -> Rule:
    rule.__name__ = f"{name}_v{version}"
    rule.__qualname__ = rule.__name__
    rule.__doc__ = doc
    return rule


def resource_chain(extractor: Optional[Extractor] = None) -> Rule:
    """``resource_chain_v1`` with a chosen extractor (default ``file_editor_paths_v1``).

    An access depends on the previous access to the same resource, whatever the
    mode: a read after a read gets an edge. Accesses with no resource get none.
    """
    ex = extractor or file_editor_paths_v1

    def rule(run: Run) -> List[Dep]:
        last: Dict[str, Dict[str, Any]] = {}
        out: List[Dep] = []
        for acc in ex(run):
            if acc.resource is None:
                continue
            prev = last.get(acc.resource)
            if prev is not None:
                out.append(_dep(dict(prev), dict(acc.address), "resource_chain", "1", acc.resource, ex.id))
            last[acc.resource] = acc.address
        return out

    return _named(rule, "resource_chain", "1", resource_chain.__doc__)


def last_writer(extractor: Optional[Extractor] = None) -> Rule:
    """``last_writer_v1`` with a chosen extractor (default ``file_editor_paths_v1``).

    An access depends on the last write to its resource; only writes move the
    pointer, so a write also depends on the write before it. Accesses with no
    resource get no edge.
    """
    ex = extractor or file_editor_paths_v1

    def rule(run: Run) -> List[Dep]:
        last_write: Dict[str, Dict[str, Any]] = {}
        out: List[Dep] = []
        for acc in ex(run):
            if acc.resource is None:
                continue
            prev = last_write.get(acc.resource)
            if prev is not None:
                out.append(_dep(dict(prev), dict(acc.address), "last_writer", "1", acc.resource, ex.id))
            if acc.mode == WRITE:
                last_write[acc.resource] = acc.address
        return out

    return _named(rule, "last_writer", "1", last_writer.__doc__)


def prior_reads(extractor: Optional[Extractor] = None) -> Rule:
    """``prior_reads_v1`` with a chosen extractor (default ``db_tool_names_v1``).

    For accesses that have no resource id, a write depends on every earlier read
    of the run (a conservative temporal upper bound: the data does not say which
    read the write used) and a read depends on nothing. Accesses of any other
    mode, and accesses that have a resource, take no part.
    """
    ex = extractor or db_tool_names_v1

    def rule(run: Run) -> List[Dep]:
        reads: List[Dict[str, Any]] = []
        out: List[Dep] = []
        for acc in ex(run):
            if acc.resource is not None:
                continue
            if acc.mode == WRITE:
                out.extend(_dep(dict(r), dict(acc.address), "prior_reads", "1", None, ex.id) for r in reads)
            elif acc.mode == READ:
                reads.append(acc.address)
        return out

    return _named(rule, "prior_reads", "1", prior_reads.__doc__)


resource_chain_v1 = resource_chain()
last_writer_v1 = last_writer()
prior_reads_v1 = prior_reads()


# -- value flow -------------------------------------------------------------------

#: id-like value: up to four letters, an optional separator, a digit, then two or more word characters or hyphens
ID_LIKE = re.compile(r"^[A-Za-z]{0,4}[-_]?\d[\w\-]{2,}$")
VALUE_FLOW_EXTRACTOR = "id_like_values_v1"


def _string_values(obj: Any) -> Iterable[str]:
    """Every string scalar nested in dict values and list items; keys are not scanned."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_values(v)


def _id_values(obj: Any) -> List[str]:
    return sorted({v for v in _string_values(obj) if ID_LIKE.match(v)})


def _own_step(step: Dict[str, Any]) -> bool:
    """An agent step whose message was an assistant message (not a user message promoted to an agent step)."""
    return step.get("source") == "agent" and _ext(step).get("role", "assistant") == "assistant"


def value_flow_v1(run: Run) -> List[Dep]:
    """An id-like value in a result feeds the later calls whose arguments contain it.

    The semantics are ``tau2-frozen-v1``:

    - **Ids.** A value is id-like when it is a string that matches
      ``^[A-Za-z]{0,4}[-_]?\\d[\\w\\-]{2,}$``. They are read from every nested string
      value of a call's arguments (an object; any other arguments give none) and of
      the result (parsed as JSON when it is a string, the string itself when it
      does not parse). The heuristic can match non-ids such as phone numbers.
    - **Producer.** A result produces its ids for the call it answers, as
      recorded in the result's ``source_call_id``; a result with no paired call
      produces nothing. When several results contain an id, the latest one
      wins. A call sees only the results that arrived before it was issued.
    - **Consumer.** A call with several id-like values that have a producer gets
      exactly one dependency, on the producer that comes latest in the run. The
      entry's ``resource`` lists the ids that point at that producer.
    - **Scope.** Calls (and the results they answer) on steps that come from an
      assistant message. A call on an agent step promoted from a user message,
      such as a simulated user acting on its own device, is neither a consumer
      nor a producer.

    Calls and results are paired by the reader, never here.
    """
    steps = {s.get("step_id"): (pos, s) for pos, s in enumerate(run.steps)}
    timeline: List[Tuple[Tuple[int, int, int], Event]] = []
    for ev in run.events():
        if ev.kind == "message":
            continue
        pos, step = steps[ev.address["step"]]
        if not _own_step(step):
            continue
        sx = _ext(step)
        ref = sx.get("source_ref")
        base = ref if isinstance(ref, int) else pos
        if ev.kind == "call":
            timeline.append(((base, 0, len(timeline)), ev))
        else:
            i = ev.address["result"]
            refs = sx.get("result_refs") or []
            r = refs[i] if i < len(refs) and isinstance(refs[i], int) else base
            timeline.append(((r, 1, i), ev))
    timeline.sort(key=lambda item: item[0])

    calls: Dict[Tuple[Any, Any], Tuple[int, Dict[str, Any]]] = {}
    last_on_id: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    out: List[Dep] = []
    for _key, ev in timeline:
        if ev.kind == "call":
            ordinal = len(calls)
            calls[(ev.address["step"], ev.call_id)] = (ordinal, ev.address)
            args = ev.content
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            ids = _id_values(args) if isinstance(args, dict) else []
            found = [(last_on_id[v], v) for v in ids if v in last_on_id]
            if found:
                (ordinal_src, addr_src), _ = max(found, key=lambda item: item[0][0])
                via = [v for (producer, v) in found if producer[0] == ordinal_src]
                out.append(_dep(dict(addr_src), dict(ev.address), "value_flow", "1",
                                [f"value:{v}" for v in sorted(via)], VALUE_FLOW_EXTRACTOR))
        else:
            produced = calls.get((ev.address["step"], ev.call_id)) if ev.call_id is not None else None
            if produced is None:
                continue
            content = ev.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except ValueError:
                    content = {}
            for v in _id_values(content):
                last_on_id[v] = produced
    return out


RULES: Dict[str, Rule] = {
    "tool_call_id_v1": tool_call_id_v1,
    "full_context_v1": full_context_v1,
    "resource_chain_v1": resource_chain_v1,
    "last_writer_v1": last_writer_v1,
    "prior_reads_v1": prior_reads_v1,
    "value_flow_v1": value_flow_v1,
}


def derive_deps(run: Run, rules: Iterable[Union[str, Rule]] = ("tool_call_id_v1",)) -> List[Dep]:
    """Dependency entries for ``run`` under the named rules, in rule order.

    A rule is a name from ``RULES`` (``"full_context"`` means its latest
    version) or a callable ``rule(run) -> list`` returning entries of the same
    shape. Config records have no steps and get no entries.
    """
    if run.kind == CONFIG:
        return []
    out: List[Dep] = []
    for rule in rules:
        if callable(rule):
            out.extend(rule(run))
            continue
        versions = [r for r in RULES if r.rsplit("_v", 1)[0] == rule]
        name = rule if rule in RULES else max(versions, key=lambda r: int(r.rsplit("_v", 1)[1]), default=None)
        if name is None:
            raise KeyError(f"unknown dependency rule {rule!r} (known: {', '.join(RULES)})")
        out.extend(RULES[name](run))
    return out
