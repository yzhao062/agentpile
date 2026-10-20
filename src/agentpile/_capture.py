"""Capture a live agent run as OTLP JSON and as a run record.

    import agentpile as ap

    with ap.capture(agent=my_agent) as cap:
        result = my_agent.run_sync("...")        # or an Agents SDK Runner call
    cap.otlp      # the finished spans as an OTLP JSON document (a dict)
    cap.run       # the run record, with the agent's configuration attached
    ap.config_from(my_agent)    # the agent's configuration as an agentpile-config/0 record

Supported agents: Pydantic AI (``pydantic_ai.Agent``) and the OpenAI Agents SDK
(``agents.Agent``); install ``agentpile[pydantic-ai]`` or ``agentpile[openai-agents]``.
Nothing here sets the global OpenTelemetry tracer provider. A capture adds
what it needs on entry, and on exit (also by exception) removes it and puts
back what it replaced; a host's own instrumentation, processors and exporters
keep working throughout, and a provider the caller owns is never shut down.

A capture records the runs in which one of its agents takes part and that
start while it is open, including runs started from other threads. A run of
other agents that happens in the same window is not recorded, and the
configuration is attached only to the agents that appear in a recorded run.

This module imports only the standard library. The OTLP encoder here is the
standard ``ExportTraceServiceRequest`` JSON encoding: ``resourceSpans``,
``scopeSpans``, hexadecimal ids, nanosecond times as strings, typed attribute
values.
"""

from __future__ import annotations

import base64
import copy
import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from .errors import AgentpileError
from .loading import load
from .record import CONFIG, EXT, TRAJECTORY, Run
from .registry import readers

READER = "otlp-json"
"""Name of the reader that turns an OTLP JSON document into a run (written separately from this module)."""


class CaptureError(AgentpileError):
    """A capture could not start, or its result cannot be built."""


# -- OTLP JSON encoder (standard library only) ---------------------------------

_SPAN_KIND = {"INTERNAL": 1, "SERVER": 2, "CLIENT": 3, "PRODUCER": 4, "CONSUMER": 5}
_STATUS_CODE = {"UNSET": 0, "OK": 1, "ERROR": 2}
_INT64 = 1 << 63


def _any_value(value: Any) -> Dict[str, Any]:
    """One OTLP ``AnyValue`` in its JSON form."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        if -_INT64 <= value < _INT64:
            return {"intValue": str(value)}
        return {"stringValue": str(value)}  # beyond int64: keep the digits, not a wrong number
    if isinstance(value, float):
        if math.isnan(value):
            return {"doubleValue": "NaN"}
        if math.isinf(value):
            return {"doubleValue": "Infinity" if value > 0 else "-Infinity"}
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytesValue": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": [{"key": str(k), "value": _any_value(v)} for k, v in value.items()]}}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_any_value(v) for v in value]}}
    if value is None:
        return {}
    return {"stringValue": str(value)}


def _attributes(attrs: Any) -> List[Dict[str, Any]]:
    if not attrs:
        return []
    return [{"key": str(k), "value": _any_value(v)} for k, v in attrs.items()]


def _enum_name(value: Any) -> str:
    return getattr(value, "name", None) or str(value).rsplit(".", 1)[-1]


def _hex_id(value: int, width: int) -> str:
    return format(int(value), "0%dx" % width)


def _trace_state(state: Any) -> Optional[str]:
    if not state:
        return None
    to_header = getattr(state, "to_header", None)
    text = to_header() if callable(to_header) else str(state)
    return text or None


def _span_document(span: Any) -> Dict[str, Any]:
    ctx = span.get_span_context() if hasattr(span, "get_span_context") else span.context
    doc: Dict[str, Any] = {
        "traceId": _hex_id(ctx.trace_id, 32),
        "spanId": _hex_id(ctx.span_id, 16),
    }
    state = _trace_state(getattr(ctx, "trace_state", None))
    if state:
        doc["traceState"] = state
    if span.parent is not None:
        doc["parentSpanId"] = _hex_id(span.parent.span_id, 16)
    doc["name"] = span.name
    doc["kind"] = _SPAN_KIND.get(_enum_name(span.kind), 0)
    doc["startTimeUnixNano"] = str(span.start_time if span.start_time is not None else 0)
    doc["endTimeUnixNano"] = str(span.end_time if span.end_time is not None else 0)
    doc["attributes"] = _attributes(span.attributes)
    dropped = getattr(span, "dropped_attributes", 0)
    if dropped:
        doc["droppedAttributesCount"] = int(dropped)
    events = []
    for event in span.events or ():
        e: Dict[str, Any] = {"timeUnixNano": str(event.timestamp), "name": event.name,
                             "attributes": _attributes(event.attributes)}
        if getattr(event, "dropped_attributes", 0):
            e["droppedAttributesCount"] = int(event.dropped_attributes)
        events.append(e)
    doc["events"] = events
    if getattr(span, "dropped_events", 0):
        doc["droppedEventsCount"] = int(span.dropped_events)
    links = []
    for link in span.links or ():
        lc = link.context
        entry: Dict[str, Any] = {"traceId": _hex_id(lc.trace_id, 32), "spanId": _hex_id(lc.span_id, 16),
                                 "attributes": _attributes(link.attributes)}
        lstate = _trace_state(getattr(lc, "trace_state", None))
        if lstate:
            entry["traceState"] = lstate
        links.append(entry)
    doc["links"] = links
    if getattr(span, "dropped_links", 0):
        doc["droppedLinksCount"] = int(span.dropped_links)
    status = span.status
    code = _STATUS_CODE.get(_enum_name(status.status_code), 0) if status is not None else 0
    sdoc: Dict[str, Any] = {"code": code}
    if status is not None and status.description:
        sdoc["message"] = status.description
    doc["status"] = sdoc
    return doc


def _resource_key(resource: Any) -> Any:
    attrs = getattr(resource, "attributes", None) or {}
    return (tuple(sorted((str(k), repr(v)) for k, v in attrs.items())), getattr(resource, "schema_url", "") or "")


def _scope_key(scope: Any) -> Any:
    if scope is None:
        return ("", "", "")
    return (getattr(scope, "name", "") or "", getattr(scope, "version", "") or "", getattr(scope, "schema_url", "") or "")


def encode_otlp(spans: Iterable[Any]) -> Dict[str, Any]:
    """Encode finished spans (OpenTelemetry SDK ``ReadableSpan`` objects) as an OTLP JSON ``ExportTraceServiceRequest``.

    Spans are grouped by resource and instrumentation scope, and ordered by
    start time (ties keep the order they finished in), so parents precede
    their children in the document.
    """
    ordered = sorted(enumerate(spans), key=lambda pair: (pair[1].start_time or 0, pair[0]))
    groups: Dict[Any, Dict[str, Any]] = {}
    for _, span in ordered:
        rkey = _resource_key(span.resource)
        group = groups.get(rkey)
        if group is None:
            resource = span.resource
            rdoc: Dict[str, Any] = {"attributes": _attributes(getattr(resource, "attributes", None))}
            group = {"resource": rdoc, "scopes": {}}
            schema_url = getattr(resource, "schema_url", "") or ""
            if schema_url:
                group["schemaUrl"] = schema_url
            groups[rkey] = group
        skey = _scope_key(getattr(span, "instrumentation_scope", None))
        scope_doc = group["scopes"].get(skey)
        if scope_doc is None:
            scope = getattr(span, "instrumentation_scope", None)
            sdoc: Dict[str, Any] = {"scope": {"name": skey[0]}, "spans": []}
            if skey[1]:
                sdoc["scope"]["version"] = skey[1]
            scope_attrs = getattr(scope, "attributes", None)
            if scope_attrs:
                sdoc["scope"]["attributes"] = _attributes(scope_attrs)
            if skey[2]:
                sdoc["schemaUrl"] = skey[2]
            group["scopes"][skey] = scope_doc = sdoc
        scope_doc["spans"].append(_span_document(span))
    resource_spans = []
    for group in groups.values():
        entry: Dict[str, Any] = {"resource": group["resource"], "scopeSpans": list(group["scopes"].values())}
        if "schemaUrl" in group:
            entry["schemaUrl"] = group["schemaUrl"]
        resource_spans.append(entry)
    return {"resourceSpans": resource_spans}


def count_spans(document: Dict[str, Any]) -> int:
    """Number of spans in an OTLP JSON document."""
    return sum(len(ss.get("spans") or []) for rs in document.get("resourceSpans") or []
               for ss in rs.get("scopeSpans") or [])


# -- attaching the configuration to a built run --------------------------------

def _fill(target: Dict[str, Any], source: Dict[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        if target.get(key) in (None, "") and source.get(key) not in (None, ""):
            target[key] = copy.deepcopy(source[key])


def _merge_unique(target: List[Any], extra: Iterable[Any]) -> None:
    for item in extra:
        if item not in target:
            target.append(item)


def _attach_to(record: Dict[str, Any], config: Dict[str, Any], top: bool) -> Set[str]:
    """Attach the configuration to the agents of one trajectory record; the ids of the configured agents that matched.

    The top-level record always gets its ``agents`` and ``tools`` containers.
    A subagent trajectory is left untouched unless it has an agent roster.
    """
    if top:
        ext = record.setdefault("extra", {}).setdefault(EXT, {})
    else:
        ext = (record.get("extra") or {}).get(EXT)
        if not isinstance(ext, dict) or not ext.get("agents"):
            return set()
    agents = ext.setdefault("agents", [])
    tools = ext.setdefault("tools", {})

    configured = {cfg["id"] for cfg in config.get("agents") or []}
    agent_ids: Dict[str, str] = {}
    for cfg in config.get("agents") or []:
        match = next((a for a in agents if a.get("id") == cfg["id"]), None)
        if match is None and cfg.get("name"):
            named = [a for a in agents if a.get("name") == cfg["name"]]
            match = named[0] if len(named) == 1 else None
        if match is None:
            continue
        _fill(match, cfg, ("name", "instructions", "model"))
        agent_ids[cfg["id"]] = match["id"]

    tool_ids: Dict[str, str] = {}
    for cid, cfg in (config.get("tools") or {}).items():
        owner_id = cfg.get("agent_or_server")
        if owner_id in configured and owner_id not in agent_ids:
            continue  # the tool of an agent that is not in this record
        owner = agent_ids.get(owner_id, owner_id)
        candidates = [(tid, t) for tid, t in tools.items() if t.get("name") == cfg.get("name")]
        narrowed = [(tid, t) for tid, t in candidates if t.get("agent_or_server") == owner]
        if not narrowed:
            narrowed = [(tid, t) for tid, t in candidates if t.get("agent_or_server") in (None, "")]
        if len(narrowed) == 1:
            tid, entry = narrowed[0]
            _fill(entry, cfg, ("agent_or_server", "type", "definition", "definition_format"))
            if entry.get("agent_or_server") in (None, ""):
                entry["agent_or_server"] = owner
        else:
            tid = cid
            n = 2
            while tid in tools:
                tid = f"{cid}#{n}"
                n += 1
            entry = copy.deepcopy(cfg)
            entry["agent_or_server"] = owner
            entry["atif_index"] = None
            tools[tid] = entry
        tool_ids[cid] = tid

    for cfg in config.get("agents") or []:
        if cfg["id"] not in agent_ids:
            continue
        match = next(a for a in agents if a.get("id") == agent_ids[cfg["id"]])
        _merge_unique(match.setdefault("tools", []), [tool_ids.get(t, t) for t in cfg.get("tools") or []])
        _merge_unique(match.setdefault("handoffs", []), [agent_ids.get(h, h) for h in cfg.get("handoffs") or []])
    return set(agent_ids)


def attach_config(record: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Merge a config record's agents and tools into a trajectory record, in place.

    The configuration belongs to the agents that appear in the record, so it
    is attached to those and to nothing else. What the trace itself recorded
    wins: a value the reader already set stays, and the configuration fills
    only what is empty (instructions, definition, handoffs). Agents match by
    id, then by unique name, in the record's own roster and in the roster of
    each of its subagent trajectories. A configured agent that matches no agent
    anywhere in the record, because it took no part in this run or because its
    name is ambiguous, is not added, and neither are its tools; its id goes into
    ``dataset.capture.config_agents_not_attached``. Tools match by name within
    the matching agent; a tool of a matched agent that matches nothing, or more
    than one entry, is added under the configuration's own id and never merged
    by guess. A handoff keeps the configured target even when that target is
    not an agent of the record.
    """
    matched = _attach_to(record, config, True)
    pending = list(record.get("subagent_trajectories") or [])
    while pending:
        sub = pending.pop(0)
        if isinstance(sub, dict):
            matched |= _attach_to(sub, config, False)
            pending.extend(sub.get("subagent_trajectories") or [])

    ext = record["extra"][EXT]
    note = ext.setdefault("dataset", {}).setdefault("capture", {})
    cds = config.get("dataset") or {}
    note["framework"] = cds.get("framework")
    note["framework_version"] = cds.get("framework_version")
    note["config_lost_fields"] = list((config.get("source") or {}).get("lost_fields") or [])
    not_attached = [cfg["id"] for cfg in config.get("agents") or [] if cfg["id"] not in matched]
    if not_attached:
        note["config_agents_not_attached"] = not_attached


def build_runs(otlp: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> List[Run]:
    """Read an OTLP JSON document with the ``otlp-json`` reader, then attach ``config`` to every trajectory.

    Raises ``CaptureError`` if the reader is not registered or the document holds no span.
    """
    if READER not in readers:
        raise CaptureError(
            f"the {READER!r} reader is not registered in this agentpile build, so a captured run cannot be turned "
            "into a run record; the OTLP document itself is available as capture.otlp")
    if not count_spans(otlp):
        raise CaptureError(
            "no span was captured: the agent did not run inside the capture, or the framework's tracing is switched "
            "off for that run (for example RunConfig(tracing_disabled=True) in the Agents SDK)")
    runs = list(load(otlp, reader=READER))
    if config is not None:
        for run in runs:
            if run.kind == TRAJECTORY:
                attach_config(run.record, config)
    return runs


# -- configuration -------------------------------------------------------------

def _agent_list(agent: Any) -> List[Any]:
    if agent is None:
        raise TypeError("an agent is required")
    if isinstance(agent, (list, tuple)):
        if not agent:
            raise TypeError("an empty list of agents")
        return list(agent)
    return [agent]


def config_from(agent: Any) -> Run:
    """The configuration of an agent as a config record (``agentpile-config/0``).

    It lists the agent's instructions and model, each tool with its full
    definition (name, description, input schema), and, for the OpenAI Agents
    SDK, handoffs and the agents they reach. A list of agents gives one
    record. What cannot be read without running the agent (dynamic
    instructions, the tools of an MCP server) is named in
    ``source.lost_fields``, never filled in.
    """
    from . import integrations

    agents = _agent_list(agent)
    by_framework: Dict[str, List[Any]] = {}
    for item in agents:
        by_framework.setdefault(integrations.framework_of(item), []).append(item)
    records = []
    for framework, members in by_framework.items():
        records.append(integrations.load(framework).config_record(members))
    record = records[0] if len(records) == 1 else _merge_configs(records)
    return Run(record, CONFIG)


def _merge_configs(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One record from the configs of agents of different frameworks (ids stay unique across them)."""
    first = copy.deepcopy(records[0])
    first["source"]["format"] = "+".join(r["source"]["format"] for r in records)
    first["dataset"] = {"frameworks": [r["dataset"] for r in records]}
    for other in records[1:]:
        taken = {a["id"] for a in first["agents"]}
        rename = {}
        for a in other["agents"]:
            if a["id"] in taken:
                rename[a["id"]] = f"{a['id']}@{other['source']['format']}"
        for a in copy.deepcopy(other["agents"]):
            a["id"] = rename.get(a["id"], a["id"])
            a["handoffs"] = [rename.get(h, h) for h in a.get("handoffs") or []]
            first["agents"].append(a)
        for tid, tool in copy.deepcopy(other["tools"]).items():
            tool["agent_or_server"] = rename.get(tool.get("agent_or_server"), tool.get("agent_or_server"))
            first["tools"][tid if tid not in first["tools"] else f"{tid}@{other['source']['format']}"] = tool
        first["source"]["lost_fields"] = list(first["source"]["lost_fields"]) + list(other["source"]["lost_fields"])
    return first


# -- the capture ---------------------------------------------------------------

class Capture:
    """The context manager returned by ``capture()``.

    After the block: ``otlp`` (the OTLP JSON document), ``runs`` and ``run``
    (the run records built from it, with the agent configuration attached),
    ``config`` (the configuration record) and ``span_count``. Inside the block
    the same attributes describe the spans that have finished so far.
    """

    def __init__(self, agents: Sequence[Any]) -> None:
        self._agents = list(agents)
        self._session: Any = None
        self._bindings: List[Any] = []
        self._state = "new"  # new -> open -> closed
        self._otlp: Optional[Dict[str, Any]] = None
        self._config: Optional[Run] = None
        self._config_error: Optional[str] = None
        self._runs: Optional[List[Run]] = None
        self._frameworks: Dict[str, List[Any]] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Capture":
        from . import integrations

        if self._state != "new":
            raise CaptureError("a capture can be entered once; make a new one with capture(agent)")
        for agent in self._agents:
            self._frameworks.setdefault(integrations.framework_of(agent), []).append(agent)
        modules = {name: integrations.load(name) for name in self._frameworks}  # fail before touching any agent
        try:
            from .integrations import _otel
        except ImportError as exc:
            raise CaptureError(
                f"capture needs the OpenTelemetry SDK ({exc}); install it with: pip install 'agentpile[otel]'") from exc
        self._session = _otel.Session()
        try:
            for name, members in self._frameworks.items():
                for agent in members:
                    binding = modules[name].Binding(agent, self._session)
                    binding.enter()
                    self._bindings.append(binding)
        except BaseException:
            self._release()
            raise
        self._state = "open"
        return self

    def _release(self) -> None:
        """Undo every binding (newest first), then close the session. A failed undo is reported, not raised."""
        for binding in reversed(self._bindings):
            try:
                binding.exit()
            except Exception as exc:
                warnings.warn(f"agentpile.capture: could not restore {type(binding).__name__}: {exc!r}",
                              RuntimeWarning, stacklevel=3)
        self._bindings = []
        if self._session is not None:
            self._session.close()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._state != "open":
            return None
        self._release()
        self._state = "closed"
        self._config = self._read_config()
        self._otlp = self._encode()
        if self._session.unfinished:
            warnings.warn(f"agentpile.capture: {self._session.unfinished} span(s) had not ended when the capture "
                          "closed and are not in the document", RuntimeWarning, stacklevel=2)
        return None

    async def __aenter__(self) -> "Capture":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return self.__exit__(exc_type, exc, tb)

    # -- results -----------------------------------------------------------

    def _require_started(self) -> None:
        if self._state == "new":
            raise CaptureError("the capture has not been entered: use it as `with capture(agent) as cap:`")

    def _encode(self) -> Dict[str, Any]:
        return encode_otlp(self._session.spans())

    @property
    def otlp(self) -> Dict[str, Any]:
        """The captured spans as an OTLP JSON document (``resourceSpans``, standard JSON encoding)."""
        self._require_started()
        return self._otlp if self._otlp is not None else self._encode()

    @property
    def span_count(self) -> int:
        self._require_started()
        return count_spans(self.otlp)

    def _read_config(self) -> Optional[Run]:
        try:
            return config_from(self._agents)
        except Exception as exc:  # a configuration that cannot be read must not lose the captured run
            self._config_error = repr(exc)
            warnings.warn(f"agentpile.capture: the agent configuration could not be read ({exc!r}); "
                          "the run record is built without it", RuntimeWarning, stacklevel=3)
            return None

    @property
    def config(self) -> Optional[Run]:
        """The configuration record of the captured agents (``None`` if it could not be read)."""
        self._require_started()
        if self._state == "closed":
            return self._config
        return self._read_config()

    @property
    def runs(self) -> List[Run]:
        """The run records built from the document, one per run the reader finds; needs the ``otlp-json`` reader."""
        self._require_started()
        if self._runs is not None:
            return list(self._runs)
        config = self.config
        runs = build_runs(self.otlp, config.record if config is not None else None)
        if self._state == "closed":
            self._runs = runs
        return list(runs)

    @property
    def run(self) -> Run:
        """The captured run record, with the agent configuration attached to its agents and tools."""
        runs = self.runs
        if len(runs) != 1:
            raise CaptureError(f"the capture holds {len(runs)} runs; read them from capture.runs")
        return runs[0]

    def __repr__(self) -> str:
        spans = self._session.exporter.get_finished_spans() if self._session is not None else ()
        return f"<Capture {self._state} agents={len(self._agents)} spans={len(spans)}>"


def capture(agent: Union[Any, Sequence[Any]]) -> Capture:
    """Capture the runs in which ``agent`` (one agent, or a list of agents) takes part, from entry to exit of the ``with`` block.

    Runs of other agents in the same window are not captured. Use ``async with``
    for an async run. See the module documentation for the lifecycle guarantees.
    """
    return Capture(_agent_list(agent))
