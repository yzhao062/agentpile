"""OTLP JSON: read traces into run records, and write run records as OTLP JSON (standard library only).

Both directions work on the OTLP/JSON encoding of ``ExportTraceServiceRequest``
(``{"resourceSpans": [{"resource", "scopeSpans": [{"scope", "spans"}]}]}``), as one
JSON document or as JSONL with one document per line (the OTel file exporter
format). Ids are hex (base64 is accepted too), enums are integers or names, int64
values are strings or numbers, ``AnyValue`` is decoded in all its forms, and the
snake-case and legacy (``instrumentationLibrarySpans``) spellings are accepted.

Writer. A trajectory becomes the span structure of Harbor's ATIF-to-OTel converter: one
AGENT root span (``invoke_agent``), one span per step, one TOOL span per tool call
(a child of the trajectory span, like the step spans), and a nested AGENT span per
entry of ``subagent_trajectories``. An agent step is an LLM span; a user or system step is a
CHAIN span named ``user_message`` or ``system_message``, never an LLM call. Each span carries two
groups of attributes:

- The record. ``agentpile.*`` attributes hold every ATIF field and every agentpile extension
  field as JSON text, one attribute per key (``agentpile.step.message``, ``agentpile.ext.actor``).
  An absent attribute is a missing value, ``null`` is null, ``""`` is an empty string. Nothing is
  truncated or normalized, and the reader rebuilds the record from these attributes alone.
- A view for observability backends: OpenInference attributes (``openinference.span.kind``,
  ``input.value``, ``llm.*``) and OTel GenAI attributes (``gen_ai.operation.name``,
  ``gen_ai.agent.name``, ``gen_ai.request.model``, ``gen_ai.tool.name``, ``gen_ai.tool.call.id``,
  ``gen_ai.tool.call.arguments``, ``gen_ai.tool.call.result``; tool attributes follow GenAI v1.41.0).
  The reader never uses the view for spans agentpile wrote. The view shows a call's arguments as the source
  gave them: the parsed object on a model span, the source value on a tool span, and the source value on a model
  span too where ATIF holds a stand-in {} for it (``arguments_state`` "unparsed"); nothing where the source recorded none.

Dependency entries (a step's ``deps``) become span links carrying ``agentpile.dep.rule``,
``agentpile.dep.rule_version`` and ``agentpile.dep.resource``. A step without a timestamp gets a
synthetic, increasing time marked ``agentpile.time_synthetic``; the reader restores the original
timestamp and never reports a synthetic time as a source time.

A run that carries source spans (the extension's ``spans`` field: a trace read from OTLP, TRAIL) is
written without repeating the source's operations. A kept span that a step, a call or a subagent
trajectory came from (its ``address`` names an agent step or a call; a subagent trajectory is keyed by
the span's id) is written as that step, call or subagent root: the span keeps its id, name, kind, parent,
times, status, events, links and attributes, and gains only the ``agentpile.*`` attributes that rebuild
the record (``agentpile.span`` holds the kept entry itself; ``agentpile.owner`` and ``agentpile.span_owner``
name the trajectory spans that hold the step or call and the entry, because the source's parent chain
need not lead to them). Only an unambiguous match counts: an id that two steps, calls or trajectories
share claims nothing, and a source span without a match is written as an ordinary span, once. Spans
agentpile adds are then not operations: a trajectory is a CHAIN span ``trajectory``, a user or system
step a CHAIN span, and a step or call that no source span stands behind a CHAIN span (``agent_step``,
``tool_call``) without model, token or tool attributes. So the written trace has the chat, tool and
agent operations the source had, and the token totals it had.

Reader. Spans written by agentpile rebuild the exact record. Spans of two framework dialects are
mapped into ATIF steps (``DIALECTS``, ``MAPPING``): Pydantic AI 2.58.0 to 2.60.0 (OTel GenAI conventions)
and the OpenAI Agents SDK 0.24.0 through openinference-instrumentation-openai-agents 2.5.3 (OpenInference).
Calls and results pair by tool call id only and nothing is inferred. A dialect is interpreted only for
the instrumentation scope versions it was tested with: any other version raises ``UnsupportedVersion``
naming the tested ones, and ``allow_untested_version=True`` reads it anyway and marks the run
(``source.dialect_tested`` false, a ``lost_fields`` note). Spans of any other dialect are read
generically: the run holds one empty placeholder step, and ``source.lost_fields`` says what was not
mapped. In every case each span is kept in the extension's ``spans`` field, with the address of the step
or call it became (or null).
"""

from __future__ import annotations

import base64
import calendar
import datetime as _dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .._io import _is_jsonl, _open_text
from ..errors import UnsupportedVersion
from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run

name = "otlp-json"
READER_VERSION = "1"

# Supported dialects: instrumentation scope name -> (dialect name, tested scope versions). Pydantic AI 2.59.0
# and 2.60.0 build the same records as 2.58.0 from the fixture scripts; each adds one key to the spans'
# model_request_parameters (2.60.0: introduced_tool_names), which the reader does not use.
DIALECTS = {
    "pydantic-ai": ("pydantic-ai/genai", ("2.58.0", "2.59.0", "2.60.0")),
    "openinference.instrumentation.openai_agents": ("openinference/openai-agents", ("2.5.3",)),
}
_TESTED = {dialect: versions for dialect, versions in DIALECTS.values()}
_SCOPE_OF = {dialect: scope for scope, (dialect, _) in DIALECTS.items()}

OPTIONS = ("allow_untested_version",)

# Tested dialects that leave empty tool call arguments out of their spans: dialect -> the rule, named in the
# ``arguments_stand_in`` field of a call whose arguments the reader reconstructed as {}. A dialect not listed
# here has no such convention, so absent arguments stay absent (``arguments_state`` "missing").
OMITS_EMPTY_ARGUMENTS = {
    "openinference/openai-agents": "openinference-instrumentation-openai-agents 2.5.3 does not export tool call "
                                   "arguments equal to '{}'",
}

MAPPING = """\
Pydantic AI 2.58.0 (gen_ai.*): a chat span (gen_ai.operation.name chat) is an agent step. Messages in its
gen_ai.input.messages that no earlier span showed become user and system steps; the history that later chat spans repeat is
not expanded. gen_ai.output.messages: text parts are the message, tool_call parts the tool_calls, reasoning parts the
reasoning_content. An execute_tool span is the result of the call with the same gen_ai.tool.call.id (gen_ai.tool.call.result
verbatim). A tool_call_response part of a later chat input is a result only for a call no execute_tool span answered. An
invoke_agent span is a roster entry (gen_ai.agent.name, gen_ai.system_instructions) and, when nested under a tool, a subagent.
OpenAI Agents SDK through OpenInference: a span of kind LLM is an agent step. New items of llm.input_messages become user and
system steps, an item with role tool and message.tool_call_id is the result of that call. llm.output_messages give the message
and tool_calls. An AGENT span with agent.name is a roster entry and the actor of its steps, graph.node.parent_id gives
handoffs, llm.tools give the tool definitions. TOOL spans carry no call id and stay in spans only.
Tool call arguments. A call with no arguments member stays absent: the ATIF field holds the required {} and the call
extension says arguments_state "missing". One tested dialect has a convention for it: OpenInference (the Agents SDK
instrumentation, 2.5.3) does not export arguments equal to '{}', so there the reader reconstructs {} and the call
extension names the rule in arguments_stand_in. Pydantic AI has no such convention (it also leaves arguments out when
content capture is off), so its absent arguments stay missing. An explicit null stays null (arguments_raw). Arguments that
are not a JSON object (text that does not parse, a list, a number) stay in arguments_raw, ATIF holds {} for them, and the
call extension says arguments_state "unparsed".
"""

# -- AnyValue ----------------------------------------------------------------------

_INT_MIN, _INT_MAX = -(1 << 63), (1 << 63) - 1
_BYTES = "$bytes"  # a bytesValue is held as {"$bytes": base64} so the record stays plain JSON


def encode_value(value: Any) -> Dict[str, Any]:
    """A JSON value as an OTLP ``AnyValue``. Null is the empty AnyValue and a dict is a kvlist."""
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        if _INT_MIN <= value <= _INT_MAX:
            return {"intValue": str(value)}
        return {"stringValue": str(value)}
    if isinstance(value, float):
        if value != value:
            return {"doubleValue": "NaN"}
        if value in (float("inf"), float("-inf")):
            return {"doubleValue": "Infinity" if value > 0 else "-Infinity"}
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [encode_value(v) for v in value]}}
    if isinstance(value, dict):
        if len(value) == 1 and isinstance(value.get(_BYTES), str):
            return {"bytesValue": value[_BYTES]}
        return {"kvlistValue": {"values": [{"key": str(k), "value": encode_value(v)} for k, v in value.items()]}}
    if isinstance(value, (bytes, bytearray)):
        return {"bytesValue": base64.b64encode(bytes(value)).decode("ascii")}
    return {"stringValue": str(value)}


def _field(obj: Dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in obj:
            return obj[n]
    return None


def _has(obj: Dict[str, Any], *names: str) -> bool:
    return any(n in obj for n in names)


def decode_value(av: Any) -> Any:
    """An OTLP ``AnyValue`` as a JSON value. The empty AnyValue is null."""
    if not isinstance(av, dict):
        return av
    if _has(av, "stringValue", "string_value"):
        return _field(av, "stringValue", "string_value")
    if _has(av, "boolValue", "bool_value"):
        return _field(av, "boolValue", "bool_value")
    if _has(av, "intValue", "int_value"):
        raw = _field(av, "intValue", "int_value")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if _has(av, "doubleValue", "double_value"):
        raw = _field(av, "doubleValue", "double_value")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if _has(av, "arrayValue", "array_value"):
        arr = _field(av, "arrayValue", "array_value") or {}
        return [decode_value(v) for v in (arr.get("values") or [])]
    if _has(av, "kvlistValue", "kvlist_value"):
        kv = _field(av, "kvlistValue", "kvlist_value") or {}
        return {str(e.get("key")): decode_value(e.get("value")) for e in (kv.get("values") or []) if isinstance(e, dict)}
    if _has(av, "bytesValue", "bytes_value"):
        return {_BYTES: _field(av, "bytesValue", "bytes_value")}
    return None


def decode_attributes(attrs: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in attrs or []:
        if isinstance(kv, dict) and "key" in kv:
            out[str(kv["key"])] = decode_value(kv.get("value"))
    return out


def encode_attributes(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"key": k, "value": encode_value(v)} for k, v in attrs.items()]


# -- ids, times, kinds, JSON text --------------------------------------------------------

_HEX = re.compile(r"^[0-9a-fA-F]+\Z")


def _hex_id(value: Any, nbytes: int) -> Optional[str]:
    """A hex id. Base64 (what protobuf's own JSON printer emits for bytes) is converted; an odd id stays verbatim."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        s = value.strip()
        if len(s) == nbytes * 2 and _HEX.match(s):
            return s.lower()
        try:
            raw = base64.b64decode(s, validate=True)
        except ValueError:
            return s
        return raw.hex() if len(raw) == nbytes else s
    return str(value)


def _nano(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_KINDS = ("UNSPECIFIED", "INTERNAL", "SERVER", "CLIENT", "PRODUCER", "CONSUMER")


def _kind_name(kind: Any) -> str:
    if kind is None:
        return "UNSPECIFIED"
    if isinstance(kind, int) and not isinstance(kind, bool):
        return _KINDS[kind] if 0 <= kind < len(_KINDS) else str(kind)
    s = str(kind)
    for prefix in ("SPAN_KIND_", "SpanKind."):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.upper() if s.upper() in _KINDS else s


def _kind_int(kind: Any) -> int:
    n = _kind_name(kind)
    return _KINDS.index(n) if n in _KINDS else 0


_STATUS = ("UNSET", "OK", "ERROR")


def _status_obj(status: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(status, dict):
        return None
    code = status.get("code", 0)
    if isinstance(code, int) and not isinstance(code, bool):
        code = _STATUS[code] if 0 <= code < len(_STATUS) else str(code)
    else:
        code = str(code)
        if code.startswith("STATUS_CODE_"):
            code = code[len("STATUS_CODE_"):]
    out: Dict[str, Any] = {"code": code}
    if status.get("message"):
        out["message"] = status["message"]
    return out


def _status_int(status: Any) -> int:
    code = status.get("code") if isinstance(status, dict) else None
    return _STATUS.index(code) if code in _STATUS else 0


_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:[.,](\d{1,9}))?\s*(Z|z|[+-]\d{2}(?::?\d{2})?)?$")


def parse_iso_ns(text: Any) -> Optional[int]:
    """ISO 8601 timestamp -> Unix nanoseconds (no offset means UTC), or None when it does not parse."""
    if not isinstance(text, str):
        return None
    m = _ISO.match(text.strip())
    if not m:
        return None
    y, mo, d, h, mi, s = (int(m.group(i)) for i in range(1, 7))
    try:
        _dt.datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None
    frac = int(m.group(7).ljust(9, "0")) if m.group(7) else 0
    zone = m.group(8)
    offset = 0
    if zone and zone not in ("Z", "z"):
        digits = zone[1:].replace(":", "")
        offset = (-1 if zone[0] == "-" else 1) * (int(digits[:2]) * 3600 + (int(digits[2:4]) * 60 if len(digits) >= 4 else 0))
    return (calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) - offset) * 1_000_000_000 + frac


def iso_from_ns(ns: int) -> Optional[str]:
    """Unix nanoseconds -> ISO 8601 UTC with microseconds (the full value stays in the span entry); None out of range."""
    secs, rem = divmod(int(ns), 1_000_000_000)
    try:
        when = _dt.datetime(1970, 1, 1) + _dt.timedelta(seconds=secs)
    except OverflowError:
        return None
    return when.strftime("%Y-%m-%dT%H:%M:%S") + ".%06dZ" % (rem // 1000)


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _unj(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class _Absent:
    """No value at all, as opposed to null."""

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT = _Absent()


# -- documents ------------------------------------------------------------------------------

def _split_documents(text: str, jsonl: bool) -> Iterator[Any]:
    last: Optional[Exception] = None
    for how in (("lines", "whole") if jsonl else ("whole", "lines")):
        try:
            if how == "whole":
                doc = json.loads(text)
                docs = doc if isinstance(doc, list) else [doc]
            else:
                docs = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError as exc:
            last = exc
            continue
        yield from docs
        return
    raise ValueError(f"not OTLP JSON or JSONL: {last}")


def _documents(source: Any) -> Iterator[Any]:
    if isinstance(source, Path):
        with _open_text(source) as fh:
            text = fh.read()
        yield from _split_documents(text, _is_jsonl(source))
    elif isinstance(source, list):
        yield from source
    elif isinstance(source, str):
        yield from _split_documents(source, False)
    else:
        yield source


def _first_document(path: Path) -> Any:
    """The first JSON document of a file, reading only the first line of a JSONL file when that line parses."""
    try:
        with _open_text(path) as fh:
            text = ""
            if _is_jsonl(path):
                for line in fh:
                    if line.strip():
                        try:
                            return json.loads(line)
                        except ValueError:
                            text = line + fh.read()  # one document spread over several lines
                        break
                else:
                    return None
            else:
                text = fh.read()
        try:
            doc = json.loads(text)
            return doc[0] if isinstance(doc, list) and doc else doc
        except ValueError:
            pass  # JSONL in a file that is not named .jsonl
        line = next((ln for ln in text.splitlines() if ln.strip()), "")
        return json.loads(line) if line else None
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def _looks_like_otlp(doc: Any) -> bool:
    if isinstance(doc, list):
        doc = doc[0] if doc else None
    return isinstance(doc, dict) and isinstance(doc.get("resourceSpans", doc.get("resource_spans")), list)


def sniff(source: Any) -> bool:
    if isinstance(source, Path):
        return source.is_file() and _looks_like_otlp(_first_document(source))
    return _looks_like_otlp(source)


class _Span:
    """One span as read: the entry kept in the record, plus what grouping needs."""

    __slots__ = ("key", "id", "entry", "attrs", "scope_name", "scope_version", "schema_url", "order")

    def __init__(self, key: str, sid: Optional[str], entry: Dict[str, Any], scope_name: str, scope_version: str,
                 schema_url: str, order: int):
        self.key = key
        self.id = sid
        self.entry = entry
        self.attrs: Dict[str, Any] = entry["attributes"]
        self.scope_name = scope_name
        self.scope_version = scope_version
        self.schema_url = schema_url
        self.order = order

    @property
    def parent(self) -> Optional[str]:
        return self.entry.get("parent")

    @property
    def start(self) -> int:
        return self.entry.get("start_time_unix_nano") or 0


def _normalize_span(sp: Dict[str, Any], resource: Dict[str, Any], scope: Dict[str, Any],
                    schema_url: str) -> Tuple[Optional[str], Dict[str, Any]]:
    sid = _hex_id(_field(sp, "spanId", "span_id"), 8)
    events = []
    for ev in sp.get("events") or []:
        if not isinstance(ev, dict):
            continue
        item: Dict[str, Any] = {"name": ev.get("name"), "time_unix_nano": _nano(_field(ev, "timeUnixNano", "time_unix_nano")),
                                "attributes": decode_attributes(ev.get("attributes"))}
        dropped = _field(ev, "droppedAttributesCount", "dropped_attributes_count")
        if dropped:
            item["dropped_attributes_count"] = dropped
        events.append(item)
    links = []
    for lk in sp.get("links") or []:
        if not isinstance(lk, dict):
            continue
        link: Dict[str, Any] = {"trace_id": _hex_id(_field(lk, "traceId", "trace_id"), 16),
                                "span_id": _hex_id(_field(lk, "spanId", "span_id"), 8),
                                "attributes": decode_attributes(lk.get("attributes"))}
        for camel, snake in (("traceState", "trace_state"), ("flags", "flags"),
                             ("droppedAttributesCount", "dropped_attributes_count")):
            val = _field(lk, camel, snake)
            if val:
                link[snake] = val
        links.append(link)
    entry: Dict[str, Any] = {
        "name": sp.get("name"),
        "kind": _kind_name(sp.get("kind")),
        "parent": _hex_id(_field(sp, "parentSpanId", "parent_span_id"), 8),
        "trace_id": _hex_id(_field(sp, "traceId", "trace_id"), 16),
        "start_time_unix_nano": _nano(_field(sp, "startTimeUnixNano", "start_time_unix_nano")),
        "end_time_unix_nano": _nano(_field(sp, "endTimeUnixNano", "end_time_unix_nano")),
        "status": _status_obj(sp.get("status")),
        "attributes": decode_attributes(sp.get("attributes")),
        "events": events,
        "links": links,
        "address": None,
    }
    for camel, snake in (("traceState", "trace_state"), ("flags", "flags"), ("droppedAttributesCount", "dropped_attributes_count"),
                         ("droppedEventsCount", "dropped_events_count"), ("droppedLinksCount", "dropped_links_count")):
        val = _field(sp, camel, snake)
        if val:
            entry[snake] = val
    if resource:
        entry["resource"] = resource
    if scope:
        entry["scope"] = scope
    if schema_url:
        entry["schema_url"] = schema_url
    return sid, entry


def _parse_spans(source: Any) -> List[_Span]:
    spans: List[_Span] = []
    for doc in _documents(source):
        if not isinstance(doc, dict):
            raise ValueError(f"an OTLP document is a JSON object, got {type(doc).__name__}")
        for rs in _field(doc, "resourceSpans", "resource_spans") or []:
            if not isinstance(rs, dict):
                continue
            resource = decode_attributes((rs.get("resource") or {}).get("attributes"))
            rs_schema = _field(rs, "schemaUrl", "schema_url") or ""
            blocks = _field(rs, "scopeSpans", "scope_spans", "instrumentationLibrarySpans", "instrumentation_library_spans") or []
            for ss in blocks:
                if not isinstance(ss, dict):
                    continue
                sc = _field(ss, "scope", "instrumentationLibrary", "instrumentation_library") or {}
                scope: Dict[str, Any] = {k: sc[k] for k in ("name", "version") if sc.get(k)}
                if sc.get("attributes"):
                    scope["attributes"] = decode_attributes(sc["attributes"])
                schema_url = _field(ss, "schemaUrl", "schema_url") or rs_schema
                for sp in ss.get("spans") or []:
                    if not isinstance(sp, dict):
                        continue
                    sid, entry = _normalize_span(sp, resource, scope, schema_url)
                    order = len(spans)
                    spans.append(_Span(sid or f"span-{order}", sid, entry, scope.get("name", ""), scope.get("version", ""),
                                       schema_url, order))
    return spans


def read(source: Any, **options: Any) -> Iterator[Run]:
    """One ``Run`` per trace id, in the order the trace ids first appear.

    ``allow_untested_version=True`` reads a framework dialect whose instrumentation scope version this release
    was not tested with; without it such a trace raises ``UnsupportedVersion``, since a changed span layout
    would give plausible but wrong steps. The run is marked ``source.dialect_tested`` false.
    """
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise TypeError(f"otlp-json reader takes no options except {', '.join(OPTIONS)}, got {unknown}")
    allow_untested = bool(options.get("allow_untested_version", False))
    traces: Dict[str, List[_Span]] = {}
    for sp in _parse_spans(source):
        traces.setdefault(sp.entry.get("trace_id") or "", []).append(sp)
    for trace_id, spans in traces.items():
        seen: Dict[str, _Span] = {}  # a repeated span id keeps both spans under distinct keys
        for s in spans:
            s.key = _unique_key(seen, s.key)
            seen[s.key] = s
        if any(isinstance(s.attrs.get("agentpile.version"), str) and s.attrs.get("agentpile.kind") == "trajectory" for s in spans):
            yield from _read_agentpile(spans)
        else:
            yield _read_foreign(trace_id, spans, allow_untested)


def _unique_key(existing: Dict[str, Any], key: str) -> str:
    if key not in existing:
        return key
    n = 2
    while f"{key}#{n}" in existing:
        n += 1
    return f"{key}#{n}"


# =============================================================================
# Writer: Run -> OTLP JSON
# =============================================================================

SYNTHETIC_EPOCH_NS = 946_684_800 * 1_000_000_000  # 2000-01-01T00:00:00Z, the base of synthetic times
_TICK = 1_000_000  # one millisecond between the synthetic times of two steps
_CALL_TICK = 1000  # one microsecond between a step and its tool call spans


def _text_of(content: Any) -> str:
    """Display text of an ATIF message or result: text parts joined, media parts as a short reference."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("image", "audio"):
                    src = part.get("source") if isinstance(part.get("source"), dict) else {}
                    out.append("[%s: %s]" % (part["type"], src.get("path", src.get("url", "unknown"))))
                else:
                    out.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return str(content)


def _split_extra(obj: Dict[str, Any]) -> Tuple[Any, Any]:
    """(extra without its agentpile key, or ABSENT; the agentpile extension, or ABSENT) of an object with ``extra``."""
    if "extra" not in obj:
        return ABSENT, ABSENT
    extra = obj["extra"]
    if isinstance(extra, dict) and isinstance(extra.get(EXT), dict):
        rest = {k: v for k, v in extra.items() if k != EXT}
        return (rest if rest else ABSENT), extra[EXT]
    return extra, ABSENT


def _put_object(attrs: Dict[str, Any], prefix: str, obj: Dict[str, Any], skip: Tuple[str, ...] = ()) -> int:
    n = 0
    for k, v in obj.items():
        if k not in skip:
            attrs[prefix + "." + k] = _j(v)
            n += 1
    return n


def _put_ext(attrs: Dict[str, Any], ext: Any, skip: Tuple[str, ...] = ()) -> None:
    """Extension fields as ``agentpile.ext.<key>``. ``agentpile.ext`` alone marks an extension that has no such key."""
    if ext is ABSENT:
        return
    if _put_object(attrs, "agentpile.ext", ext, skip) == 0:
        attrs["agentpile.ext"] = "{}"


def _traj_id(traj: Dict[str, Any]) -> Optional[str]:
    return traj.get("trajectory_id") or traj.get("session_id")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _step_times(steps: List[Dict[str, Any]]) -> List[Tuple[int, bool]]:
    """(start ns, synthetic) per step: the source timestamp where it parses, an increasing stand-in elsewhere."""
    parsed = [parse_iso_ns(s.get("timestamp")) for s in steps]
    real = [p for p in parsed if p is not None]
    lead = 0
    for p in parsed:
        if p is not None:
            break
        lead += 1
    base = (real[0] - lead * _TICK) if real else SYNTHETIC_EPOCH_NS
    out: List[Tuple[int, bool]] = []
    prev: Optional[int] = None
    for p in parsed:
        if p is not None:
            out.append((p, False))
            prev = p
        else:
            prev = base if prev is None else prev + _TICK
            out.append((prev, True))
    return out


def _genai_tool_definition(defn: Any) -> Optional[Dict[str, Any]]:
    """An OpenAI function schema as a GenAI v1.41.0 function tool definition, or None for any other shape."""
    if isinstance(defn, dict) and isinstance(defn.get("function"), dict) and isinstance(defn["function"].get("name"), str):
        fn = defn["function"]
        out: Dict[str, Any] = {"type": "function", "name": fn["name"]}
        if fn.get("description") is not None:
            out["description"] = fn["description"]
        if fn.get("parameters") is not None:
            out["parameters"] = fn["parameters"]
        return out
    return None


def _subs_of(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The embedded subagent trajectories the writer nests as spans; any other shape is written as one attribute."""
    value = traj.get("subagent_trajectories")
    return value if isinstance(value, list) and value and all(isinstance(s, dict) for s in value) else []


def _call_extension(step_ext: Any, call_id: Any) -> Dict[str, Any]:
    calls = step_ext.get("calls") if isinstance(step_ext, dict) else None
    cx = calls.get(call_id) if isinstance(calls, dict) and isinstance(call_id, str) else None
    return cx if isinstance(cx, dict) else {}


def _shown_arguments(call: Dict[str, Any], cx: Dict[str, Any], prefer_raw: bool) -> Any:
    """The arguments a backend is shown: ABSENT when the source recorded none (or the reader rebuilt {} for it).

    A tool span (``prefer_raw``) shows the source value whenever the extension kept one. A model span shows the
    parsed arguments, except for a call whose arguments the reader marked ``arguments_state`` "unparsed": ATIF
    holds a stand-in {} there, so the span shows the source value like the tool span does.
    """
    state = cx.get("arguments_state")
    if state == "missing" or "arguments_stand_in" in cx:
        return ABSENT
    if "arguments_raw" in cx and cx["arguments_raw"] is None:  # an explicit null in the source
        return None
    if "arguments_raw" in cx and (prefer_raw or state == "unparsed"):
        return cx["arguments_raw"]
    return call.get("arguments")


class _Claim:
    """A kept source span that a step, a call or a subagent trajectory came from, with the id it is written under."""

    __slots__ = ("owner", "key", "held", "entry", "n", "sid")

    def __init__(self, owner: str, key: str, held: Any, n: int, sid: str):
        self.owner = owner  # path of the trajectory whose ``spans`` field holds the entry
        self.key = key
        self.held = held  # what the field held, which a hand-edited record may make something other than an object
        self.entry = held if isinstance(held, dict) else {}
        self.n = n  # position in that field
        self.sid = sid


class _Writer:
    def __init__(self, record: Dict[str, Any], trace_id: Optional[str] = None):
        self.rec = record
        self.trace = self._pick_trace(record, trace_id)
        self.spans: List[Dict[str, Any]] = []
        self.used: set = set()
        self.traj_span: Dict[Any, str] = {}
        self.step_span: Dict[Tuple[Any, Any], str] = {}
        self.call_span: Dict[Tuple[Any, Any, Any], str] = {}
        self.result_calls: Dict[Tuple[Any, Any], List[Any]] = {}
        self.kept_span: Dict[str, str] = {}
        self.kept_id: Dict[Tuple[str, str], str] = {}
        self.todo_links: List[Tuple[Dict[str, Any], Any, List[Any], str, str]] = []
        # plan: which kept spans the record's steps, calls and subagents came from
        self.sourced = False  # the record carries source spans, so the source's operations are written, not new ones
        self.kept: Dict[str, Dict[str, Any]] = {}  # trajectory path -> its ``spans`` field
        self.span_ids: Dict[str, Dict[str, str]] = {}  # trajectory path -> kept span key -> written span id
        self.root_ids: Dict[str, str] = {}
        self.claim_step: Dict[Tuple[str, int], _Claim] = {}
        self.claim_call: Dict[Tuple[str, int, int], _Claim] = {}
        self.claim_traj: Dict[str, _Claim] = {}
        self.claimed: set = set()

    @staticmethod
    def _pick_trace(record: Dict[str, Any], given: Optional[str]) -> str:
        if given:
            if len(given) != 32 or not _HEX.match(given):
                raise ValueError("trace_id must be 32 hex characters")
            return given.lower()
        tid = _traj_id(record)
        if isinstance(tid, str) and len(tid) == 32 and _HEX.match(tid) and set(tid) != {"0"}:
            return tid.lower()
        seed = tid if isinstance(tid, str) else json.dumps(record, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(("agentpile:" + seed).encode("utf-8")).digest()[:16].hex()

    def _sid(self, path: str) -> str:
        n = 0
        while True:
            h = hashlib.sha256(f"{self.trace}:{path}:{n}".encode("utf-8")).hexdigest()[:16]
            if h not in self.used and h != "0" * 16:
                self.used.add(h)
                return h
            n += 1

    def build(self) -> List[Dict[str, Any]]:
        self._plan()
        self._emit_traj(self.rec, "0", "trajectory", 0, None, None)
        for span, tid, entries, own, root_id in self.todo_links:
            span["links"] = list(span.get("links", [])) + [self._link(e, tid, own, root_id) for e in entries]
        return self.spans

    # -- plan: ids of kept spans, and the kept spans that are the source of a step, a call or a subagent ------------

    def _plan(self) -> None:
        trajs: List[Tuple[str, Dict[str, Any]]] = []

        def walk(traj: Dict[str, Any], tpath: str) -> None:
            trajs.append((tpath, traj))
            for k, sub in enumerate(_subs_of(traj)):
                walk(sub, f"{tpath}/sub{k}")

        walk(self.rec, "0")
        holders: List[Tuple[str, Dict[str, Any]]] = []
        for tpath, traj in trajs:
            ext = _split_extra(traj)[1]
            kept = ext.get("spans") if isinstance(ext, dict) else None
            if isinstance(kept, dict) and kept:
                holders.append((tpath, kept))
                self.kept[tpath] = kept
        self.sourced = bool(holders)
        # A kept span keeps its own id when that is a valid unused span id, so ids in labels and links still match.
        for tpath, kept in holders:
            for key in kept:
                if isinstance(key, str) and len(key) == 16 and _HEX.match(key) and key == key.lower() \
                        and key != "0" * 16 and key not in self.used:
                    self.used.add(key)
                    self.kept_id[(tpath, key)] = key
        for tpath, kept in holders:
            ids: Dict[str, str] = {}
            for n, key in enumerate(kept):
                ids[key] = self.kept_id.get((tpath, key)) or self._sid(f"{tpath}/kept/{n}")
                self.kept_span.setdefault(key, ids[key])
            self.span_ids[tpath] = ids
        self._plan_claims(trajs, holders)
        for tpath, _ in trajs:
            claim = self.claim_traj.get(tpath)
            self.root_ids[tpath] = claim.sid if claim is not None else self._sid("traj:" + tpath)

    def _plan_claims(self, trajs: List[Tuple[str, Dict[str, Any]]], holders: List[Tuple[str, Dict[str, Any]]]) -> None:
        """A kept span is the source of a step or call when its address names it, and of a subagent trajectory when
        the trajectory id is the span's key. Only an unambiguous match counts: an id shared by two trajectories,
        steps or calls claims nothing, and a span or a step is claimed at most once."""
        if not holders:
            return
        tid_path: Dict[str, Optional[str]] = {}
        step_at: Dict[Tuple[str, int], Dict[str, Any]] = {}
        step_pos: Dict[str, Dict[Any, Optional[int]]] = {}
        call_pos: Dict[Tuple[str, int], Dict[str, Optional[int]]] = {}
        for tpath, traj in trajs:
            tid = _traj_id(traj)
            if isinstance(tid, str):
                tid_path[tid] = tpath if tid not in tid_path else None
            steps = traj.get("steps")
            positions: Dict[Any, Optional[int]] = {}
            for i, st in enumerate(steps if isinstance(steps, list) else []):
                if not isinstance(st, dict):
                    continue
                step_at[(tpath, i)] = st
                sid = st.get("step_id")
                if isinstance(sid, str) or _is_int(sid):
                    positions[sid] = i if sid not in positions else None
                calls = st.get("tool_calls")
                if isinstance(calls, list) and calls and all(isinstance(c, dict) for c in calls):
                    where: Dict[str, Optional[int]] = {}
                    for j, c in enumerate(calls):
                        cid = c.get("tool_call_id")
                        if isinstance(cid, str):
                            where[cid] = j if cid not in where else None
                    call_pos[(tpath, i)] = where
            step_pos[tpath] = positions
        candidates: Dict[Tuple[str, int, int], List[_Claim]] = {}
        for holder, kept in holders:
            for n, (key, entry) in enumerate(kept.items()):
                if not isinstance(key, str) or not isinstance(entry, dict):
                    continue
                claim = _Claim(holder, key, entry, n, self.span_ids[holder][key])
                addr = entry.get("address")
                if addr is None:
                    sub_path = tid_path.get(key)
                    if sub_path is not None and sub_path != "0" and sub_path not in self.claim_traj:
                        self.claim_traj[sub_path] = claim
                        self.claimed.add((holder, key))
                    continue
                if not isinstance(addr, dict) or "step" not in addr or {"result", "span", "agent", "tool"} & set(addr):
                    continue
                where_t = addr.get("trajectory")
                target = holder if where_t is None else (tid_path.get(where_t) if isinstance(where_t, str) else None)
                want = addr["step"]
                if target is None or not (isinstance(want, str) or _is_int(want)):
                    continue
                i = step_pos.get(target, {}).get(want)
                if i is None:
                    continue
                if "call" in addr:
                    call = addr["call"]
                    j = call_pos.get((target, i), {}).get(call) if isinstance(call, str) else None
                    if j is not None:
                        candidates.setdefault((target, i, j), []).append(claim)
                elif step_at[(target, i)].get("source") == "agent" and (target, i) not in self.claim_step:
                    self.claim_step[(target, i)] = claim
                    self.claimed.add((holder, key))
        for (target, i, j), group in candidates.items():
            step = step_at[(target, i)]
            cid = step["tool_calls"][j]["tool_call_id"]
            named = _call_extension(_split_extra(step)[1], cid).get("span_id")
            pick = next((c for c in group if c.key == named), group[0])  # the span the reader paired with the call
            self.claim_call[(target, i, j)] = pick
            self.claimed.add((pick.owner, pick.key))

    # -- a span that is a kept source span --------------------------------------------------------------------

    def _adopt(self, claim: _Claim, fallback_parent: Optional[str], fallback_start: int) -> Tuple[Dict[str, Any], bool]:
        """The span of a kept entry without its attributes, and whether its times had to be made up."""
        entry = claim.entry
        start, end = entry.get("start_time_unix_nano"), entry.get("end_time_unix_nano")
        synthetic = not _is_int(start) or not _is_int(end)
        start = start if _is_int(start) else fallback_start
        end = end if _is_int(end) else start
        parent = entry.get("parent")
        ids = self.span_ids[claim.owner]
        pid = ids[parent] if isinstance(parent, str) and parent != claim.key and parent in ids else fallback_parent
        span: Dict[str, Any] = {"traceId": self.trace, "spanId": claim.sid}
        if pid:
            span["parentSpanId"] = pid
        span.update({"name": entry.get("name") if isinstance(entry.get("name"), str) else "",
                     "kind": _kind_int(entry.get("kind")),
                     "startTimeUnixNano": str(start), "endTimeUnixNano": str(end)})
        status = entry.get("status")
        if isinstance(status, dict) and _status_int(status):
            span["status"] = {"code": _status_int(status), **({"message": status["message"]} if status.get("message") else {})}
        events = []
        for ev in entry.get("events") if isinstance(entry.get("events"), list) else []:
            if isinstance(ev, dict):
                item: Dict[str, Any] = {"name": ev.get("name") if isinstance(ev.get("name"), str) else "",
                                        "attributes": encode_attributes(ev["attributes"]) if isinstance(ev.get("attributes"), dict) else []}
                if _is_int(ev.get("time_unix_nano")):
                    item["timeUnixNano"] = str(ev["time_unix_nano"])
                events.append(item)
        if events:
            span["events"] = events
        links = []
        for lk in entry.get("links") if isinstance(entry.get("links"), list) else []:
            if isinstance(lk, dict) and isinstance(lk.get("trace_id"), str) and isinstance(lk.get("span_id"), str) \
                    and len(lk["trace_id"]) == 32 and len(lk["span_id"]) == 16:
                links.append({"traceId": lk["trace_id"], "spanId": lk["span_id"],
                              "attributes": encode_attributes(lk["attributes"]) if isinstance(lk.get("attributes"), dict) else []})
        if links:
            span["links"] = links
        return span, synthetic

    def _adopted_attrs(self, claim: _Claim, record: Dict[str, Any], synthetic: bool, owner: Optional[str]) -> Dict[str, Any]:
        """The record attributes first, then the source span's own attributes unchanged, then what ties the two together."""
        attrs = dict(record)
        original = claim.entry.get("attributes") if isinstance(claim.entry.get("attributes"), dict) else {}
        for k, v in original.items():
            if not str(k).startswith("agentpile."):
                attrs[k] = v
        attrs["agentpile.span_key"] = claim.key
        attrs["agentpile.span_index"] = claim.n
        attrs["agentpile.span_owner"] = self.root_ids[claim.owner]
        if owner is not None:
            attrs["agentpile.owner"] = owner  # the trajectory span this step or call belongs to
        if synthetic:
            attrs["agentpile.time_synthetic"] = True
        attrs["agentpile.span"] = _j(claim.held)
        return attrs

    # -- one trajectory ------------------------------------------------------------------

    def _emit_traj(self, traj: Dict[str, Any], tpath: str, kind: str, index: int, parent_id: Optional[str],
                   owner_id: Optional[str]) -> str:
        steps_raw = traj.get("steps")
        steps = []
        for s in steps_raw if isinstance(steps_raw, list) else []:
            if not isinstance(s, dict):
                raise ValueError(f"steps must be objects, got {type(s).__name__}")
            steps.append(s)
        tid = _traj_id(traj)
        root_id = self.root_ids[tpath]
        first_span = len(self.spans)
        root_claim = self.claim_traj.get(tpath)
        extra_attr, ext = _split_extra(traj)
        agent = traj.get("agent") if isinstance(traj.get("agent"), dict) else {}
        roster = {a["id"]: (a.get("name") if isinstance(a.get("name"), str) else a["id"])
                  for a in (ext.get("agents") if isinstance(ext, dict) and isinstance(ext.get("agents"), list) else [])
                  if isinstance(a, dict) and isinstance(a.get("id"), str)}

        attrs: Dict[str, Any] = {"agentpile.version": EXT_VERSION, "agentpile.kind": kind, "agentpile.index": index}
        if owner_id is not None and self.sourced:  # a source span's own parent chain need not lead to the embedding trajectory
            attrs["agentpile.owner"] = owner_id  # the trajectory span this subagent trajectory is embedded in
        if root_claim is None:
            attrs.update(self._root_view(traj, agent, steps, container=self.sourced))
        _put_object(attrs, "agentpile.trajectory", traj, ("agent", "steps", "subagent_trajectories", "extra"))
        _put_object(attrs, "agentpile.agent", agent)
        if extra_attr is not ABSENT:
            attrs["agentpile.trajectory.extra"] = _j(extra_attr)
        subs: List[Dict[str, Any]] = []
        if "subagent_trajectories" in traj:
            subs = _subs_of(traj)
            if not subs:
                attrs["agentpile.trajectory.subagent_trajectories"] = _j(traj["subagent_trajectories"])
        kept = self.kept.get(tpath)
        _put_ext(attrs, ext, ("spans",) if kept else ())

        times = _step_times(steps)
        root_synthetic = False
        if root_claim is not None:  # the source span of this subagent is its span
            root, root_synthetic = self._adopt(root_claim, parent_id, times[0][0] if times else SYNTHETIC_EPOCH_NS)
        else:
            label = str(agent.get("name", ""))
            root = {"traceId": self.trace, "spanId": root_id,
                    "name": ("trajectory " if self.sourced else "invoke_agent ") + label, "kind": 1}
            if parent_id:
                root["parentSpanId"] = parent_id
        self.spans.append(root)
        self.traj_span.setdefault(tid, root_id)

        step_ids = []
        for i in range(len(steps)):
            claim = self.claim_step.get((tpath, i))
            step_ids.append(claim.sid if claim is not None else self._sid(f"{tpath}/step/{i}"))
        call_ids: Dict[Tuple[int, Any], str] = {}
        end_max = times[0][0] if times else SYNTHETIC_EPOCH_NS
        for i, step in enumerate(steps):
            self.step_span.setdefault((tid, step.get("step_id")), step_ids[i])
            obs = step.get("observation")
            results = (obs.get("results") if isinstance(obs, dict) and isinstance(obs.get("results"), list) else [])
            self.result_calls[(tid, step.get("step_id"))] = [r.get("source_call_id") if isinstance(r, dict) else None for r in results]
            start, synth = times[i]
            step_claim = self.claim_step.get((tpath, i))
            if step_claim is not None and _is_int(step_claim.entry.get("start_time_unix_nano")):
                start = step_claim.entry["start_time_unix_nano"]  # the source span's own time
            end = max(start, times[i + 1][0] if i + 1 < len(times) else start)
            calls = step.get("tool_calls")
            calls = calls if isinstance(calls, list) and calls and all(isinstance(c, dict) for c in calls) else []
            sx = _split_extra(step)[1]
            cxs = sx.get("calls") if isinstance(sx, dict) and isinstance(sx.get("calls"), dict) else {}
            for j, call in enumerate(calls):
                call_claim = self.claim_call.get((tpath, i, j))
                csid = call_claim.sid if call_claim is not None else self._sid(f"{tpath}/step/{i}/call/{j}")
                cid = call.get("tool_call_id")
                call_ids[(i, cid)] = csid
                self.call_span.setdefault((tid, step.get("step_id"), cid), csid)
                t = start + (j + 1) * _CALL_TICK
                end = max(end, t)
                self.spans.append(self._call_span(call, j, i, csid, root_id, t, cxs, results, call_claim))
            end_max = max(end_max, end)
            self.spans.append(self._step_span(step, i, step_ids[i], root_id, start, end, synth, tid, roster, agent, step_claim))
        for k, sub in enumerate(subs):  # under the call that references the subagent, else its step, else the root
            self._emit_traj(sub, f"{tpath}/sub{k}", "subagent", k, self._subagent_parent(sub, steps, step_ids, call_ids, root_id),
                            root_id)
        if kept:
            self._emit_kept(tpath, root_id, times[0][0] if times else SYNTHETIC_EPOCH_NS)
        if any(s for _, s in times):
            attrs["agentpile.time_synthetic"] = True
        if root_claim is not None:
            root["attributes"] = encode_attributes(self._adopted_attrs(root_claim, attrs, root_synthetic, None))
            return root_id
        starts, ends = [times[0][0] if times else SYNTHETIC_EPOCH_NS], [end_max]
        if self.sourced:  # the source spans are not on the step clock, and the trajectory span covers all of them
            for sp in self.spans[first_span + 1:]:
                for field, bucket in (("startTimeUnixNano", starts), ("endTimeUnixNano", ends)):
                    if isinstance(sp.get(field), str) and sp[field].isdigit():
                        bucket.append(int(sp[field]))
        root["startTimeUnixNano"] = str(min(starts))
        root["endTimeUnixNano"] = str(max(ends))
        root["attributes"] = encode_attributes(attrs)
        return root_id

    @staticmethod
    def _subagent_parent(sub: Dict[str, Any], steps: List[Dict[str, Any]], step_ids: List[str],
                         call_ids: Dict[Tuple[int, Any], str], root_id: str) -> str:
        sub_id = sub.get("trajectory_id")
        if sub_id is None:
            return root_id
        for i, step in enumerate(steps):
            obs = step.get("observation")
            for res in (obs.get("results") if isinstance(obs, dict) and isinstance(obs.get("results"), list) else []):
                refs = res.get("subagent_trajectory_ref") if isinstance(res, dict) else None
                for ref in refs if isinstance(refs, list) else []:
                    if isinstance(ref, dict) and ref.get("trajectory_id") == sub_id:
                        return call_ids.get((i, res.get("source_call_id"))) or step_ids[i]
        return root_id

    # -- views for observability backends -----------------------------------------------------

    @staticmethod
    def _root_view(traj: Dict[str, Any], agent: Dict[str, Any], steps: List[Dict[str, Any]], container: bool) -> Dict[str, Any]:
        if container:
            # The record carries the source's own spans, which hold the agent, model and token attributes. This span only
            # holds them together, so it is neither an agent operation nor a source of token counts.
            v: Dict[str, Any] = {"openinference.span.kind": "CHAIN"}
            if isinstance(traj.get("session_id"), str):
                v["session.id"] = traj["session_id"]
                v["gen_ai.conversation.id"] = traj["session_id"]
            if isinstance(traj.get("trajectory_id"), str):
                v["trajectory.id"] = traj["trajectory_id"]
            return v
        v = {"openinference.span.kind": "AGENT", "gen_ai.operation.name": "invoke_agent"}
        if isinstance(agent.get("name"), str):
            v["gen_ai.agent.name"] = agent["name"]
            v["agent.name"] = agent["name"]
        if isinstance(agent.get("version"), str):
            v["gen_ai.agent.version"] = agent["version"]
        if isinstance(agent.get("model_name"), str):
            v["gen_ai.request.model"] = agent["model_name"]
            v["llm.model_name"] = agent["model_name"]
        defs = agent.get("tool_definitions")
        if isinstance(defs, list) and defs:
            converted = [_genai_tool_definition(d) for d in defs]
            if all(c is not None for c in converted):
                v["gen_ai.tool.definitions"] = _j(converted)
        if isinstance(traj.get("session_id"), str):
            v["session.id"] = traj["session_id"]
            v["gen_ai.conversation.id"] = traj["session_id"]
        if isinstance(traj.get("trajectory_id"), str):
            v["trajectory.id"] = traj["trajectory_id"]
        fm = traj.get("final_metrics") if isinstance(traj.get("final_metrics"), dict) else {}
        for src, dsts in (("total_prompt_tokens", ("llm.token_count.prompt", "gen_ai.usage.input_tokens")),
                          ("total_completion_tokens", ("llm.token_count.completion", "gen_ai.usage.output_tokens")),
                          ("total_cached_tokens", ("llm.token_count.prompt_details.cache_read", "gen_ai.usage.cache_read.input_tokens"))):
            if _is_int(fm.get(src)):
                for d in dsts:
                    v[d] = fm[src]
        if _is_int(fm.get("total_prompt_tokens")) and _is_int(fm.get("total_completion_tokens")):
            v["llm.token_count.total"] = fm["total_prompt_tokens"] + fm["total_completion_tokens"]
        if _is_num(fm.get("total_cost_usd")):
            v["llm.cost.total"] = fm["total_cost_usd"]
        first_user = next((s for s in steps if s.get("source") == "user"), None)
        last_agent = next((s for s in reversed(steps) if s.get("source") == "agent"), None)
        if first_user is not None:
            v["input.value"] = _text_of(first_user.get("message"))
        if last_agent is not None:
            v["output.value"] = _text_of(last_agent.get("message"))
        return v

    def _step_view(self, step: Dict[str, Any], sx_d: Dict[str, Any], roster: Dict[str, str], agent: Dict[str, Any],
                   text: str, model_call: bool) -> Tuple[str, int, Dict[str, Any]]:
        """(span name, span kind, view attributes) of a step no source span stands behind.

        An agent step is a model call (``chat``) only in a record without source spans: where the source's own spans
        are written, a step without one is not a model call (an earlier call's output re-sent as history), so it is an
        ``agent_step`` span that carries the record and no model, token or message attributes."""
        source = step.get("source")
        v: Dict[str, Any] = {}
        if source != "agent":
            v["openinference.span.kind"] = "CHAIN"
            v["input.value"] = text
            v["input.mime_type"] = "text/plain"
            return {"user": "user_message", "system": "system_message"}.get(source, "step"), 1, v
        model = step.get("model_name")
        v["openinference.span.kind"] = "LLM" if model_call else "CHAIN"
        if model_call:
            v["gen_ai.operation.name"] = "chat"
        actor = sx_d.get("actor")
        who = roster.get(actor) if isinstance(actor, str) else None
        who = who or (agent.get("name") if isinstance(agent.get("name"), str) else None)
        if who:
            v["gen_ai.agent.name"] = who
        if not model_call:
            v["output.value"] = text
            v["output.mime_type"] = "text/plain"
            return "agent_step", 1, v
        if isinstance(model, str):
            v["gen_ai.request.model"] = model
            v["llm.model_name"] = model
        m = step.get("metrics") if isinstance(step.get("metrics"), dict) else {}
        for src, dsts in (("prompt_tokens", ("llm.token_count.prompt", "gen_ai.usage.input_tokens")),
                          ("completion_tokens", ("llm.token_count.completion", "gen_ai.usage.output_tokens")),
                          ("cached_tokens", ("llm.token_count.prompt_details.cache_read", "gen_ai.usage.cache_read.input_tokens"))):
            if _is_int(m.get(src)):
                for d in dsts:
                    v[d] = m[src]
        if _is_int(m.get("prompt_tokens")) and _is_int(m.get("completion_tokens")):
            v["llm.token_count.total"] = m["prompt_tokens"] + m["completion_tokens"]
        if _is_num(m.get("cost_usd")):
            v["llm.cost.total"] = m["cost_usd"]
        v["output.value"] = text
        v["output.mime_type"] = "text/plain"
        parts: List[Dict[str, Any]] = [{"type": "text", "content": text}] if text else []
        calls = [c for c in (step.get("tool_calls") if isinstance(step.get("tool_calls"), list) else []) if isinstance(c, dict)]
        shown = [_shown_arguments(c, _call_extension(sx_d, c.get("tool_call_id")), False) for c in calls]
        for c, args in zip(calls, shown):
            part = {"type": "tool_call", "id": c.get("tool_call_id"), "name": c.get("function_name")}
            if args is not ABSENT:
                part["arguments"] = args
            parts.append(part)
        # GenAI requires finish_reason on an output message and ATIF has no such field, so it is omitted (Pydantic AI does the same).
        v["gen_ai.output.messages"] = _j([{"role": "assistant", "parts": parts}])
        v["llm.output_messages.0.message.role"] = "assistant"
        if text:
            v["llm.output_messages.0.message.content"] = text
        for j, (c, args) in enumerate(zip(calls, shown)):
            base = "llm.output_messages.0.message.tool_calls.%d.tool_call." % j
            v[base + "id"] = c.get("tool_call_id")
            v[base + "function.name"] = c.get("function_name")
            if args is not ABSENT:  # OpenInference holds the arguments as text, so a source string goes in as it is
                v[base + "function.arguments"] = args if isinstance(args, str) else _j(args)
        return ("chat " + model if isinstance(model, str) and model else "chat"), 3, v

    def _step_span(self, step: Dict[str, Any], i: int, sid: str, root_id: str, start: int, end: int, synth: bool,
                   tid: Any, roster: Dict[str, str], agent: Dict[str, Any], claim: Optional[_Claim]) -> Dict[str, Any]:
        extra_attr, sx = _split_extra(step)
        sx_d = sx if isinstance(sx, dict) else {}
        attrs: Dict[str, Any] = {"agentpile.version": EXT_VERSION, "agentpile.kind": "step", "agentpile.index": i}
        if claim is None:
            model_call = step.get("source") == "agent" and not self.sourced and step.get("llm_call_count") != 0
            span_name, span_kind, view = self._step_view(step, sx_d, roster, agent, _text_of(step.get("message")), model_call)
            attrs.update(view)
            if synth:
                attrs["agentpile.time_synthetic"] = True
        # the record
        _put_object(attrs, "agentpile.step", step, ("tool_calls", "observation", "extra"))
        if "observation" in step:
            attrs["agentpile.step.observation"] = _j(step["observation"])
        if "tool_calls" in step:
            tc = step["tool_calls"]
            if not (isinstance(tc, list) and tc and all(isinstance(c, dict) for c in tc)):
                attrs["agentpile.step.tool_calls"] = _j(tc)
        if extra_attr is not ABSENT:
            attrs["agentpile.step.extra"] = _j(extra_attr)
        skip: List[str] = []
        deps_list: List[Any] = []
        if isinstance(sx, dict):
            if "calls" in sx:
                skip.append("calls")
                calls_d = sx["calls"]
                ids = {c.get("tool_call_id") for c in (step.get("tool_calls") if isinstance(step.get("tool_calls"), list) else [])
                       if isinstance(c, dict)}
                leftover = {k: v for k, v in calls_d.items() if k not in ids} if isinstance(calls_d, dict) else calls_d
                if not isinstance(calls_d, dict) or leftover or not calls_d:
                    attrs["agentpile.ext.calls"] = _j(leftover)  # entries no call span can carry, or an empty/odd value
            if "deps" in sx:
                skip.append("deps")
                if isinstance(sx["deps"], list) and sx["deps"]:
                    deps_list = sx["deps"]
                else:
                    attrs["agentpile.ext.deps"] = _j(sx["deps"])
        _put_ext(attrs, sx, tuple(skip))
        if claim is not None:  # the source span is this step's span
            span, span_synth = self._adopt(claim, root_id, start)
            span["attributes"] = encode_attributes(self._adopted_attrs(claim, attrs, span_synth, root_id))
        else:
            span = {"traceId": self.trace, "spanId": sid, "parentSpanId": root_id, "name": span_name,
                    "kind": span_kind, "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
                    "attributes": encode_attributes(attrs)}
        if deps_list:
            self.todo_links.append((span, tid, deps_list, sid, root_id))
        return span

    def _call_span(self, call: Dict[str, Any], j: int, i: int, csid: str, root_id: str, t: int,
                   cxs: Dict[str, Any], results: List[Any], claim: Optional[_Claim]) -> Dict[str, Any]:
        cid = call.get("tool_call_id")
        fname = call.get("function_name")
        cx = cxs.get(cid) if isinstance(cid, str) else None
        cx_d = cx if isinstance(cx, dict) else {}
        head: Dict[str, Any] = {"agentpile.version": EXT_VERSION, "agentpile.kind": "call", "agentpile.index": j,
                                "agentpile.step_index": i}
        record: Dict[str, Any] = {}
        _put_object(record, "agentpile.call", call, ("extra",))
        if "extra" in call:
            record["agentpile.call.extra"] = _j(call["extra"])
        if isinstance(cx, dict):
            _put_ext(record, cx)
        if claim is not None:  # the source span is this call's span: its own attributes stay as they are
            span, span_synth = self._adopt(claim, root_id, t)
            span["attributes"] = encode_attributes(self._adopted_attrs(claim, dict(head, **record), span_synth, root_id))
            return span
        attrs: Dict[str, Any] = dict(head, **{"agentpile.time_synthetic": True})
        # A call is a tool operation (``execute_tool``) in a record without source spans. Where the source's own spans are
        # written, no span of the source ran this call, so the span only carries the call record.
        operation = not self.sourced
        if operation:
            attrs["openinference.span.kind"] = "TOOL"
            attrs["gen_ai.operation.name"] = "execute_tool"
            if isinstance(fname, str):
                attrs["gen_ai.tool.name"] = fname
                attrs["tool.name"] = fname
            if isinstance(cid, str):
                attrs["gen_ai.tool.call.id"] = cid
            shown = _shown_arguments(call, cx_d, True)
            if shown is not ABSENT:
                arg_text = shown if isinstance(shown, str) else _j(shown)
                attrs["gen_ai.tool.call.arguments"] = arg_text
                attrs["input.value"] = arg_text
                attrs["input.mime_type"] = "text/plain" if isinstance(shown, str) else "application/json"
            for res in results:
                if isinstance(res, dict) and isinstance(cid, str) and res.get("source_call_id") == cid:
                    attrs["gen_ai.tool.call.result"] = _text_of(res.get("content"))
                    attrs["output.value"] = attrs["gen_ai.tool.call.result"]
                    break
        else:
            attrs["openinference.span.kind"] = "CHAIN"
        attrs.update(record)
        span: Dict[str, Any] = {"traceId": self.trace, "spanId": csid, "parentSpanId": root_id,
                                "name": ("execute_tool " if operation else "tool_call ") + (fname if isinstance(fname, str) else ""),
                                "kind": 1, "startTimeUnixNano": str(t), "endTimeUnixNano": str(t)}
        err = cx_d.get("error")
        if err and operation:
            if isinstance(err, dict) and isinstance(err.get("type"), str):
                attrs["error.type"] = err["type"]
            msg = err.get("message") if isinstance(err, dict) and isinstance(err.get("message"), str) else None
            span["status"] = {"code": 2, **({"message": msg} if msg else {})}
        span["attributes"] = encode_attributes(attrs)
        return span

    # -- links -----------------------------------------------------------------------------------

    def _resolve(self, addr: Any, cur: Any) -> Optional[str]:
        """The span an address names: a kept span, a call, a step (a result resolves to its call or step), or a trajectory."""
        if not isinstance(addr, dict):
            return None
        if isinstance(addr.get("span"), str):
            return self.kept_span.get(addr["span"])
        t = addr.get("trajectory")
        t = cur if t is None else t
        if "step" in addr:
            step = addr["step"]
            if "call" in addr:
                return self.call_span.get((t, step, addr["call"])) or self.step_span.get((t, step))
            if _is_int(addr.get("result")):
                calls = self.result_calls.get((t, step)) or []
                r = addr["result"]
                cid = calls[r] if 0 <= r < len(calls) else None
                return (self.call_span.get((t, step, cid)) if cid is not None else None) or self.step_span.get((t, step))
            return self.step_span.get((t, step))
        return self.traj_span.get(t) if set(addr) == {"trajectory"} else None

    def _link(self, entry: Any, tid: Any, own: str, root_id: str) -> Dict[str, Any]:
        """A link from the step that holds a dependency entry to the other end of it; unresolved ends link to the root."""
        attrs: Dict[str, Any] = {"agentpile.dep": _j(entry)}
        target: Optional[str] = None
        if isinstance(entry, dict):
            for key in ("rule", "rule_version", "resource"):
                if isinstance(entry.get(key), str):
                    attrs["agentpile.dep." + key] = entry[key]
            ends = [self._resolve(entry.get("src"), tid), self._resolve(entry.get("dst"), tid)]
            target = next((e for e in ends if e is not None and e != own), None)
            if target is None and ends[0] == own and ends[1] == own:
                target = own  # both ends are this step
        attrs["agentpile.dep.resolved"] = target is not None
        return {"traceId": self.trace, "spanId": target or root_id, "attributes": encode_attributes(attrs)}

    # -- kept source spans no step, call or subagent came from ---------------------------------------------

    def _emit_kept(self, tpath: str, root_id: str, base: int) -> None:
        ids = self.span_ids[tpath]
        for n, (key, entry) in enumerate(self.kept[tpath].items()):
            if (tpath, key) in self.claimed:
                continue  # written as the step, call or subagent span it is the source of
            claim = _Claim(tpath, key, entry, n, ids[key])
            span, synthetic = self._adopt(claim, root_id, base)
            head = {"agentpile.version": EXT_VERSION, "agentpile.kind": "span", "agentpile.index": n}
            span["attributes"] = encode_attributes(self._adopted_attrs(claim, head, synthetic, None))
            self.spans.append(span)


def to_document(run: Run, trace_id: Optional[str] = None) -> Dict[str, Any]:
    """The run as an OTLP/JSON ``ExportTraceServiceRequest`` (a dict)."""
    if run.kind != TRAJECTORY:
        raise ValueError("the otlp-json writer takes trajectory records; a config record has no steps to write as spans")
    from .. import __version__
    spans = _Writer(run.record, trace_id).build()
    agent = run.record.get("agent") if isinstance(run.record.get("agent"), dict) else {}
    service = agent.get("name") if isinstance(agent.get("name"), str) and agent.get("name") else "agentpile"
    resource = {"service.name": service, "telemetry.sdk.language": "python", "telemetry.sdk.name": "agentpile",
                "telemetry.sdk.version": __version__}
    return {"resourceSpans": [{"resource": {"attributes": encode_attributes(resource)},
                               "scopeSpans": [{"scope": {"name": "agentpile", "version": __version__}, "spans": spans}]}]}


def dumps(run: Run, indent: Optional[int] = None, trace_id: Optional[str] = None) -> str:
    doc = to_document(run, trace_id=trace_id)
    text = json.dumps(doc, ensure_ascii=False, indent=indent)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:  # a lone surrogate in the source: escape instead of failing on write
        text = json.dumps(doc, ensure_ascii=True, indent=indent)
    return text


def write(run: Run, dest: Union[str, Path], append: bool = False, indent: Optional[int] = None) -> None:
    """Write the run as OTLP JSON. ``.jsonl`` and ``.ndjson`` get one compact document per line; ``append=True`` adds a line."""
    path = Path(dest)
    jsonl = _is_jsonl(path)
    if append and not jsonl:
        raise ValueError("append=True needs a .jsonl or .ndjson destination: one document per line")
    text = dumps(run, indent=None if jsonl else (2 if indent is None else indent))
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")


# =============================================================================
# Reader: spans written by agentpile
# =============================================================================

def _obj(attrs: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    p = prefix + "."
    return {k[len(p):]: _unj(v) for k, v in attrs.items() if k.startswith(p)}


def _ext_of(attrs: Dict[str, Any]) -> Any:
    """The agentpile extension held on a span: a dict, or ABSENT when the span held none."""
    keyed = _obj(attrs, "agentpile.ext")
    if keyed or "agentpile.ext" in attrs:
        return keyed
    return ABSENT


def _combine_extra(extra_attr: Any, ext: Any) -> Any:
    if ext is ABSENT:
        return extra_attr
    base: Dict[str, Any] = {} if extra_attr is ABSENT else dict(extra_attr)
    base[EXT] = ext
    return base


def _order(s: _Span) -> int:
    v = s.attrs.get("agentpile.index")
    return v if isinstance(v, int) else 0


def _kept_order(s: _Span) -> int:
    """Position of the span's entry in the ``spans`` field it came from."""
    v = s.attrs.get("agentpile.span_index")
    return v if isinstance(v, int) and not isinstance(v, bool) else _order(s)


def _rebuild_call(sp: _Span) -> Tuple[Dict[str, Any], Any]:
    return _obj(sp.attrs, "agentpile.call"), _ext_of(sp.attrs)


def _rebuild_step(sp: _Span, call_spans: List[_Span]) -> Dict[str, Any]:
    step = _obj(sp.attrs, "agentpile.step")
    extra_attr = step.pop("extra", ABSENT)
    ext = _ext_of(sp.attrs)
    built = [_rebuild_call(c) for c in sorted(call_spans, key=_order)]
    if built and "tool_calls" not in step:
        step["tool_calls"] = [c for c, _ in built]
    if isinstance(ext, dict):
        calls_attr = ext.pop("calls", ABSENT)
        deps_attr = ext.pop("deps", ABSENT)
        cxs = {c["tool_call_id"]: cx for c, cx in built if cx is not ABSENT and isinstance(c.get("tool_call_id"), str)}
        if calls_attr is not ABSENT or cxs:
            merged = calls_attr if (calls_attr is not ABSENT and not isinstance(calls_attr, dict)) else dict(calls_attr if calls_attr is not ABSENT else {})
            if isinstance(merged, dict):
                merged.update(cxs)
            ext["calls"] = merged
        if deps_attr is not ABSENT:
            ext["deps"] = deps_attr
        else:
            deps = [_unj(lk["attributes"]["agentpile.dep"]) for lk in sp.entry["links"]
                    if isinstance(lk.get("attributes"), dict) and "agentpile.dep" in lk["attributes"]]
            if deps:
                ext["deps"] = deps
    extra = _combine_extra(extra_attr, ext)
    if extra is not ABSENT:
        step["extra"] = extra
    return step


def _read_agentpile(spans: List[_Span]) -> Iterator[Run]:
    by_id = {s.id: s for s in spans if s.id}

    def kind_of(s: _Span) -> Optional[str]:
        return s.attrs.get("agentpile.kind") if isinstance(s.attrs.get("agentpile.version"), str) else None

    def nearest_root(s: Optional[_Span]) -> Optional[_Span]:
        guard = 0
        while s is not None and guard < 100000:
            if kind_of(s) in ("trajectory", "subagent"):
                return s
            s = by_id.get(s.parent) if s.parent else None
            guard += 1
        return None

    def named_root(s: _Span, attr: str) -> Optional[_Span]:
        """The trajectory span an attribute names by id (the writer sets it where the parent chain is the source's own)."""
        oid = s.attrs.get(attr)
        root = by_id.get(oid) if isinstance(oid, str) else None
        return root if root is not None and kind_of(root) in ("trajectory", "subagent") else None

    top = [s for s in spans if kind_of(s) == "trajectory"]
    table: Dict[str, Dict[str, List[_Span]]] = {"step": {}, "call": {}, "subagent": {}}
    held: Dict[str, List[_Span]] = {}  # trajectory span key -> spans whose kept entry its ``spans`` field holds
    foreign: List[_Span] = []
    for s in spans:
        k = kind_of(s)
        if k in ("step", "call", "span", "subagent"):
            if k == "span":
                anchor = named_root(s, "agentpile.span_owner") or nearest_root(s)
            elif k == "subagent":
                anchor = named_root(s, "agentpile.owner") or (nearest_root(by_id.get(s.parent)) if s.parent else None)
            else:
                anchor = named_root(s, "agentpile.owner") or nearest_root(s)
            if k != "span":
                table[k].setdefault((anchor or top[0]).key, []).append(s)
            if k == "span" or "agentpile.span" in s.attrs:  # a source span, or one written as this step, call or subagent
                holder = named_root(s, "agentpile.span_owner") or anchor
                held.setdefault((holder or top[0]).key, []).append(s)
        elif k is None or k not in ("trajectory",):
            foreign.append(s)

    def build(root: _Span, first: bool) -> Dict[str, Any]:
        attrs = root.attrs
        traj = _obj(attrs, "agentpile.trajectory")
        extra_attr = traj.pop("extra", ABSENT)
        out: Dict[str, Any] = {}
        for key in ("schema_version", "session_id", "trajectory_id"):
            if key in traj:
                out[key] = traj.pop(key)
        out["agent"] = _obj(attrs, "agentpile.agent")
        call_spans: Dict[Any, List[_Span]] = {}
        for c in table["call"].get(root.key, []):
            call_spans.setdefault(c.attrs.get("agentpile.step_index"), []).append(c)
        out["steps"] = [_rebuild_step(s, call_spans.get(_order(s), [])) for s in sorted(table["step"].get(root.key, []), key=_order)]
        out.update(traj)
        children = sorted(table["subagent"].get(root.key, []), key=_order)
        if children:
            out["subagent_trajectories"] = [build(c, False) for c in children]
        ext = _ext_of(attrs)
        kept = sorted(held.get(root.key, []), key=_kept_order)
        extra_spans = {s.attrs.get("agentpile.span_key", s.key): _unj(s.attrs.get("agentpile.span")) for s in kept}
        if first:  # spans in the trace that agentpile did not write
            for f in foreign:
                extra_spans[_unique_key(extra_spans, f.key)] = f.entry
        if extra_spans:
            ext = ext if isinstance(ext, dict) else {}
            ext["spans"] = extra_spans
        extra = _combine_extra(extra_attr, ext)
        if extra is not ABSENT:
            out["extra"] = extra
        return out

    for n, root in enumerate(top):
        yield Run(build(root, n == 0), TRAJECTORY)


# =============================================================================
# Reader: spans of other sources
# =============================================================================

def _semconv_info(spans: List[_Span]) -> Dict[str, Any]:
    """Scopes and schema URLs the spans declare, and the semantic convention version when a schema URL gives one."""
    scopes: List[Dict[str, Any]] = []
    urls: List[str] = []
    for s in spans:
        if s.scope_name or s.scope_version:
            item = {"name": s.scope_name, "version": s.scope_version}
            if item not in scopes:
                scopes.append(item)
        if s.schema_url and s.schema_url not in urls:
            urls.append(s.schema_url)
    out: Dict[str, Any] = {}
    if scopes:
        out["scopes"] = scopes
    if urls:
        versions: List[str] = []
        for u in urls:
            v = u.rstrip("/").rsplit("/", 1)[-1]
            if v not in versions:
                versions.append(v)
        out["schema_urls"] = urls
        out["semconv_version"] = versions[0] if len(versions) == 1 else versions
    return out


def _detect(spans: List[_Span]) -> Tuple[Optional[str], List[str]]:
    found = []
    for scope, (dialect, _) in DIALECTS.items():
        scoped = [s for s in spans if s.scope_name == scope]
        marker = "gen_ai.operation.name" if dialect.startswith("pydantic") else "openinference.span.kind"
        if any(marker in s.attrs for s in scoped):
            found.append(dialect)
    if len(found) == 1:
        return found[0], []
    if len(found) > 1:
        return None, ["steps: spans of two supported dialects (%s) share one trace" % ", ".join(found)]
    return None, []


def _scope_versions(spans: List[_Span], dialect: str) -> List[str]:
    """The instrumentation scope versions the spans of a dialect declare ("" when a span declares none)."""
    scope = _SCOPE_OF[dialect]
    return sorted({s.scope_version for s in spans if s.scope_name == scope})


def _is_tested(found: List[str], dialect: str) -> bool:
    """True when the spans declare one scope version and it is one this release was tested with."""
    return len(found) == 1 and found[0] in _TESTED[dialect]


def _check_tested(spans: List[_Span], dialect: str, allow_untested: bool) -> None:
    """Refuse to interpret a dialect at a scope version it was not tested with, unless the caller opted in."""
    found = _scope_versions(spans, dialect)
    if _is_tested(found, dialect) or allow_untested:
        return
    shown = ", ".join(repr(v) if v else "none declared" for v in found)
    tested = ", ".join(_TESTED[dialect])
    raise UnsupportedVersion(
        f"{_SCOPE_OF[dialect]} version {shown} is not a tested version of the {dialect} dialect (tested: {tested}); "
        f"its spans are not interpreted into steps because the layout may differ. Pass allow_untested_version=True "
        f"to read them anyway (the run is then marked source.dialect_tested = false)")


def _read_foreign(trace_id: str, spans: List[_Span], allow_untested: bool = False) -> Run:
    dialect, notes = _detect(spans)
    info = _semconv_info(spans)
    if dialect is None:
        return _generic_run(trace_id, spans, info, notes)
    _check_tested(spans, dialect, allow_untested)
    return _Builder(trace_id, spans, dialect, info).run()


def _generic_run(trace_id: str, spans: List[_Span], info: Dict[str, Any], notes: List[str]) -> Run:
    lost = list(notes) or ["steps: spans of an unsupported dialect are not mapped to steps; they are kept in spans"]
    lost += ["tool_calls: not mapped", "observation: not mapped"]
    source: Dict[str, Any] = {"format": name, "record_key": trace_id, "reader": name, "reader_version": READER_VERSION,
                              "lost_fields": lost, "dialect": "generic"}
    source.update(info)
    ext = {"version": EXT_VERSION, "source": source, "task": {"id": None, "text": None},
           "outcome": {"success": None, "score": None, "raw": None}, "labels": [], "agents": [], "tools": {},
           "spans": {s.key: s.entry for s in spans}}
    traj = {"schema_version": ATIF_VERSION, "trajectory_id": trace_id,
            "agent": {"name": "unknown", "version": "unknown"},
            "steps": [{"step_id": 1, "source": "system", "message": "",
                       "extra": {EXT: {"message_state": "missing", "empty_source": True}}}],
            "extra": {EXT: ext}}
    return Run(traj, TRAJECTORY)


def _text_key(item: Dict[str, Any]) -> str:
    return "\n".join(item["parts"])


def _norm_content(raw: Any) -> Tuple[str, Dict[str, Any]]:
    """(ATIF content, extra result fields): a missing, null or non-text source value keeps its own form in the fields."""
    if raw is ABSENT:
        return "", {"content_state": "missing"}
    if raw is None:
        return "", {"content_state": "null"}
    if isinstance(raw, str):
        return raw, {}
    return json.dumps(raw, ensure_ascii=False), {"content_raw": raw}


class _Builder:
    """Pydantic AI (OTel GenAI) and OpenAI Agents (OpenInference) spans -> one trajectory, plus one per delegated agent."""

    def __init__(self, trace_id: str, spans: List[_Span], dialect: str, info: Dict[str, Any]):
        self.trace_id = trace_id
        self.spans = spans
        self.dialect = dialect
        self.info = info
        self.genai = dialect == "pydantic-ai/genai"
        self.by_id = {s.id: s for s in spans if s.id}
        self.by_key = {s.key: s for s in spans}
        self.lost: List[str] = []
        self.addresses: Dict[str, Optional[Dict[str, Any]]] = {s.key: None for s in spans}
        self.delegated: set = set()

    def note(self, text: str) -> None:
        if text not in self.lost:
            self.lost.append(text)

    # -- span classification

    def in_scope(self, s: _Span) -> bool:
        return DIALECTS.get(s.scope_name, (None,))[0] == self.dialect

    def op(self, s: _Span) -> Optional[str]:
        v = s.attrs.get("gen_ai.operation.name")
        return v if isinstance(v, str) else None

    def oi_kind(self, s: _Span) -> Optional[str]:
        v = s.attrs.get("openinference.span.kind")
        return v if isinstance(v, str) else None

    def is_llm(self, s: _Span) -> bool:
        if not self.in_scope(s):
            return False
        return self.op(s) in ("chat", "generate_content", "text_completion") if self.genai else self.oi_kind(s) == "LLM"

    def is_agent(self, s: _Span) -> bool:
        if not self.in_scope(s):
            return False
        return self.op(s) == "invoke_agent" if self.genai else (self.oi_kind(s) == "AGENT" and isinstance(s.attrs.get("agent.name"), str))

    def is_tool(self, s: _Span) -> bool:
        if not self.in_scope(s):
            return False
        return self.op(s) == "execute_tool" if self.genai else self.oi_kind(s) == "TOOL"

    def agent_name(self, s: _Span) -> Optional[str]:
        v = s.attrs.get("gen_ai.agent.name" if self.genai else "agent.name")
        return v if isinstance(v, str) and v else None

    def ancestors(self, s: _Span) -> Iterator[_Span]:
        cur = self.by_id.get(s.parent) if s.parent else None
        guard = 0
        while cur is not None and guard < 100000:
            yield cur
            cur = self.by_id.get(cur.parent) if cur.parent else None
            guard += 1

    def unit_of(self, s: _Span) -> Optional[str]:
        """The key of the delegated agent whose run contains this span, or None for the main run."""
        for a in self.ancestors(s):
            if a.key in self.delegated:
                return a.key
        return None

    def depth(self, key: Optional[str]) -> int:
        if key is None:
            return 0
        return 1 + sum(1 for a in self.ancestors(self.by_key[key]) if a.key in self.delegated)

    # -- runs

    def run(self) -> Run:
        found = _scope_versions(self.spans, self.dialect)
        if not _is_tested(found, self.dialect):  # only reached when the caller opted in
            self.note("dialect: instrumentation scope version %s was not tested (tested: %s); the spans were interpreted "
                      "with the tested layout because allow_untested_version=True"
                      % (", ".join(repr(v) if v else "none declared" for v in found), ", ".join(_TESTED[self.dialect])))
        agents = [s for s in self.spans if self.is_agent(s)]
        self.delegated = {a.key for a in agents if any(self.is_tool(x) for x in self.ancestors(a))}
        units: Dict[Optional[str], Dict[str, List[_Span]]] = {None: {"llm": [], "tool": [], "agent": []}}
        for k in self.delegated:
            units[k] = {"llm": [], "tool": [], "agent": []}
        for s in self.spans:
            for role, test in (("llm", self.is_llm), ("tool", self.is_tool), ("agent", self.is_agent)):
                if test(s):
                    # a delegated agent span belongs to its own run; everything else to the nearest delegated ancestor
                    units[s.key if s.key in self.delegated and role == "agent" else self.unit_of(s)][role].append(s)
        built: Dict[Optional[str], Dict[str, Any]] = {}
        for key in sorted(units, key=lambda k: -self.depth(k)):  # leaves first, so a parent can reference its subagents
            built[key] = self.build_unit(key, units[key], built)
        if not self.genai and any(self.is_tool(s) and s.attrs.get("output.value") is not None for s in self.spans):
            self.note("openinference TOOL span output: these spans carry no tool call id, so they stay in spans only "
                      "(results come from later llm.input_messages with a message.tool_call_id)")
        root = built[None]
        for traj in self.walk(root):
            traj["extra"][EXT]["source"]["lost_fields"] = list(self.lost)
        root["extra"][EXT]["spans"] = {s.key: dict(s.entry, address=self.addresses.get(s.key)) for s in self.spans}
        return Run(root, TRAJECTORY)

    def walk(self, traj: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        yield traj
        for sub in traj.get("subagent_trajectories") or []:
            yield from self.walk(sub)

    def build_unit(self, key: Optional[str], unit: Dict[str, List[_Span]], built: Dict[Optional[str], Dict[str, Any]]) -> Dict[str, Any]:
        tid = self.trace_id if key is None else (self.by_key[key].id or key)
        llms = sorted(unit["llm"], key=lambda s: (s.start, s.order))
        parsed = [self.parse_llm(s) for s in llms]
        asm = _Steps(self, tid)
        roster, tools, defs, top_name, models = self.roster(unit, parsed, llms, asm)
        for s, p in zip(llms, parsed):
            asm.add_llm(s, p)
        by_call: Dict[str, List[_Span]] = {}
        if self.genai:
            for t in sorted(unit["tool"], key=lambda x: (x.start, x.order)):
                cid = t.attrs.get("gen_ai.tool.call.id")
                if isinstance(cid, str) and cid:
                    by_call.setdefault(cid, []).append(t)
                else:
                    self.note("gen_ai.operation.name=execute_tool span without gen_ai.tool.call.id: kept in spans only")
        steps = asm.build(by_call)
        if not steps:  # ATIF needs one step; a trace with no model call span has none to map
            self.note("steps: no model call span in the trace; one empty placeholder step stands in")
            steps = [{"step_id": 1, "source": "system", "message": "",
                      "extra": {EXT: {"message_state": "missing", "empty_source": True}}}]

        agent: Dict[str, Any] = {"name": top_name or "unknown", "version": "unknown"}
        if self.genai:
            for a in unit["agent"]:
                v = a.attrs.get("gen_ai.agent.version")
                if isinstance(v, str) and v:
                    agent["version"] = v
                    break
        if len(models) == 1:
            agent["model_name"] = models[0]
        if defs:
            agent["tool_definitions"] = defs
        traj: Dict[str, Any] = {"schema_version": ATIF_VERSION}
        for s in unit["agent"] + llms:
            v = s.attrs.get("gen_ai.conversation.id" if self.genai else "session.id")
            if isinstance(v, str) and v:
                traj["session_id"] = v
                break
        traj["trajectory_id"] = tid
        traj["agent"] = agent
        traj["steps"] = steps
        source: Dict[str, Any] = {"format": name, "record_key": self.trace_id, "reader": name, "reader_version": READER_VERSION,
                                  "lost_fields": [], "dialect": self.dialect}
        source.update(self.info)
        source["dialect_tested"] = _is_tested(_scope_versions(self.spans, self.dialect), self.dialect)
        traj["extra"] = {EXT: {"version": EXT_VERSION, "source": source, "task": {"id": None, "text": None},
                               "outcome": {"success": None, "score": None, "raw": None}, "labels": [],
                               "agents": roster, "tools": tools}}
        subs = [(k2, sub) for k2, sub in built.items() if k2 is not None and self.unit_of(self.by_key[k2]) == key]
        if subs:
            traj["subagent_trajectories"] = [sub for _, sub in subs]
            for k2, sub in subs:
                asm.attach_subagent(steps, self.by_key[k2], sub["trajectory_id"])
        return traj

    # -- LLM span parsing

    def parse_llm(self, s: _Span) -> Dict[str, Any]:
        a = s.attrs
        if self.genai:
            items_in = self.genai_items(a.get("gen_ai.input.messages"), "gen_ai.input.messages")
            items_out = self.genai_items(a.get("gen_ai.output.messages"), "gen_ai.output.messages")
            model = a.get("gen_ai.response.model") if isinstance(a.get("gen_ai.response.model"), str) else a.get("gen_ai.request.model")
            usage = {"prompt_tokens": a.get("gen_ai.usage.input_tokens"), "completion_tokens": a.get("gen_ai.usage.output_tokens"),
                     "cached_tokens": a.get("gen_ai.usage.cache_read.input_tokens")}
            instructions = self.genai_instructions(s)
            if "gen_ai.input.messages" not in a and "gen_ai.output.messages" not in a:
                self.note("gen_ai.input.messages and gen_ai.output.messages are absent on a model call span (content capture off, "
                          "or a data format that keeps content in span events): that step has no message")
            defs = []
            raw_defs = _unj(a["gen_ai.tool.definitions"]) if a.get("gen_ai.tool.definitions") is not None else []
            for d in raw_defs if isinstance(raw_defs, list) else []:
                if isinstance(d, dict) and isinstance(d.get("name"), str):
                    fn = {"name": d["name"]}
                    for k in ("description", "parameters"):
                        if d.get(k) is not None:
                            fn[k] = d[k]
                    defs.append({"type": "function", "function": fn})
                else:
                    self.note("gen_ai.tool.definitions: a definition that is not a named function is not mapped")
        else:
            items_in = self.oi_items(a, "llm.input_messages")
            items_out = self.oi_items(a, "llm.output_messages")
            model = a.get("llm.model_name") if isinstance(a.get("llm.model_name"), str) else None
            usage = {"prompt_tokens": a.get("llm.token_count.prompt"), "completion_tokens": a.get("llm.token_count.completion"),
                     "cached_tokens": a.get("llm.token_count.prompt_details.cache_read")}
            instructions = next((_text_key(it) for it in items_in if it["k"] == "text" and it["role"] in ("system", "developer")), None)
            if not items_in and not items_out:
                self.note("llm.input_messages and llm.output_messages are absent on a model call span (content hidden): "
                          "that step has no message")
            defs = []
            indexes = sorted({int(m.group(1)) for k in a for m in [re.match(r"^llm\.tools\.(\d+)\.tool\.json_schema$", k)] if m})
            for i in indexes:
                d = _unj(a.get(f"llm.tools.{i}.tool.json_schema"))
                if isinstance(d, dict):
                    defs.append(d)
        return {"in": items_in, "out": items_out, "model": model, "usage": usage, "instructions": instructions, "defs": defs}

    def genai_instructions(self, s: _Span) -> Optional[str]:
        for c in [s] + [x for x in self.ancestors(s) if self.is_agent(x)][:1]:
            raw = c.attrs.get("gen_ai.system_instructions")
            if raw is None:
                continue
            parsed = _unj(raw)
            if isinstance(parsed, list):
                texts = [p["content"] for p in parsed if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("content"), str)]
                if texts:
                    return "\n\n".join(texts)
            elif isinstance(parsed, str):
                return parsed
        return None

    def genai_items(self, raw: Any, where: str) -> List[Dict[str, Any]]:
        """Messages as items: one text item per message, then its reasoning, calls and results (tool_call_response parts)."""
        if raw is None:
            return []
        msgs = _unj(raw)
        if not isinstance(msgs, list):
            self.note(f"{where}: not a JSON list of messages, not mapped")
            return []
        items: List[Dict[str, Any]] = []
        for msg in msgs:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            parts = msg.get("parts")
            if parts is None and isinstance(msg.get("content"), str):
                parts = [{"type": "text", "content": msg["content"]}]
            texts: List[str] = []
            rest: List[Dict[str, Any]] = []
            for p in parts if isinstance(parts, list) else []:
                t = p.get("type") if isinstance(p, dict) else None
                if t == "text" and isinstance(p.get("content"), str):
                    texts.append(p["content"])
                elif t == "tool_call":
                    rest.append({"k": "call", "id": p.get("id"), "name": p.get("name"), "args": p["arguments"] if "arguments" in p else ABSENT})
                elif t == "tool_call_response":
                    # the convention names the field "response"; Pydantic AI 2.58.0 writes "result"
                    rest.append({"k": "result", "id": p.get("id"), "name": p.get("name"),
                                 "content": p["result"] if "result" in p else (p["response"] if "response" in p else ABSENT)})
                elif t in ("reasoning", "thinking") and isinstance(p.get("content"), str):
                    rest.append({"k": "reasoning", "text": p["content"]})
                else:
                    self.note(f"{where}: message part type {t!r} is not mapped")
            if texts:
                items.append({"k": "text", "role": role, "parts": texts})
            items.extend(rest)
        return items

    def oi_items(self, attrs: Dict[str, Any], prefix: str) -> List[Dict[str, Any]]:
        msgs: Dict[int, Dict[str, Any]] = {}
        p = prefix + "."
        for k, v in attrs.items():
            if not k.startswith(p):
                continue
            idx, _, tail = k[len(p):].partition(".")
            if not idx.isdigit():
                continue
            m = msgs.setdefault(int(idx), {"contents": {}, "calls": {}})
            mc = re.match(r"^message\.contents\.(\d+)\.message_content\.(\w+)$", tail)
            tc = re.match(r"^message\.tool_calls\.(\d+)\.tool_call\.(id|function\.name|function\.arguments)$", tail)
            if tail in ("message.role", "message.content", "message.tool_call_id", "message.name"):
                m[tail[len("message."):]] = v
            elif mc:
                m["contents"].setdefault(int(mc.group(1)), {})[mc.group(2)] = v
            elif tc:
                m["calls"].setdefault(int(tc.group(1)), {})[tc.group(2)] = v
            else:
                self.note(f"{prefix}.N.{re.sub(r'(?<=[.])[0-9]+(?=[.]|$)', 'N', tail)}: attribute not mapped")
        items: List[Dict[str, Any]] = []
        for i in sorted(msgs):
            m = msgs[i]
            role = m.get("role")
            texts: List[str] = []
            reasoning: List[str] = []
            for j in sorted(m["contents"]):
                c = m["contents"][j]
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    texts.append(c["text"])
                elif c.get("type") == "reasoning":
                    if isinstance(c.get("text"), str):
                        reasoning.append(c["text"])
                else:
                    self.note(f"{prefix}.N.message.contents.M.message_content.type={c.get('type')!r}: not mapped")
            if not m["contents"] and isinstance(m.get("content"), str):
                texts = [m["content"]]
            if role == "tool":
                items.append({"k": "result", "id": m.get("tool_call_id"), "name": m.get("name"),
                              "content": m["content"] if "content" in m else ABSENT})
                continue
            if texts:
                items.append({"k": "text", "role": role, "parts": texts})
            items.extend({"k": "reasoning", "text": r} for r in reasoning)
            for j in sorted(m["calls"]):
                c = m["calls"][j]
                items.append({"k": "call", "id": c.get("id"), "name": c.get("function.name"),
                              "args": c["function.arguments"] if "function.arguments" in c else ABSENT})
        return items

    # -- roster and tools

    def roster(self, unit: Dict[str, List[_Span]], parsed: List[Dict[str, Any]], llms: List[_Span], asm: "_Steps"):
        names: List[str] = []
        info: Dict[str, Dict[str, Any]] = {}

        def entry(n: str) -> Dict[str, Any]:
            if n not in info:
                names.append(n)
                info[n] = {"id": n, "name": n, "instructions": None, "model": None, "tools": [], "handoffs": [], "_models": []}
            return info[n]

        agent_spans = sorted(unit["agent"], key=lambda s: (s.start, s.order))
        for a in agent_spans:
            n = self.agent_name(a)
            if n:
                entry(n)
        tools: Dict[str, Dict[str, Any]] = {}
        defs: List[Dict[str, Any]] = []
        known: Dict[Tuple[str, str], str] = {}
        for s, p in zip(llms, parsed):
            who = asm.actor_of(s)
            if not who:
                continue
            e = entry(who)
            if e["instructions"] is None and p["instructions"] is not None:
                e["instructions"] = p["instructions"]
            if isinstance(p["model"], str) and p["model"] not in e["_models"]:
                e["_models"].append(p["model"])
            for d in p["defs"]:
                fname = d["function"]["name"] if isinstance(d.get("function"), dict) and isinstance(d["function"].get("name"), str) else None
                if fname is None:
                    continue
                dkey = (who, json.dumps(d, sort_keys=True))
                tool_id = known.get(dkey)
                if tool_id is None:
                    tool_id, n = f"{who}/{fname}", 2
                    while tool_id in tools:
                        tool_id, n = f"{who}/{fname}#{n}", n + 1
                    known[dkey] = tool_id
                    tools[tool_id] = {"name": fname, "agent_or_server": who, "type": None, "definition": d,
                                      "definition_format": "openai", "atif_index": len(defs),
                                      "permission": "unknown", "permission_from": "unknown"}
                    defs.append(d)
                if tool_id not in e["tools"]:
                    e["tools"].append(tool_id)
        if not self.genai:
            for a in agent_spans:
                parent, n = a.attrs.get("graph.node.parent_id"), self.agent_name(a)
                if isinstance(parent, str) and n:
                    pe = entry(parent)
                    if n not in pe["handoffs"]:
                        pe["handoffs"].append(n)
        roster, models = [], []
        for n in names:
            e = info[n]
            ms = e.pop("_models")
            e["model"] = ms[0] if len(ms) == 1 else None
            models.extend(m for m in ms if m not in models)
            roster.append(e)
        top = next((self.agent_name(a) for a in agent_spans if self.agent_name(a)), None) or (names[0] if names else None)
        asm.tool_ids = {(t["agent_or_server"], t["name"]): tid for tid, t in tools.items()}
        return roster, tools, defs, top, models


class _Steps:
    """The LLM spans of one trajectory, in start order, as ATIF steps. History a later span repeats is not repeated."""

    def __init__(self, b: _Builder, tid: str):
        self.b = b
        self.tid = tid
        self.drafts: List[Dict[str, Any]] = []
        self.have: Counter = Counter()  # items already shown, by key
        self.calls: Dict[str, Dict[str, Any]] = {}
        self.echoes: Dict[str, Tuple[Dict[str, Any], _Span]] = {}  # call id -> result a later model input carried
        self.stray: List[Tuple[Dict[str, Any], _Span]] = []  # results of a call no span issued
        self.actors: Dict[str, Optional[str]] = {}
        self.tool_ids: Dict[Tuple[Any, Any], str] = {}

    def actor_of(self, s: _Span) -> Optional[str]:
        if s.key not in self.actors:
            who = next((self.b.agent_name(a) for a in self.b.ancestors(s) if self.b.is_agent(a) and self.b.agent_name(a)), None)
            self.actors[s.key] = who or (self.b.agent_name(s) if self.b.genai else None)
        return self.actors[s.key]

    @staticmethod
    def key_of(it: Dict[str, Any]) -> Tuple:
        if it["k"] == "text":
            return ("text", it["role"], _text_key(it))
        if it["k"] == "call" and not (isinstance(it["id"], str) and it["id"]):
            return ("call", None, str(it["name"]), json.dumps(it["args"], sort_keys=True, default=str))
        if it["k"] in ("call", "result"):
            return (it["k"], it["id"] if isinstance(it["id"], str) else json.dumps(it["id"], sort_keys=True, default=str))
        return ("reasoning", it["text"])

    def add_llm(self, s: _Span, p: Dict[str, Any]) -> None:
        seen: Counter = Counter()
        open_agent: Optional[Dict[str, Any]] = None  # assistant items of the input that no span in this trace produced
        for it in p["in"]:
            key = self.key_of(it)
            seen[key] += 1
            if seen[key] <= self.have[key]:
                open_agent = None
                continue
            self.have[key] += 1
            if it["k"] == "text" and it["role"] in ("system", "developer", "user"):
                open_agent = None
                self.drafts.append({"source": "system" if it["role"] in ("system", "developer") else "user",
                                    "texts": list(it["parts"]), "role": it["role"], "span": s, "calls": [], "reasoning": []})
            elif it["k"] in ("text", "reasoning", "call"):
                if open_agent is None:
                    open_agent = self.agent_draft(s, None)
                    self.drafts.append(open_agent)
                self.fold(open_agent, it)
            elif isinstance(it["id"], str) and it["id"] in self.calls:
                self.echoes.setdefault(it["id"], (it, s))
            else:
                self.stray.append((it, s))
        draft = self.agent_draft(s, p)
        for it in p["out"]:
            self.have[self.key_of(it)] += 1
            self.fold(draft, it)
        self.drafts.append(draft)

    def agent_draft(self, s: _Span, p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        d: Dict[str, Any] = {"source": "agent", "texts": [], "reasoning": [], "calls": [], "span": s, "own": p is not None,
                             "actor": self.actor_of(s) if p is not None else None}
        if p is not None:
            d["model"], d["usage"] = p["model"], p["usage"]
        return d

    def fold(self, draft: Dict[str, Any], it: Dict[str, Any]) -> None:
        if it["k"] == "text":
            draft["texts"].extend(it["parts"])
        elif it["k"] == "reasoning":
            draft["reasoning"].append(it["text"])
        else:
            call = {"id": it["id"], "name": it["name"], "args": it["args"], "tool_spans": []}
            draft["calls"].append(call)
            if isinstance(it["id"], str) and it["id"] and it["id"] not in self.calls:
                self.calls[it["id"]] = call

    def attach_subagent(self, steps: List[Dict[str, Any]], agent_span: _Span, sub_tid: str) -> None:
        """An agent run nested under an execute_tool span is a subagent: the result of that call references it."""
        if not self.b.genai:
            return
        for a in self.b.ancestors(agent_span):
            if self.b.is_tool(a):
                cid = a.attrs.get("gen_ai.tool.call.id")
                for st in steps:
                    for res in (st.get("observation") or {}).get("results") or []:
                        if cid is not None and res.get("source_call_id") == cid:
                            res.setdefault("subagent_trajectory_ref", []).append({"trajectory_id": sub_tid})
                break

    # -- assembly

    def build(self, by_call: Dict[str, List[_Span]]) -> List[Dict[str, Any]]:
        b = self.b
        for n, d in enumerate(self.drafts, 1):
            d["id"] = n
        used: set = set()
        parts = [self.step_of(d, by_call, used) for d in self.drafts]
        agent_idx = [i for i, d in enumerate(self.drafts) if d["source"] == "agent"]
        strays = [(t, cid) for cid, ts in by_call.items() for t in ts if t.key not in used]
        if (strays or self.stray) and not agent_idx:
            b.note("tool results without any agent step are kept in spans only")
        elif agent_idx:
            into = parts[agent_idx[-1]]  # a result no call claims goes on the latest agent step, unpaired
            for t, cid in strays:
                name_ = t.attrs.get("gen_ai.tool.name")
                self.add(into, None, t.attrs.get("gen_ai.tool.call.result", ABSENT), t.key,
                         {"name": name_, "tool_call_id": cid} if isinstance(name_, str) else {"tool_call_id": cid})
            for it, s in self.stray:
                fields = {"name": it["name"]} if it.get("name") else {}
                if it.get("id") is not None:
                    fields["tool_call_id"] = it["id"]
                self.add(into, None, it.get("content", ABSENT), s.key, fields)
        out = []
        for d, (step, res) in zip(self.drafts, parts):
            if res["results"]:
                step["observation"] = {"results": res["results"]}
                sx = step["extra"][EXT]
                sx["result_refs"], sx["result_fields"] = res["refs"], res["fields"]
            out.append(step)
            if d["source"] == "agent" and d["own"]:
                b.addresses[d["span"].key] = {"trajectory": self.tid, "step": d["id"]}
            for c in d["calls"]:
                for t in c["tool_spans"]:
                    b.addresses[t.key] = {"trajectory": self.tid, "step": d["id"], "call": c["final_id"]}
        return out

    @staticmethod
    def add(part: Tuple[Dict[str, Any], Dict[str, Any]], call_id: Optional[str], raw: Any, ref: Any,
            fields: Optional[Dict[str, Any]]) -> None:
        content, extra_fields = _norm_content(raw)
        merged = dict(fields or {}, **extra_fields)
        res = part[1]
        res["results"].append({"source_call_id": call_id, "content": content})
        res["refs"].append(ref)
        res["fields"].append(merged or None)

    def step_of(self, d: Dict[str, Any], by_call: Dict[str, List[_Span]], used: set):
        b = self.b
        s: _Span = d["span"]
        sx: Dict[str, Any] = {"source_ref": s.key}
        step: Dict[str, Any] = {"step_id": d["id"], "source": d["source"]}
        texts = d["texts"]
        step["message"] = "" if not texts else (texts[0] if len(texts) == 1 else [{"type": "text", "text": t} for t in texts])
        res: Dict[str, List[Any]] = {"results": [], "refs": [], "fields": []}
        if d["source"] != "agent":
            if d.get("role") == "developer":
                sx["role"] = "developer"
            step["extra"] = {EXT: sx}
            return step, res
        ns = s.entry.get("start_time_unix_nano")
        if d["own"] and ns and iso_from_ns(ns):
            step["timestamp"] = iso_from_ns(ns)
        if not texts:
            sx["message_state"] = "missing"
        if isinstance(d.get("model"), str):
            step["model_name"] = d["model"]
        if d["reasoning"]:
            step["reasoning_content"] = "\n\n".join(d["reasoning"])
        metrics = {k: v for k, v in (d.get("usage") or {}).items() if _is_int(v)}
        if metrics:
            step["metrics"] = metrics
        if d.get("actor"):
            sx["actor"] = d["actor"]
        step["extra"] = {EXT: sx}
        part = (step, res)
        calls_out: List[Dict[str, Any]] = []
        cxs: Dict[str, Dict[str, Any]] = {}
        claimed: List[Tuple[int, int, _Span, Dict[str, Any]]] = []
        echoed: List[Tuple[str, Tuple[Dict[str, Any], _Span]]] = []
        for k, c in enumerate(d["calls"]):
            cid, fname = c["id"], c["name"]
            cx: Dict[str, Any] = {}
            if not isinstance(cid, str) or not cid or cid in cxs:
                if cid is not None:
                    cx["source_id"] = cid
                final = f"ap-{d['id']}-{k}"
                cx["id_generated"] = True
            else:
                final = cid
            c["final_id"] = final
            args, parsed = c["args"], {}
            if args is ABSENT:  # ATIF requires an object, so {} stands in and the call extension says why
                rule = OMITS_EMPTY_ARGUMENTS.get(b.dialect)
                if rule is not None:
                    cx["arguments_stand_in"] = rule  # the dialect leaves empty arguments out: {} is rebuilt, not read
                else:
                    cx["arguments_state"] = "missing"  # the source recorded no arguments member at all
            else:
                if isinstance(args, dict):
                    parsed = args
                else:
                    val = None
                    if isinstance(args, str):
                        try:
                            val = json.loads(args)
                        except ValueError:
                            val = None
                    if isinstance(val, dict):
                        parsed = val
                    else:
                        cx["arguments_raw"] = args
                        if args is not None:  # {} stands in for a value that is not an object (an explicit null is no failed parse)
                            cx["arguments_state"] = "unparsed"
            calls_out.append({"tool_call_id": final, "function_name": fname if isinstance(fname, str) else "", "arguments": parsed})
            if not isinstance(fname, str):
                cx["function_name_raw"] = fname
            cx["source_ref"] = s.key
            tool_id = self.tool_ids.get((d.get("actor"), fname))
            if tool_id:
                cx["tool_id"] = tool_id
            if isinstance(cid, str) and cid and not cx.get("id_generated"):
                mine = [t for t in by_call.get(cid, []) if t.key not in used]
                for t in mine:
                    used.add(t.key)
                    c["tool_spans"].append(t)
                if mine:
                    first = mine[0]
                    cx["span_id"] = first.id
                    if first.parent:
                        cx["parent"] = first.parent
                    st = first.entry.get("status")
                    if isinstance(st, dict) and st.get("code") == "ERROR":
                        err: Dict[str, Any] = {"status": "ERROR"}
                        if st.get("message"):
                            err["message"] = st["message"]
                        if isinstance(first.attrs.get("error.type"), str):
                            err["type"] = first.attrs["error.type"]
                        cx["error"] = err
                    for t in mine:
                        claimed.append((t.start, t.order, t, {"call": final}))
                elif cid in self.echoes and self.calls.get(cid) is c:
                    echoed.append((final, self.echoes[cid]))
            cxs[final] = cx
        if calls_out:
            step["tool_calls"] = calls_out
            sx["calls"] = {k: v for k, v in cxs.items() if v}
        for _, _, t, meta in sorted(claimed, key=lambda x: (x[0], x[1])):
            name_ = t.attrs.get("gen_ai.tool.name")
            self.add(part, meta["call"], t.attrs.get("gen_ai.tool.call.result", ABSENT), t.key,
                     {"name": name_} if isinstance(name_, str) else None)
        for final, (it, s2) in echoed:
            self.add(part, final, it.get("content", ABSENT), s2.key, {"name": it["name"]} if it.get("name") else None)
        if not sx.get("calls"):
            sx.pop("calls", None)
        return step, res
