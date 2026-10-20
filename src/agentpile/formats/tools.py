"""Readers for native tool-definition inputs, each producing an ``agentpile-config/0`` record.

Three shapes, three readers (registered as ``mcp-tools-list``, ``openai-tools``
and ``anthropic-tools``):

- **MCP** ``tools/list`` result: ``{"tools": [{name, description, inputSchema,
  annotations, ...}], "nextCursor": ...}``, the same object inside a JSON-RPC
  response (``{"jsonrpc": "2.0", "id": ..., "result": {...}}``), or the bare
  list of tools.
- **OpenAI** tool list: ``[{"type": "function", "function": {name, description,
  parameters}}]`` (Chat Completions), the flat ``{"type": "function", name,
  parameters}`` form (Responses), or the legacy ``functions`` form ``{name,
  description, parameters}``; as a bare list or ``{"tools": [...]}``.
- **Anthropic** tool list: ``[{name, description, input_schema}]``, a bare list
  or ``{"tools": [...]}``.

A tool list has no execution, so the record is a config record, never a
trajectory with invented steps. It holds one tool per definition:

- ``definition`` is the full definition exactly as the source gave it (schema,
  description, annotations, vendor fields), and ``definition_format`` names the
  shape (``mcp``, ``openai``, ``anthropic``).
- ``name`` is the tool's name; the tool id is that name, with ``#<index>`` added
  to a repeat of a name and for a definition that has none (``name`` is then
  ``""``). Nothing else is added: no server, no agent, no capability type.
- ``permission`` comes from MCP annotations when they state it: ``readOnlyHint:
  true`` is ``read``, ``readOnlyHint: false`` or ``destructiveHint: true`` is
  ``write``, each with ``permission_from: "annotation"``. A tool with no
  annotations, with annotations that say neither, or with annotations that
  contradict each other (read-only and destructive) is ``unknown`` with
  ``permission_from: "unknown"``. The other two shapes carry no annotations, so
  their tools are ``unknown``. Annotations are hints from the tool's server and
  are not checked here.
- Members of the document outside the tool list (``nextCursor``, ``_meta``, the
  JSON-RPC envelope) stay in ``dataset``.

Each reader's ``sniff`` claims only its own shape, strictly enough that no
trajectory, message list, config record or span document is taken for a tool
list. A bare tool list is claimed by its shape, whether it is parsed, a JSONL
line or the top level of a ``.json`` file; a ``.json`` file holding a list of
wrapped tool lists is several documents. A list with no tools is claimed by
none, because its shape is not evidenced; pass ``reader=`` to load it. ``read``
is more lenient than ``sniff`` (a definition without a name is kept, an extra
member of the wrapper goes to ``dataset``) but raises ``ValueError`` for a
document that holds no tool list.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .._io import _is_jsonl, _open_text
from ..record import CONFIG, CONFIG_SCHEMA, Run

READER_VERSION = "1"
SNIFF_LIMIT = 32 * 1024 * 1024  # a tool list is small; a larger file is not sniffed as one

MCP_WRAPPER_KEYS = {"tools", "nextCursor", "_meta"}
JSONRPC_KEYS = {"jsonrpc", "id", "result"}


# -- documents ------------------------------------------------------------------------

def _is_envelope(doc: Any) -> bool:
    """A document that wraps a tool list: ``{"tools": [...]}`` or a JSON-RPC response around one."""
    if not isinstance(doc, dict):
        return False
    if isinstance(doc.get("tools"), list):
        return True
    return "jsonrpc" in doc and isinstance(doc.get("result"), dict) and isinstance(doc["result"].get("tools"), list)


def _envelopes(value: Any) -> Optional[List[Any]]:
    """The wrapped tool lists in a list of them, or None when ``value`` is not such a list."""
    if isinstance(value, list) and value and all(_is_envelope(item) for item in value):
        return list(value)
    return None


def _split(value: Any) -> List[Any]:
    """The documents of one parsed JSON value: a list of wrapped tool lists is several, anything else is one."""
    envelopes = _envelopes(value)
    return envelopes if envelopes is not None else [value]


def _load(path: Path, first_only: bool = False) -> List[Any]:
    """The tool-list documents in a file: one per line of a JSONL file, else the whole file."""
    out: List[Any] = []
    with _open_text(path) as fh:
        if _is_jsonl(path):
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
                    if first_only:
                        break
            return out
        return _split(json.load(fh))


def _first_claimable(path: Path) -> Any:
    """The first document of a file that detection may claim.

    A line of a JSONL file is a document. A ``.json`` file holding a list of
    wrapped tool lists is several documents, claimed through its first item; any
    other top-level value, a bare tool list included, is one document that the
    shape-specific sniffers inspect.
    """
    if _is_jsonl(path):
        docs = _load(path, first_only=True)
        return docs[0] if docs else None
    with _open_text(path) as fh:
        value = json.load(fh)
    if isinstance(value, list):
        envelopes = _envelopes(value)
        return envelopes[0] if envelopes else value
    return value


_PEEKED: Dict[str, Any] = {}


def _peek(path: Path) -> Any:
    """The first document of a file that detection may claim, or None (not JSON, or too large).

    The three tool-list sniffers look at the same file in turn, so the last
    answer is kept.
    """
    try:
        stat = path.stat()
        if stat.st_size > SNIFF_LIMIT:
            return None
        key = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
        if _PEEKED.get("key") != key:
            doc = _first_claimable(path)
            _PEEKED.clear()
            _PEEKED.update(key=key, doc=doc)
        return _PEEKED["doc"]
    except (ValueError, OSError, UnicodeDecodeError):
        return None


def _unwrap(doc: Any) -> Optional[Tuple[List[Any], Dict[str, Any], Dict[str, Any]]]:
    """``(tools, other members of the tool-list object, other members of the JSON-RPC envelope)`` or None."""
    if isinstance(doc, list):
        return doc, {}, {}
    if not isinstance(doc, dict):
        return None
    envelope: Dict[str, Any] = {}
    if "jsonrpc" in doc and isinstance(doc.get("result"), dict):
        envelope = {k: v for k, v in doc.items() if k != "result"}
        doc = doc["result"]
    if isinstance(doc.get("tools"), list):
        return doc["tools"], {k: v for k, v in doc.items() if k != "tools"}, envelope
    return None


# -- the three shapes ---------------------------------------------------------------------

def _str_name(d: Dict[str, Any]) -> Optional[str]:
    name = d.get("name")
    return name if isinstance(name, str) and name else None


def _openai_function(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The object that holds the name and parameters of an OpenAI tool definition."""
    fn = d.get("function")
    return fn if isinstance(fn, dict) else d


def _openai_name(d: Dict[str, Any]) -> Optional[str]:
    return _str_name(_openai_function(d))


def _own_type(d: Dict[str, Any]) -> Optional[str]:
    return d.get("type") if isinstance(d.get("type"), str) else None


def _no_type(d: Dict[str, Any]) -> Optional[str]:
    return None


def _mcp_permission(d: Dict[str, Any]) -> Tuple[str, str]:
    """``(permission, permission_from)`` from the tool's annotations, only for what they state."""
    ann = d.get("annotations")
    if isinstance(ann, dict):
        read_only, destructive = ann.get("readOnlyHint"), ann.get("destructiveHint")
        if read_only is True and destructive is True:
            return "unknown", "unknown"  # the hints contradict each other
        if read_only is True:
            return "read", "annotation"
        if read_only is False or destructive is True:
            return "write", "annotation"
    return "unknown", "unknown"


def _unknown_permission(d: Dict[str, Any]) -> Tuple[str, str]:
    return "unknown", "unknown"


def _mcp_item(d: Any) -> bool:
    return isinstance(d, dict) and _str_name(d) is not None


def _mcp_evidence(d: Any) -> bool:
    return _mcp_item(d) and isinstance(d.get("inputSchema"), dict)


def _anthropic_item(d: Any) -> bool:
    return isinstance(d, dict) and _str_name(d) is not None


def _anthropic_evidence(d: Any) -> bool:
    return _anthropic_item(d) and isinstance(d.get("input_schema"), dict)


def _openai_evidence(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    if d.get("type") == "function":
        fn = d.get("function")
        if isinstance(fn, dict):
            return _str_name(fn) is not None
        return _str_name(d) is not None and isinstance(d.get("parameters"), dict)
    # the legacy ``functions`` form: no type, a name and a parameters schema
    return ("type" not in d and "function" not in d and _str_name(d) is not None
            and isinstance(d.get("parameters"), dict))


def _openai_item(d: Any) -> bool:
    # a hosted tool (``{"type": "web_search"}``) sits beside function definitions
    return _openai_evidence(d) or (isinstance(d, dict) and isinstance(d.get("type"), str) and d["type"] != "function")


class ToolListReader:
    """A reader for one tool-definition shape: ``sniff`` claims the shape, ``read`` builds the config record."""

    def __init__(self, name: str, definition_format: str, item: Callable[[Any], bool],
                 evidence: Callable[[Any], bool], tool_name: Callable[[Dict[str, Any]], Optional[str]],
                 tool_type: Callable[[Dict[str, Any]], Optional[str]],
                 permission: Callable[[Dict[str, Any]], Tuple[str, str]],
                 wrapper_keys: Optional[set] = None, jsonrpc: bool = False):
        self.name = name
        self.definition_format = definition_format
        self._item, self._evidence = item, evidence
        self._tool_name, self._tool_type, self._permission = tool_name, tool_type, permission
        self._wrapper_keys, self._jsonrpc = wrapper_keys or {"tools"}, jsonrpc

    # -- sniff

    def _claims(self, doc: Any) -> bool:
        if isinstance(doc, dict):
            if self._jsonrpc and doc.get("jsonrpc") == "2.0" and set(doc) <= JSONRPC_KEYS:
                doc = doc.get("result")
                if not isinstance(doc, dict):
                    return False
            if not (isinstance(doc.get("tools"), list) and set(doc) <= self._wrapper_keys):
                return False
            doc = doc["tools"]
        if not isinstance(doc, list) or not doc:
            return False
        return all(self._item(d) for d in doc) and any(self._evidence(d) for d in doc)

    def sniff(self, source: Any) -> bool:
        if isinstance(source, Path):
            return source.is_file() and self._claims(_peek(source))
        return self._claims(source)

    # -- read

    def read(self, source: Any, **options: Any) -> Iterator[Run]:
        if options:
            raise TypeError(f"{self.name} reader takes no options, got {sorted(options)}")
        path = source if isinstance(source, Path) else None
        docs = _load(path) if path is not None else _split(source)
        for k, doc in enumerate(docs):
            key = _record_key(path, k, len(docs))
            yield self._record(doc, key, path)

    def _record(self, doc: Any, key: str, path: Optional[Path]) -> Run:
        parts = _unwrap(doc)
        if parts is None:
            raise ValueError(f"{key}: not a {self.name} document (expected a list of tool definitions or an object with a "
                             f"'tools' list, got {type(doc).__name__})")
        defs, rest, envelope = parts
        names = [self._tool_name(d) if isinstance(d, dict) else None for d in defs]
        seen = Counter()
        tools: Dict[str, Dict[str, Any]] = {}
        for index, d in enumerate(defs):
            if not isinstance(d, dict):
                raise ValueError(f"{key}: tool definition {index} is not a JSON object")
            name = names[index]
            tid = name if name and seen[name] == 0 else f"{name or 'tool'}#{index}"
            while tid in tools:
                tid += "'"
            if name:
                seen[name] += 1
            permission, permission_from = self._permission(d)
            tools[tid] = {"name": name or "", "agent_or_server": None, "type": self._tool_type(d),
                          "definition": copy.deepcopy(d), "definition_format": self.definition_format,
                          "atif_index": None, "permission": permission, "permission_from": permission_from}
        source: Dict[str, Any] = {"format": self.name, "dataset": None, "file": path.name if path is not None else None,
                                  "record_key": key, "reader": self.name, "reader_version": READER_VERSION,
                                  "lost_fields": []}
        dataset: Dict[str, Any] = {}
        if rest:
            dataset["tool_list_members"] = copy.deepcopy(rest)
        if envelope:
            dataset["jsonrpc"] = copy.deepcopy(envelope)
        record: Dict[str, Any] = {"schema": CONFIG_SCHEMA, "source": source, "task": {"id": None, "text": None},
                                  "agents": [], "tools": tools, "labels": []}
        if dataset:
            record["dataset"] = dataset
        return Run(record, CONFIG)


def _record_key(path: Optional[Path], k: int, total: int) -> str:
    if path is None:
        return f"tools-{k}"
    base = path.name[:-3] if path.name.endswith(".gz") else path.name
    stem = Path(base).stem or "tools"
    return stem if total == 1 else f"{stem}-{k}"


mcp = ToolListReader("mcp-tools-list", "mcp", _mcp_item, _mcp_evidence, _str_name, _no_type, _mcp_permission,
                     wrapper_keys=MCP_WRAPPER_KEYS, jsonrpc=True)
openai = ToolListReader("openai-tools", "openai", _openai_item, _openai_evidence, _openai_name, _own_type,
                        _unknown_permission)
anthropic = ToolListReader("anthropic-tools", "anthropic", _anthropic_item, _anthropic_evidence, _str_name, _own_type,
                           _unknown_permission)
