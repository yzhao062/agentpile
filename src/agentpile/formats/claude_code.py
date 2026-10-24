"""Claude Code session files: the JSONL a session writes under ``~/.claude/projects/<project>/<session>.jsonl``.

The format is internal to Claude Code and has no public specification; this
reader was built on 2.1.x files. One file becomes one trajectory, read in file
order, and every step keeps its line number as ``source_ref``:

- A user record without a ``tool_result`` block becomes a user step. String
  content stays a string; list content follows the messages reader's part rule
  (a base64 image becomes a text stand-in, and the original list stays in
  ``message_raw``). ``isMeta`` and ``isCompactSummary`` go to ``source_fields``.
- Claude Code writes one assistant record per content block. Records sharing a
  ``message.id`` become one agent step, placed at the line of the message's
  first record: text is the message, thinking ``reasoning_content``, tool uses
  ``tool_calls`` (each call keeps its own record's line as its ``source_ref``),
  and the usage is the message's last record's: ATIF metrics from its token
  counts, and the raw usage in ``source_fields``.
- A ``tool_result`` block becomes an observation result on the step holding its
  call, paired by ``tool_use_id`` with a call whose message starts on an earlier
  line. Claude Code writes a result as soon as its tool finishes, which can be
  between, or before, later records of the same message; with the step at the
  message's first line and each result at its own line, ``Run.events()`` gives
  one assistant message followed by its results; its ``uuid`` and
  ``parentUuid`` go to ``result_fields``. A result with no such call stays
  unpaired on the latest agent step, as in the messages reader.
- A ``system`` record with subtype ``compact_boundary`` or ``api_error`` becomes
  a system step whose other fields go to ``source_fields``; other subtypes are
  counted.
- An ``attachment`` record is kept verbatim in ``extra["agentpile"]["dataset"]``,
  since the file does not record how Claude Code renders it into the prompt.
  Every other record type is counted and not copied, and so is ``toolUseResult``
  apart from its ``agentId`` and ``runId``. ``source.lost_fields`` names the
  keys left out.

``dataset`` also holds the node index (one row per line whose record has a
``uuid``), the branch points other than parallel calls, the subagent links and
the line accounting, in which every non-empty line is mapped, kept as an
attachment, counted, skipped as an exact duplicate or unparseable. Files named
``agent-<id>.jsonl`` under ``<session>/subagents/`` become subagent trajectories
with ``trajectory_id`` ``<id>``, linked through ``subagent_trajectory_ref`` from
the result whose ``toolUseResult.agentId`` names them, or whose
``toolUseResult.runId`` names their folder ``subagents/workflows/<runId>/``. A
link is never guessed by position or time: a named id or run with no file, an id
two files share, and a file no result names are listed in ``dataset``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .._io import _is_jsonl, _open_text
from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run
from .messages import _NO_RAW, _parse_arguments, _parts

name = "claude-code-jsonl"
READER_VERSION = "1"
SNIFF_LINES = 50
MESSAGE_TYPES = ("user", "assistant", "system")
SYSTEM_STEPS = ("compact_boundary", "api_error")
NODE_FIELDS = ["line", "uuid", "parentUuid", "logicalParentUuid", "type", "subtype"]
CATEGORIES = ("mapped", "attachment", "counted", "exact_duplicate", "unparseable")
AGENT_FILE = re.compile(r"^agent-(.+)\.jsonl$")
LINK_KEYS = ("agentId", "runId")
# Keys every record repeats; the trajectory keeps sessionId and version, and the first cwd and gitBranch.
ENVELOPE = {"uuid", "parentUuid", "logicalParentUuid", "type", "timestamp", "sessionId", "version", "cwd",
            "gitBranch", "isSidechain", "userType", "entrypoint", "slug"}
_USER_KEYS = ENVELOPE | {"message", "isMeta", "isCompactSummary", "isVisibleInTranscriptOnly", "toolUseResult"}
_ASSISTANT_KEYS = ENVELOPE | {"message", "requestId"}
_ABSENT = object()


def _head(path: Path, n: int) -> Optional[List[Any]]:
    out: List[Any] = []
    try:
        with _open_text(path) as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
                    if len(out) >= n:
                        break
    except (ValueError, UnicodeDecodeError, OSError):
        return None
    return out


def _qualifies(rec: Dict[str, Any]) -> bool:
    return rec.get("type") in MESSAGE_TYPES and all(isinstance(rec.get(k), str) for k in ("sessionId", "uuid", "version"))


def sniff(source: Any) -> bool:
    """A JSONL file whose first non-empty lines are typed records, at least one a message carrying a session id, a uuid and a version.

    Claude Code opens a file with bookkeeping records that carry none of these,
    so the first qualifying record is usually on line 3 to 6.
    """
    if not isinstance(source, Path) or not source.is_file() or not _is_jsonl(source):
        return False
    head = _head(source, SNIFF_LINES)
    if not head or not all(isinstance(r, dict) and isinstance(r.get("type"), str) for r in head):
        return False
    return any(_qualifies(r) for r in head)


def _digest(line: str) -> bytes:
    return hashlib.blake2b(line.encode("utf-8", "surrogatepass"), digest_size=16).digest()


def _ints(*values: Any) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


class _File:
    """One session file read into ATIF steps plus the reader's own accounting."""

    def __init__(self, path: Path):
        self.path = path
        self.steps: List[Dict[str, Any]] = []
        self.parts: Dict[int, Dict[str, Any]] = {}  # agent step index -> blocks gathered across the message's records
        self.msg_step: Dict[str, int] = {}  # message.id -> step index
        self.call_at: Dict[str, Tuple[int, int]] = {}  # call id -> (step index, line of the message's first record)
        self.answered: Counter = Counter()
        self.last_agent: Optional[int] = None
        self.seen: Dict[str, bytes] = {}  # uuid -> digest of the first line carrying it
        self.nodes: List[List[Any]] = []
        self.kinds: Dict[int, str] = {}  # line -> node kind, for branch labels
        self.duplicate_lines: set = set()
        self.attachments: List[Dict[str, Any]] = []
        self.acc: Counter = Counter()
        self.by_type: Counter = Counter()
        self.blocks: Counter = Counter()
        self.lost: set = set()
        self.versions: List[str] = []
        self.first: Dict[str, Any] = {}
        self.links: List[Tuple[int, int, str, str]] = []  # (step index, result index, "agentId" or "runId", value)
        self.pending: List[Tuple[int, Any, Any, Dict[str, Any], int]] = []  # results waiting for the whole file

    # -- reading -------------------------------------------------------------------

    def read(self) -> "_File":
        with _open_text(self.path) as fh:
            for lineno, line in enumerate(fh, 1):
                if line.strip():
                    self._line(lineno, line.strip())
        return self

    def _line(self, lineno: int, line: str) -> None:
        self.acc["lines"] += 1
        try:
            rec = json.loads(line)
        except ValueError:
            rec = None
        if not isinstance(rec, dict):
            self.acc["unparseable"] += 1
            return
        kind = rec.get("type") if isinstance(rec.get("type"), str) else "unknown"
        for key in ("sessionId", "cwd", "gitBranch"):
            if key not in self.first and isinstance(rec.get(key), str):
                self.first[key] = rec[key]
        if isinstance(rec.get("version"), str) and rec["version"] not in self.versions:
            self.versions.append(rec["version"])
        uuid = rec.get("uuid")
        if isinstance(uuid, str):
            self.nodes.append([lineno] + [rec.get(k) for k in NODE_FIELDS[1:]])
            digest = _digest(line)
            if uuid in self.seen:
                if self.seen[uuid] == digest:
                    self.acc["exact_duplicate"] += 1
                    self.duplicate_lines.add(lineno)
                    return
                self.acc["conflicting_records"] += 1
            else:
                self.seen[uuid] = digest
        msg = rec.get("message")
        if kind == "user" and isinstance(msg, dict):
            self.acc["mapped"] += 1
            self._user(lineno, rec, msg)
        elif kind == "assistant" and isinstance(msg, dict):
            self.acc["mapped"] += 1
            self._assistant(lineno, rec, msg)
        elif kind == "system" and rec.get("subtype") in SYSTEM_STEPS:
            self.acc["mapped"] += 1
            self._system(lineno, rec)
        elif kind == "attachment":
            self.acc["attachment"] += 1
            self.attachments.append({"line": lineno, "record": rec})
        else:
            self.acc["counted"] += 1
            self.by_type[f"system:{rec.get('subtype')}" if kind == "system" else kind] += 1
        if isinstance(uuid, str) and lineno not in self.kinds:
            self.kinds[lineno] = self._kind(kind, rec)

    @staticmethod
    def _kind(kind: str, rec: Dict[str, Any]) -> str:
        msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
        blocks = msg.get("content") if isinstance(msg.get("content"), list) else []
        types = {b.get("type") for b in blocks if isinstance(b, dict)}
        if kind == "assistant":
            return "call" if "tool_use" in types else "assistant"
        if kind == "user":
            return "result" if "tool_result" in types else "user_text"
        if kind == "system":
            return f"system:{rec.get('subtype')}"
        return kind

    def _note_lost(self, kind: str, rec: Dict[str, Any], mapped: set, msg: Dict[str, Any], msg_keys: set) -> None:
        self.lost.update(f"{kind}.{k}" for k in rec if k not in mapped)
        self.lost.update(f"{kind}.message.{k}" for k in msg if k not in msg_keys)

    def _new_step(self, source: str, message: Any, sx: Dict[str, Any], rec: Dict[str, Any]) -> int:
        step: Dict[str, Any] = {"step_id": len(self.steps) + 1, "source": source, "message": message}
        if isinstance(rec.get("timestamp"), str):
            step["timestamp"] = rec["timestamp"]
        step["extra"] = {EXT: sx}
        self.steps.append(step)
        return len(self.steps) - 1

    def _user(self, lineno: int, rec: Dict[str, Any], msg: Dict[str, Any]) -> None:
        self._note_lost("user", rec, _USER_KEYS, msg, {"role", "content"})
        content = msg.get("content", _ABSENT)
        if isinstance(content, str):
            self.blocks["user:text"] += 1
        if isinstance(content, list):
            for b in content:
                self.blocks["user:" + str(b.get("type") if isinstance(b, dict) else type(b).__name__)] += 1
            results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
            if results:
                tur = rec.get("toolUseResult")
                # the Agent tool names its subagent's id, the Workflow tool its run's folder; nothing else is kept
                links = {k: tur[k] for k in LINK_KEYS if isinstance(tur.get(k), str)} \
                    if isinstance(tur, dict) and len(results) == 1 else {}
                if "toolUseResult" in rec:
                    self.lost.add("user.toolUseResult")
                for b in results:
                    self._result(lineno, rec, b, links)
                rest = [b for b in content if not (isinstance(b, dict) and b.get("type") == "tool_result")]
                if not rest:
                    return
                content = rest
        sx: Dict[str, Any] = {"source_ref": lineno}
        fields = {k: rec[k] for k in ("isMeta", "isCompactSummary", "isVisibleInTranscriptOnly") if rec.get(k)}
        if fields:
            sx["source_fields"] = fields
        if isinstance(content, str):
            message: Any = content
        elif isinstance(content, list):
            message, raw = _parts(content)
            if raw is not _NO_RAW:
                sx["message_raw"] = raw
        elif content is _ABSENT or content is None:
            sx["message_state"] = "missing" if content is _ABSENT else "null"
            message = ""
        else:  # any other value is JSON text in ATIF, and the source value stays in the extension
            sx["message_raw"] = content
            message = json.dumps(content, ensure_ascii=False)
        self._new_step("user", message, sx, rec)

    def _result(self, lineno: int, rec: Dict[str, Any], block: Dict[str, Any], links: Dict[str, str]) -> None:
        self.acc["results"] += 1
        cid = block.get("tool_use_id")
        fields: Dict[str, Any] = {k: rec[k] for k in ("uuid", "parentUuid") if k in rec}
        content = block.get("content", _ABSENT)
        if content is _ABSENT or content is None:
            fields["content_state"] = "missing" if content is _ABSENT else "null"
            content = ""
        elif isinstance(content, list):
            content, raw = _parts(content)
            if raw is not _NO_RAW:
                fields["content_raw"] = raw
        elif not isinstance(content, str):
            fields["content_raw"] = content
            content = json.dumps(content, ensure_ascii=False)
        if "is_error" in block:
            fields["is_error"] = block["is_error"]
            if block["is_error"] is True:
                fields["error"] = True
        other = {k: v for k, v in block.items() if k not in ("type", "tool_use_id", "content", "is_error")}
        if other:
            fields["source_fields"] = other
        fields.update(links)
        if self.last_agent is None:  # no message has started, so no call can pair: a step of its own
            self.acc["unpaired_results"] += 1
            if cid is not None:
                fields["tool_call_id"] = cid
            sx: Dict[str, Any] = {"source_ref": lineno, "role": "tool"}
            for key, marker in (("content_raw", "message_raw"), ("content_state", "message_state")):
                if key in fields:
                    sx[marker] = fields.pop(key)
            if fields:
                sx["source_fields"] = fields
            self._new_step("user", content, sx, rec)
            return
        # paired after the whole file is read: the record holding the call can come after its result
        self.pending.append((lineno, cid, content, fields, self.last_agent))

    def _place_results(self) -> None:
        placed: Dict[int, List[Tuple[int, Optional[str], Any, Dict[str, Any]]]] = {}
        for lineno, cid, content, fields, last_agent in self.pending:
            target = self.call_at.get(cid) if isinstance(cid, str) else None
            if target is not None and target[1] < lineno:
                index, paired = target[0], cid
                self.acc["paired"] += 1
                self.answered[cid] += 1
            else:
                index, paired = last_agent, None
                self.acc["unpaired_results"] += 1
                if cid is not None:
                    fields["tool_call_id"] = cid  # kept, though it matches no call of an earlier message
            placed.setdefault(index, []).append((lineno, paired, content, fields))
        for index, entries in placed.items():
            step = self.steps[index]
            sx = step["extra"][EXT]
            results = step.setdefault("observation", {"results": []})["results"]
            for lineno, paired, content, fields in sorted(entries, key=lambda e: e[0]):
                results.append({"source_call_id": paired, "content": content})
                sx.setdefault("result_refs", []).append(lineno)
                sx.setdefault("result_fields", []).append(fields or None)
                for key in LINK_KEYS:
                    if key in fields:
                        self.links.append((index, len(results) - 1, key, fields[key]))

    def _assistant(self, lineno: int, rec: Dict[str, Any], msg: Dict[str, Any]) -> None:
        self._note_lost("assistant", rec, _ASSISTANT_KEYS, msg,
                        {"id", "role", "type", "model", "content", "usage", "stop_reason", "stop_sequence"})
        mid = msg.get("id")
        if isinstance(mid, str) and mid in self.msg_step:
            index = self.msg_step[mid]
            gathered = self.parts[index]
            gathered["lines"].append(lineno)
        else:
            index = self._new_step("agent", "", {"source_ref": lineno}, rec)
            gathered = self.parts[index] = {"lines": [lineno], "content": [], "thinking": [], "usage": None,
                                            "model": None, "fields": {}}
            if isinstance(mid, str):
                self.msg_step[mid] = index
            self.last_agent = index
        if gathered["model"] is None and isinstance(msg.get("model"), str):
            gathered["model"] = msg["model"]
        if isinstance(msg.get("usage"), dict):  # the last record carries the final usage when a message's records differ
            gathered["usage"] = msg["usage"]
        for key in ("stop_reason", "stop_sequence"):
            if msg.get(key) is not None:
                gathered["fields"][key] = msg[key]
        if "requestId" in rec and "requestId" not in gathered["fields"]:
            gathered["fields"]["requestId"] = rec["requestId"]
        content = msg.get("content")
        blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}] if isinstance(content, str) else [])
        first_line = gathered["lines"][0]
        for b in blocks:
            kind = b.get("type") if isinstance(b, dict) else None
            self.blocks["assistant:" + str(kind if kind is not None else type(b).__name__)] += 1
            if kind == "tool_use":
                self._call(index, lineno, first_line, b)
            elif kind == "thinking" and isinstance(b.get("thinking"), str):
                gathered["thinking"].append(b["thinking"])
                self.lost.update(f"thinking.{k}" for k in b if k not in ("type", "thinking"))
            else:
                gathered["content"].append(b)

    def _call(self, index: int, lineno: int, first_line: int, block: Dict[str, Any]) -> None:
        self.acc["calls"] += 1
        step = self.steps[index]
        calls = step.setdefault("tool_calls", [])
        cid = block.get("id")
        cx: Dict[str, Any] = {"source_ref": lineno}
        if not isinstance(cid, str) or not cid or cid in self.call_at:
            if isinstance(cid, str) and cid in self.call_at:
                self.acc["conflicting_calls"] += 1
            if cid is not None:
                cx["source_id"] = cid
            cid = f"ap-{step['step_id']}-{len(calls)}"
            cx["id_generated"] = True
            self.acc["unanswerable_calls"] += 1  # results name the source id, which pairs with the first call
        else:
            self.call_at[cid] = (index, first_line)
        if "input" not in block:
            args: Dict[str, Any] = {}
            cx["arguments_state"] = "missing"
        else:
            args, raw = _parse_arguments(block["input"])
            if raw is not _NO_RAW:
                cx["arguments_raw"] = raw
                if raw is not None:
                    cx["arguments_state"] = "unparsed"
        fname = block.get("name")
        if not isinstance(fname, str):
            cx["function_name_raw"] = fname
        other = {k: v for k, v in block.items() if k not in ("type", "id", "name", "input")}
        if other:
            cx["source_fields"] = other
        calls.append({"tool_call_id": cid, "function_name": fname if isinstance(fname, str) else "", "arguments": args})
        step["extra"][EXT].setdefault("calls", {})[cid] = cx

    def _system(self, lineno: int, rec: Dict[str, Any]) -> None:
        sx: Dict[str, Any] = {"source_ref": lineno,
                              "source_fields": {k: v for k, v in rec.items() if k not in ENVELOPE and k != "content"}}
        content = rec.get("content", _ABSENT)
        if isinstance(content, str):
            message = content
        else:
            sx["message_state"] = "missing" if content is _ABSENT else "null"
            message = ""
        if rec.get("logicalParentUuid") is not None:
            sx["source_fields"]["logicalParentUuid"] = rec["logicalParentUuid"]
        self._new_step("system", message, sx, rec)

    # -- assembling ----------------------------------------------------------------

    def _finish_agent_steps(self) -> List[str]:
        models: List[str] = []
        for index, gathered in self.parts.items():
            step = self.steps[index]
            sx = step["extra"][EXT]
            content = gathered["content"]
            if not content:
                sx["message_state"] = "missing"
            elif len(content) == 1 and isinstance(content[0], dict) and set(content[0]) == {"type", "text"} \
                    and content[0].get("type") == "text" and isinstance(content[0].get("text"), str):
                step["message"] = content[0]["text"]
            else:
                step["message"], raw = _parts(content)
                if raw is not _NO_RAW:
                    sx["message_raw"] = raw
            if gathered["thinking"]:
                step["reasoning_content"] = "\n\n".join(gathered["thinking"])
            if gathered["model"] is not None:
                step["model_name"] = gathered["model"]
                if gathered["model"] not in models:
                    models.append(gathered["model"])
            usage = gathered["usage"]
            fields = dict(gathered["fields"])
            if usage is not None:
                metrics = _metrics(usage)
                if metrics:
                    step["metrics"] = metrics
                fields["usage"] = usage
            if len(gathered["lines"]) > 1:
                fields["lines"] = gathered["lines"]
            if fields:
                sx["source_fields"] = fields
            # keep ATIF's key order: message fields before tool calls and the observation
            for key in ("tool_calls", "observation", "extra"):
                if key in step:
                    step[key] = step.pop(key)
        return models

    def _branches(self) -> List[Dict[str, Any]]:
        children: Dict[Any, Dict[str, int]] = {}
        for row in self.nodes:
            line, uuid, parent = row[0], row[1], row[2]
            # an attachment beside a message (newer files hang hook output beside a tool result) is no alternative
            if line in self.duplicate_lines or not isinstance(parent, str) or row[4] == "attachment":
                continue
            children.setdefault(parent, {}).setdefault(uuid, line)
        out = []
        for parent, kids in children.items():
            if len(kids) < 2:
                continue
            kinds = [self.kinds.get(line, "") for line in kids.values()]
            if all(k in ("call", "result") for k in kinds) and "call" in kinds and "result" in kinds:
                continue  # parallel calls and their results: structure, not alternatives
            if sum(k == "user_text" for k in kinds) >= 2:
                label = "user texts"
            elif "system:api_error" in kinds and any(k in ("assistant", "call") for k in kinds):
                label = "assistant and api_error"
            else:
                label = "other"
            out.append({"parent": parent, "children": list(kids), "label": label})
        return out

    def trajectory(self, trajectory_id: Optional[str], ext: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        self._place_results()
        models = self._finish_agent_steps()
        if not self.steps:  # ATIF requires one step; a file of bookkeeping records has no message to show
            self.steps.append({"step_id": 1, "source": "system", "message": "",
                               "extra": {EXT: {"message_state": "missing", "empty_source": True}}})
        calls_without_result = sum(1 for cid in self.call_at if not self.answered[cid]) + self.acc["unanswerable_calls"]
        accounting: Dict[str, Any] = {"lines": self.acc["lines"]}
        accounting.update({c: self.acc[c] for c in CATEGORIES})
        accounting.update({"counted_by_type": dict(self.by_type), "blocks": dict(self.blocks),
                           "calls": self.acc["calls"], "results": self.acc["results"], "paired": self.acc["paired"],
                           "calls_without_result": calls_without_result,
                           "unpaired_results": self.acc["unpaired_results"],
                           "conflicting_records": self.acc["conflicting_records"],
                           "conflicting_calls": self.acc["conflicting_calls"]})
        dataset: Dict[str, Any] = {"versions": self.versions}
        dataset.update({k: self.first[k] for k in ("sessionId", "cwd", "gitBranch") if k in self.first})
        dataset.update({"accounting": accounting, "nodes": {"fields": NODE_FIELDS, "rows": self.nodes},
                        "branches": self._branches(), "attachments": self.attachments})
        session_id = self.first.get("sessionId") or _stem(self.path)
        tid = trajectory_id or session_id
        agent: Dict[str, Any] = {"name": "claude-code", "version": self.versions[0] if self.versions else "unknown"}
        if len(models) == 1:
            agent["model_name"] = models[0]
        ext = dict(ext or {})
        source = dict(ext.get("source") or {"format": name, "file": self.path.name, "record_key": tid,
                                            "reader": name, "reader_version": READER_VERSION})
        source["lost_fields"] = sorted(self.lost)
        ext["source"] = source
        ext["dataset"] = dict(ext.get("dataset") or {}, **dataset)
        ext["version"] = EXT_VERSION
        return {"schema_version": ATIF_VERSION, "session_id": session_id, "trajectory_id": tid, "agent": agent,
                "steps": self.steps, "extra": {EXT: ext}}


def _metrics(usage: Dict[str, Any]) -> Dict[str, Any]:
    """ATIF metrics from Anthropic usage: prompt tokens count cache reads and writes."""
    out: Dict[str, Any] = {}
    inp, cached, written = (usage.get(k) for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    if _ints(inp):
        out["prompt_tokens"] = inp + (cached if _ints(cached) else 0) + (written if _ints(written) else 0)
    if _ints(usage.get("output_tokens")):
        out["completion_tokens"] = usage["output_tokens"]
    if _ints(cached):
        out["cached_tokens"] = cached
    return out


def _stem(path: Path) -> str:
    return path.name[: -len(".jsonl")] if path.name.endswith(".jsonl") else path.stem


def _subagent_files(path: Path) -> Tuple[List[Path], int]:
    """(agent-*.jsonl files under ``<session>/subagents/``, count of other files there)."""
    base = path.parent / _stem(path) / "subagents"
    if not base.is_dir():
        return [], 0
    files = sorted(p for p in base.rglob("*") if p.is_file())
    agents = [p for p in files if AGENT_FILE.match(p.name)]
    return agents, len(files) - len(agents)


def session_to_trajectory(path: Path, trajectory_id: Optional[str] = None,
                          ext: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ATIF trajectory of one session file, with its subagent files nested under the trajectories that start them."""
    main = _File(path).read()
    root = main.trajectory(trajectory_id, ext)
    agents, other = _subagent_files(path)
    if not agents and not other and not main.links:
        return root
    base = path.parent / _stem(path)
    subs: Dict[str, Tuple[_File, Dict[str, Any]]] = {}
    files: Dict[str, str] = {}
    runs: Dict[str, List[str]] = {}  # workflow run folder -> ids of the agent files in it
    shared = Counter(AGENT_FILE.match(p.name).group(1) for p in agents)
    for p in agents:
        rel = p.relative_to(base).as_posix()
        agent_id = AGENT_FILE.match(p.name).group(1)
        while agent_id in subs:  # two files with one id in different folders: each gets its own trajectory id
            agent_id += "'"
        f = _File(p).read()
        sub_ext = {"source": {"format": name, "file": rel, "record_key": agent_id,
                              "reader": name, "reader_version": READER_VERSION}}
        subs[agent_id] = (f, f.trajectory(agent_id, sub_ext))
        files[agent_id] = rel
        parts = rel.split("/")
        if len(parts) == 4 and parts[1] == "workflows":  # subagents/workflows/<run>/agent-<id>.jsonl
            runs.setdefault(parts[2], []).append(agent_id)
    holders = [(None, main, root)] + [(agent_id, f, traj) for agent_id, (f, traj) in subs.items()]
    placed: Dict[str, Optional[str]] = {}  # subagent id -> id of the subagent holding it (None: the root)
    missing: Dict[str, set] = {"agentId": set(), "runId": set()}
    ambiguous: set = set()
    referenced: set = set()

    def under(holder: Optional[str], agent_id: str) -> bool:
        while holder is not None:
            if holder == agent_id:
                return True
            holder = placed.get(holder)
        return False

    for holder, f, traj in holders:
        for step_index, result_index, key, value in f.links:
            if key == "agentId" and shared[value] > 1:  # neither file is guessed to be the named one
                ambiguous.add(value)
                continue
            named = ([value] if value in subs else []) if key == "agentId" else runs.get(value, [])
            if not named:
                missing[key].add(value)
                continue
            res = traj["steps"][step_index]["observation"]["results"][result_index]
            res.setdefault("subagent_trajectory_ref", []).extend({"trajectory_id": a} for a in named)
            referenced.update(named)
            for agent_id in named:  # a file that names itself or an ancestor stays where it is
                if agent_id not in placed and not under(holder, agent_id):
                    placed[agent_id] = holder
    for agent_id, (_f, sub) in subs.items():
        holder = placed.get(agent_id)
        parent = root if holder is None else subs[holder][1]
        parent.setdefault("subagent_trajectories", []).append(sub)
    root["extra"][EXT]["dataset"]["subagents"] = {
        "files": files, "missing_agent_ids": sorted(missing["agentId"]), "missing_runs": sorted(missing["runId"]),
        "ambiguous_agent_ids": sorted(ambiguous), "unreferenced": sorted(set(subs) - referenced), "other_files": other}
    return root


def read(source: Any, **options: Any) -> Iterator[Run]:
    if options:
        raise TypeError(f"{name} reader takes no options, got {sorted(options)}")
    if not isinstance(source, Path):
        raise TypeError(f"{name} reads a session file, got {type(source).__name__}")
    yield Run(session_to_trajectory(source), TRAJECTORY)
