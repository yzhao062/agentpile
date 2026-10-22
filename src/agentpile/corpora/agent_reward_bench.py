"""AgentRewardBench (McGill-NLP/agent-reward-bench): browser-agent runs with expert labels.

One run per file ``cleaned/<benchmark>/<agent>/<experiment>/<task_id>.json``
(1,302 files over WebArena, VisualWebArena, WorkArena and AssistantBench), joined
to the rows of ``data/annotations.csv`` on (benchmark, agent, task id). The
screenshots and the automatic judge outputs are not read. A run file holds a
header plus one record per BrowserGym step, and a record carries the page the
agent saw and the action it chose from it. Records are not loaded whole: a file
can reach 265 MB because every record holds three browser-state dumps, so the
reader parses one record at a time and drops the dumps (``source.lost_fields``).

- Steps. A system step from the system prompt, a user step from ``goal``, then
  for each record with an action its model inputs and one agent step. The
  inputs are the chat messages before the first reply: every user prompt is a
  user step (it carries that call's page), and a system message is a new system
  step only when it differs from the latest one; the agent step's ``inputs``
  maps each input ``chat_message`` to the step holding it. Content maps as in
  the message reader: text parts stay text parts, any other part (the image
  parts, whose url the dataset replaced with "REMOVED") becomes a marked text
  stand-in, and the whole source content is kept in ``message_raw``, which the
  event view returns. The agent step's message is ``reasoning`` (when
  the model gave only an action, the source puts the raw reply there). Its
  single call is the verb of the action string. Arguments come from a literal
  parse of the argument list: positional ones are named ``arg0``, ``arg1``, ...
  and keyword ones keep their name. ``arguments_raw`` is always the whole action
  string, and text after the closing parenthesis (a ``#`` comment, or an HTML
  comment in one run) goes to the call's ``source_fields``. A string that does
  not parse leaves ``arguments`` empty, sets ``arguments_parsed`` false and marks the call
  ``arguments_state: "unparsed"``; only then does a model span of the OTLP export show the action
  string instead of the parsed arguments.
- Results. The source has no call ids and no result field. By BrowserGym's step
  semantics the page after action ``i`` is the observation of record ``i + 1``,
  so that record's AXTree text is the result of call ``i`` and its url, tabs,
  focused element, screenshot path and ``last_action_error`` go to
  ``result_fields`` with ``paired_by: "next_record"``. Call ids are generated
  (``id_generated``). A non-empty ``last_action_error`` is the call's ``error``.
  The observation of record 0, the page before any action, has no call to attach
  to and sits in the user step's ``source_fields.observation``. A last action
  with no later record (a crashed run) has no result.
- A record without an action contributes only its observation, unless the agent
  replied (a non-empty ``chat_messages``): then it is an agent step without a
  call (``source_fields.action``), as in the two runs whose reply never parsed.
  Chat turns from the first reply on, when there is more than that reply (the
  retries of the 26 records whose first reply did not parse), are kept in
  ``source_fields.chat_messages_tail``.
- Labels. Every annotation row gives four run-level labels (success, side
  effect, optimality, looping) with the CSV value as written and
  ``label_source`` ``expert:<annotator>``. 106 runs have two rows; the rows stay
  separate. ``outcome.success`` follows the first row in CSV order (Successful
  is true, Unsuccessful false, Unsure null), ``outcome.score`` is the
  environment's ``cum_reward``, and ``outcome.raw`` keeps every annotator's value.
- The action space is prose in the prompt, so ``tools`` is empty. The run id is
  ``<benchmark>/<agent>/<task id>``, the task id being the file name as the CSV
  spells it: the same VisualWebArena task has ``visualwebarena.resized.N`` for
  two agents and ``visualwebarena.N`` for the third, and nothing here merges them.
"""

from __future__ import annotations

import ast
import codecs
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ..formats.messages import _NO_RAW, _parts
from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run
from ._base import Dataset

ANNOTATIONS = "data/annotations.csv"
BENCHMARKS = ("webarena", "visualwebarena", "workarena", "assistantbench")
LABEL_FIELDS = ("trajectory_success", "trajectory_side_effect", "trajectory_optimality", "trajectory_looping")
DROPPED = ("axtree_obj", "bounding_boxes", "extra_element_properties", "axtree_pruned")
LOST_FIELDS = [
    "steps[].axtree_obj", "steps[].bounding_boxes", "steps[].extra_element_properties", "steps[].axtree_pruned",
]
_OBSERVATION_KEYS = ("num", "url", "open_pages_urls", "focused_element", "last_action_error", "screenshot_path")
_SUCCESS = {"successful": True, "unsuccessful": False}
_VERB = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


# -- streaming JSON reader ------------------------------------------------------

_DECODER = json.JSONDecoder()
_SPACE = " \t\r\n"


class _Stream:
    """A UTF-8 file read in chunks, handing out one JSON value at a time."""

    def __init__(self, fh, chunk: int):
        self.fh, self.chunk = fh, chunk
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buf, self.pos, self.eof = "", 0, False

    def more(self, want: int = 0) -> None:
        raw = self.fh.read(max(self.chunk, want))
        text = self.decoder.decode(raw, final=not raw)
        self.eof = not raw
        self.buf, self.pos = self.buf[self.pos:] + text, 0

    def peek(self) -> str:
        """The next non-space character, or "" at the end of the file."""
        while True:
            while self.pos < len(self.buf) and self.buf[self.pos] in _SPACE:
                self.pos += 1
            if self.pos < len(self.buf):
                return self.buf[self.pos]
            if self.eof:
                return ""
            self.more()

    def take(self, *chars: str) -> str:
        got = self.peek()
        if got not in chars:
            raise ValueError(f"expected {' or '.join(map(repr, chars))}, found {got!r}")
        self.pos += 1
        return got

    def value(self) -> Any:
        self.peek()
        while True:
            try:
                obj, end = _DECODER.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.more(len(self.buf) - self.pos)  # double the window until the value fits
                continue
            if end >= len(self.buf) and not self.eof:  # a number may continue in the next chunk
                self.more()
                continue
            self.pos = end
            return obj


def read_run_file(path: Path, drop: Iterable[str] = DROPPED, chunk: int = 1 << 22) -> Tuple[Dict[str, Any], List[Any]]:
    """The header keys and the ``steps`` records of one run file.

    Memory stays near one record instead of the whole file. ``drop`` names the
    record keys deleted as soon as a record is parsed.
    """
    head: Dict[str, Any] = {}
    steps: List[Any] = []
    drop = tuple(drop)
    with Path(path).open("rb") as fh:
        stream = _Stream(fh, chunk)
        stream.take("{")
        if stream.peek() == "}":
            return head, steps
        while True:
            key = stream.value()
            stream.take(":")
            if key == "steps" and stream.peek() == "[":
                stream.take("[")
                if stream.peek() == "]":
                    stream.take("]")
                else:
                    while True:
                        record = stream.value()
                        if isinstance(record, dict):
                            for name in drop:
                                record.pop(name, None)
                        steps.append(record)
                        if stream.take(",", "]") == "]":
                            break
            else:
                head[key] = stream.value()
            if stream.take(",", "}") == "}":
                break
    return head, steps


# -- action strings -------------------------------------------------------------

def _close_paren(text: str, start: int) -> int:
    """Index of the parenthesis closing the one opened just before ``start``, or -1."""
    depth, i, n = 1, start, len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"":
            quote = text[i:i + 3] if text[i:i + 3] in ("'''", '"""') else ch
            i += len(quote)
            while i < n and not text.startswith(quote, i):
                i += 2 if text[i] == "\\" else 1
            if i >= n:
                return -1
            i += len(quote)
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i if ch == ")" else -1
        i += 1
    return -1


def parse_action(action: str) -> Optional[Tuple[str, Dict[str, Any], bool, str]]:
    """``(verb, arguments, parsed, trailing)`` of a BrowserGym action string, or None without a leading call.

    Only literals are evaluated. ``parsed`` is false when the argument list is
    unbalanced or holds anything but literals; ``arguments`` is then empty.
    ``trailing`` is the text after the call's closing parenthesis.
    """
    match = _VERB.match(action)
    if match is None:
        return None
    verb = match.group(1)
    end = _close_paren(action, match.end())
    if end < 0:
        return verb, {}, False, ""
    trailing = action[end + 1:].strip()
    arguments: Dict[str, Any] = {}
    try:
        call = ast.parse(f"_({action[match.end():end]})", mode="eval").body
        if not isinstance(call, ast.Call):
            raise ValueError("not a call")
        for k, node in enumerate(call.args):
            if isinstance(node, ast.Starred):
                raise ValueError("starred argument")
            arguments[f"arg{k}"] = ast.literal_eval(node)
        for kw in call.keywords:
            if kw.arg is None or kw.arg in arguments:
                raise ValueError("unpackable or repeated keyword")
            arguments[kw.arg] = ast.literal_eval(kw.value)
        json.dumps(arguments)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return verb, {}, False, trailing
    return verb, arguments, True, trailing


# -- run records ----------------------------------------------------------------

def _text(content: Any) -> Tuple[str, Any]:
    """A message for ATIF, plus the raw value when it was not a string."""
    if isinstance(content, str):
        return content, None
    return json.dumps(content, ensure_ascii=False), content


def _observation(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """(AXTree text, the remaining observation fields) of one record."""
    fields = {k: record[k] for k in _OBSERVATION_KEYS if k in record}
    axtree = record.get("axtree")
    if "axtree" not in record or axtree is None:
        fields["content_state"] = "missing" if "axtree" not in record else "null"
        return "", fields
    if not isinstance(axtree, str):
        fields["content_raw"] = axtree
        return json.dumps(axtree, ensure_ascii=False), fields
    return axtree, fields


def _chat(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    chat = record.get("chat_messages")
    return [m for m in chat if isinstance(m, dict)] if isinstance(chat, list) else []


def _first_reply(chat: List[Dict[str, Any]]) -> int:
    """Index of the first assistant turn; the turns before it are the model call's inputs."""
    return next((k for k, m in enumerate(chat) if m.get("role") == "assistant"), len(chat))


def _chat_tail(chat: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Turns not carried by the input steps and the one accepted reply."""
    rest = chat[_first_reply(chat):]
    return None if [m.get("role") for m in rest] == ["assistant"] else rest or None


def _content(content: Any) -> Tuple[Any, Any]:
    """An input message for ATIF, plus the source content when ATIF holds a stand-in (``_NO_RAW`` otherwise)."""
    if isinstance(content, list):
        return _parts(content)
    message, raw = _text(content)
    return message, _NO_RAW if raw is None else raw


def _last_reply(chat: List[Dict[str, Any]]) -> Any:
    replies = [m.get("content") for m in chat if m.get("role") == "assistant"]
    return replies[-1] if replies else None


def _annotations(root: Path) -> Dict[Tuple[str, str, str], List[Tuple[int, Dict[str, str]]]]:
    """(benchmark, model_name, task_id) -> [(row index, row)] in CSV order."""
    path = root / ANNOTATIONS
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the annotation rows are the labels of every run")
    rows: Dict[Tuple[str, str, str], List[Tuple[int, Dict[str, str]]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            rows.setdefault((row["benchmark"], row["model_name"], row["task_id"]), []).append((i, row))
    return rows


def _benchmarks(benchmark: Any) -> Optional[Tuple[str, ...]]:
    if benchmark is None:
        return None
    wanted = (benchmark,) if isinstance(benchmark, str) else tuple(benchmark)
    unknown = [b for b in wanted if b not in BENCHMARKS]
    if unknown:
        raise ValueError(f"benchmark must be from {list(BENCHMARKS)}, got {unknown}")
    return wanted


class AgentRewardBench(Dataset):
    name = "agent_reward_bench"
    format = "agent_reward_bench-json"
    incremental = True
    upfront = (ANNOTATIONS,)  # joined to every run, so read before the first

    def records(self, root: Path, benchmark: Any = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"agent_reward_bench takes benchmark= only, got {sorted(options)}")
        wanted = _benchmarks(benchmark)
        rows = _annotations(root)
        for path in self.paths(root, "cleaned/*/*/*/*.json", key=lambda p: p.as_posix()):
            rel = path.relative_to(root).as_posix()
            bench, agent_dir = rel.split("/")[1:3]
            if wanted is not None and bench not in wanted:
                continue
            head, steps = read_run_file(self.ready(path))
            raw = {"head": head, "steps": steps, "annotations": rows.get((bench, agent_dir, path.stem), [])}
            yield (f"{bench}/{agent_dir}/{path.stem}", raw,
                   {"file": rel, "config": bench, "split": "full", "task_id": path.stem})

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        head, records, annotations = raw["head"], raw["steps"], raw["annotations"]
        agent = head.get("agent") if isinstance(head.get("agent"), str) else "unknown"
        steps: List[Dict[str, Any]] = []
        unpaired: List[Dict[str, Any]] = []

        def add(source: str, message: Any, sx: Dict[str, Any]) -> Dict[str, Any]:
            step = {"step_id": len(steps) + 1, "source": source, "message": message, "extra": {EXT: sx}}
            steps.append(step)
            return step

        def add_input(j: int, k: int, m: Dict[str, Any]) -> Dict[str, Any]:
            role = m.get("role")
            sx: Dict[str, Any] = {"source_ref": {"record": j, "chat_message": k}}
            if role not in ("system", "user"):
                sx["role"] = role
            if m.get("content") is None:
                message = ""
                sx["message_state"] = "missing" if "content" not in m else "null"
            else:
                message, kept = _content(m["content"])
                if kept is not _NO_RAW:
                    sx["message_raw"] = kept
            return add("system" if role == "system" else "user", message, sx)

        # the system prompt comes from the first record the agent was queried on; a later system
        # message equal to the latest system step reuses it instead of repeating it
        system_step: Optional[Dict[str, Any]] = None
        system_content: Any = None
        first = next(((j, c) for j, c in ((j, _chat(r)) for j, r in enumerate(records) if isinstance(r, dict)) if c), None)
        if first is not None:
            j0, chat0 = first
            k0 = next((k for k, m in enumerate(chat0[:_first_reply(chat0)]) if m.get("role") == "system"), None)
            if k0 is not None:
                system_step, system_content = add_input(j0, k0, chat0[k0]), chat0[k0].get("content")

        goal_step = None
        if isinstance(head.get("goal"), (str, list, dict)):
            message, rawval = _text(head["goal"])
            sx = {"source_ref": "goal", "source_fields": {} if rawval is None else {"goal": rawval}}
            goal_step = add("user", message, sx)

        pending: Optional[Tuple[Dict[str, Any], str]] = None  # the call whose result is the next record
        for j, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            if pending is not None:
                step, cid = pending
                content, fields = _observation(rec)
                fields["paired_by"] = "next_record"
                step["observation"] = {"results": [{"source_call_id": cid, "content": content}]}
                sx = step["extra"][EXT]
                sx["result_refs"], sx["result_fields"] = [j], [fields]
                if isinstance(rec.get("last_action_error"), str) and rec["last_action_error"]:
                    sx["calls"][cid]["error"] = rec["last_action_error"]
            elif j == 0 and goal_step is not None:
                content, fields = _observation(rec)
                goal_step["extra"][EXT]["source_fields"]["observation"] = dict(fields, axtree=content)
            else:
                content, fields = _observation(rec)
                unpaired.append(dict(fields, record=j, axtree=content))
            pending = None

            action, reasoning, chat = rec.get("action"), rec.get("reasoning"), _chat(rec)
            has_call = isinstance(action, str) and bool(action.strip())
            if not (has_call or isinstance(reasoning, str) or chat):
                continue  # the final record: its observation was all it carried
            inputs = []
            for k, m in enumerate(chat[:_first_reply(chat)]):
                if (m.get("role") == "system" and system_step is not None
                        and m.get("content") is not None and m["content"] == system_content):
                    held = system_step
                else:
                    held = add_input(j, k, m)
                    if m.get("role") == "system":
                        system_step, system_content = held, m.get("content")
                inputs.append({"chat_message": k, "step": held["step_id"]})
            sx = {"source_ref": j, "actor": agent}
            if inputs:
                sx["inputs"] = inputs
            sf = {}
            if "num" in rec:
                sf["num"] = rec["num"]
            if rec.get("stats") is not None:
                sf["stats"] = rec["stats"]
            if isinstance(reasoning, str):
                message = reasoning
            else:
                reply = _last_reply(chat)
                if reply is not None:
                    message, rawval = _text(reply)
                    sf["message_from"] = "last_assistant_reply"
                    if rawval is not None:
                        sf["reply_raw"] = rawval
                else:
                    message = ""
                    sx["message_state"] = "null" if "reasoning" in rec else "missing"
            tail = _chat_tail(chat)
            if tail:
                sf["chat_messages_tail"] = tail
            step = add("agent", message, sx)
            if has_call:
                cid = f"ap-{step['step_id']}-0"
                parsed = parse_action(action)
                verb, arguments, ok, trailing = parsed if parsed is not None else ("", {}, False, "")
                cx: Dict[str, Any] = {"id_generated": True, "arguments_raw": action}
                cf: Dict[str, Any] = {}
                if parsed is None:
                    cx["function_name_raw"] = action
                if not ok:
                    cf["arguments_parsed"] = False
                    cx["arguments_state"] = "unparsed"  # {} stands in for the action string; a parsed one has no mark
                if trailing:
                    cf["trailing"] = trailing
                if cf:
                    cx["source_fields"] = cf
                step["tool_calls"] = [{"tool_call_id": cid, "function_name": verb, "arguments": arguments}]
                sx["calls"] = {cid: cx}
                pending = (step, cid)
            else:
                sf["action"] = action
            if sf:
                sx["source_fields"] = sf

        if not steps:  # ATIF requires one step; a run file without records or goal is a missing message
            add("system", "", {"message_state": "missing", "empty_source": True})

        rows = [row for _, row in annotations]
        labels = [{"name": field, "target": {"kind": "run", "address": {"trajectory": key}}, "value": row[field],
                   "label_source": f"expert:{(row.get('annotator_name') or '').strip() or 'unknown'}",
                   "source_ref": {"file": ANNOTATIONS, "row": i, "field": field,
                                  "annotator_name": row.get("annotator_name")}}
                  for i, row in annotations for field in LABEL_FIELDS if field in row]
        verdicts = [{"annotator_name": row.get("annotator_name"), "value": row.get("trajectory_success")}
                    for row in rows]
        first = (verdicts[0]["value"] or "").strip().lower() if verdicts else ""
        info = head.get("summary_info") if isinstance(head.get("summary_info"), dict) else {}
        reward = info.get("cum_reward")
        outcome = {"success": _SUCCESS.get(first),
                   "score": reward if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None,
                   "raw": {"trajectory_success": verdicts, "success_from": "first row of data/annotations.csv",
                           "cum_reward": reward, "cum_raw_reward": info.get("cum_raw_reward")}}

        dataset = {k: v for k, v in head.items() if k not in ("agent", "model") and (k != "goal" or goal_step is None)}
        dataset["annotation_rows"] = [dict(row, row=i) for i, row in annotations]
        if unpaired:
            dataset["unpaired_observations"] = unpaired
        source = self.source(key, where)
        source["lost_fields"] = list(LOST_FIELDS)
        model = head.get("model") if isinstance(head.get("model"), str) else None
        ext = {
            "version": EXT_VERSION,
            "source": source,
            "task": {"id": where["task_id"], "text": head["goal"] if isinstance(head.get("goal"), str) else None},
            "outcome": outcome,
            "labels": labels,
            "agents": [{"id": agent, "name": agent, "instructions": None, "model": model, "tools": [], "handoffs": []}],
            "tools": {},
            "dataset": dataset,
        }
        atif_agent: Dict[str, Any] = {"name": agent, "version": "unknown"}
        if model is not None:
            atif_agent["model_name"] = model
        traj = {"schema_version": ATIF_VERSION, "session_id": key, "trajectory_id": key, "agent": atif_agent,
                "steps": steps, "extra": {EXT: ext}}
        return Run(traj, TRAJECTORY)


dataset = AgentRewardBench()
