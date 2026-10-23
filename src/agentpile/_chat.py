"""Chat rows: each trajectory as an OpenAI-style message list with its tools.

``to_chat(runs)`` gives one ``{"messages": [...], "tools": [...]}`` row per trajectory, the shape that OpenAI
fine-tuning, Hugging Face chat templates and message-list auditors take. A row holds only what the record holds.
Where the chat shape needs a value the source lacks (a call's arguments, a result's text), the row carries a marked
or empty value, and the report counts it; prompts, schemas and results the source lacks are not made up. The
standard library is enough.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from ._tables import _iter_runs
from .record import CONFIG, MISSING, Run, _ext

SIMULATOR_NAME = "user_simulator"  # the roster name tau2-bench's reader gives its simulated user; its turns are user context
COUNTS = ("unparsed_arguments", "null_arguments", "missing_arguments", "results_moved", "results_without_call",
          "unanswered_calls", "projected", "content_parts", "content_json", "null_content", "missing_content",
          "tools_skipped")
TEXT_PARTS = ("text", "input_text", "output_text")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text(value: Any, counts: Dict[str, int]) -> str:
    """Message text: a string as it is, content parts as their text, ``""`` for null or absent, else JSON text."""
    if isinstance(value, str):
        return value
    if value is MISSING:
        counts["missing_content"] += 1
        return ""
    if value is None:
        counts["null_content"] += 1
        return ""
    if isinstance(value, list):
        counts["content_parts"] += 1
        return "\n".join(_part_text(part) for part in value)
    counts["content_json"] += 1
    return _json_text(value)


def _part_text(part: Any) -> str:
    """A text part as its text; any other part (an image, a file) as JSON text, so nothing is dropped."""
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and part.get("type") in TEXT_PARTS and isinstance(part.get("text"), str):
        return part["text"]
    return _json_text(part)


def _arguments(call: Dict[str, Any], cx: Dict[str, Any], counts: Dict[str, int]) -> Dict[str, Any]:
    """A call's arguments as a chat row shows them: the parsed object; ``{"_raw": ...}`` with the source value for
    arguments that did not parse or were null; ``{}`` where the source recorded none."""
    state = cx.get("arguments_state")
    if state == "missing" or "arguments_stand_in" in cx:
        counts["missing_arguments"] += 1
        return {}
    if "arguments_raw" in cx and cx["arguments_raw"] is None:
        counts["null_arguments"] += 1
        return {"_raw": None}
    if state == "unparsed":
        counts["unparsed_arguments"] += 1
        return {"_raw": copy.deepcopy(cx.get("arguments_raw"))}
    args = call.get("arguments")
    if isinstance(args, dict):
        return copy.deepcopy(args)
    if args is None:  # a native ATIF call without arguments
        counts["missing_arguments"] += 1
        return {}
    counts["unparsed_arguments"] += 1
    return {"_raw": copy.deepcopy(args)}


def _openai_tool(defn: Any) -> Optional[Dict[str, Any]]:
    """A tool definition as an OpenAI function tool, from the OpenAI (nested, flat or legacy), MCP or Anthropic shape;
    None for a definition with no function name."""
    if not isinstance(defn, dict):
        return None
    inner = defn["function"] if isinstance(defn.get("function"), dict) else defn
    if not (isinstance(inner.get("name"), str) and inner["name"]):
        return None
    if defn.get("type") == "function" and inner is not defn:
        return copy.deepcopy(defn)
    fn: Dict[str, Any] = {"name": inner["name"]}
    if isinstance(inner.get("description"), str):
        fn["description"] = inner["description"]
    schema = next((inner[k] for k in ("parameters", "inputSchema", "input_schema") if isinstance(inner.get(k), dict)), None)
    if schema is not None:
        fn["parameters"] = copy.deepcopy(schema)
    if "strict" in inner:  # OpenAI's flat and nested shapes both record it
        fn["strict"] = copy.deepcopy(inner["strict"])
    return {"type": "function", "function": fn}


def _tools(run: Run, counts: Dict[str, int]) -> List[Dict[str, Any]]:
    """The agent's tool definitions, then the definitions of ``run.tools``; a normalized definition seen before is
    left out, and two different ones with one name both stay."""
    defs = (run.record.get("agent") or {}).get("tool_definitions")
    found = list(defs) if isinstance(defs, list) else []
    found += [entry["definition"] for entry in run.tools.values()
              if isinstance(entry, dict) and entry.get("definition") is not None]
    tools: List[Dict[str, Any]] = []
    seen = set()
    for defn in found:
        tool = _openai_tool(defn)
        if tool is None:
            counts["tools_skipped"] += 1
            continue
        key = json.dumps(tool, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            tools.append(tool)
    return tools


def _row(run: Run, perspective: Optional[str], as_json: bool) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """``(row, report entry)`` for one run; the row is None for a config record or a trajectory with no events."""
    counts = dict.fromkeys(COUNTS, 0)
    entry: Dict[str, Any] = {"input": None, "row": None, "id": run.id, "source": copy.deepcopy(run.source),
                             "skipped": None, "assistant_turns": 0, "counts": counts, "messages": []}
    if run.kind == CONFIG:
        entry["skipped"] = "config"
        return None, entry
    timeline = [(step, event) for step, event in run._timeline() if event is not None]
    if not timeline:
        entry["skipped"] = "no events"
        return None, entry
    roster = [agent for agent in run.agents if isinstance(agent, dict)]
    simulators = {agent.get("id") for agent in roster if agent.get("name") == SIMULATOR_NAME} - {None}
    actors = {e.actor for _s, e in timeline if e.role == "agent" and e.actor is not None}
    agents = ({agent.get("id") for agent in roster} | actors) - simulators - {None}
    if perspective is not None and perspective not in agents:
        what = "the simulated user" if perspective in simulators else "not an agent"
        raise ValueError(f"perspective {perspective!r} is {what} of run {run.id!r} (its agents: {sorted(map(str, agents))})")
    named = perspective is None and len(agents) > 1  # assistant turns of a multi-agent run carry the agent's name
    sole = next(iter(agents)) if len(agents) == 1 else None

    def side(role: Optional[str], actor: Optional[str]) -> Tuple[str, Optional[str]]:
        """``(chat role, label)``: "assistant" with its name or None, "system" or "user", or "context" for a turn of
        another agent or the simulator, shown to the assistant as user text that starts with ``[label]``."""
        if role in ("system", "user"):
            return role, None
        if actor in simulators:
            return "context", actor
        if perspective is None or (actor if actor is not None else sole) == perspective:
            return "assistant", actor if named else None
        return "context", actor if actor is not None else "agent"

    # one entry per turn: [message, report item, (message, report item) of each result placed after it]
    turns: List[List[Any]] = []
    calls: Dict[Any, Tuple[Optional[int], int, Optional[str], Optional[str]]] = {}  # latest call per id
    answered = set()
    current: Optional[Tuple[int, int]] = None  # (id of the step, turn index) that later calls of the step join
    seen: Dict[int, int] = {}
    for position, (step, event) in enumerate(timeline):
        here = {"position": position, "address": dict(event.address)}
        if event.kind == "message":
            role, label = side(event.role, event.actor)
            current = None
            if role == "assistant":
                msg: Dict[str, Any] = {"role": "assistant"}
                if label is not None:
                    msg["name"] = label
                msg["content"] = event.content  # text once the turn's calls are known
                turns.append([msg, {"events": [here]}, []])
                current = (id(step), len(turns) - 1)
                continue
            text = _text(event.content, counts)
            if role == "context" and event.actor not in simulators:  # a simulated user's message is plain user text
                counts["projected"] += 1
                text = "[%s] %s" % (label, text)
            turns.append([{"role": "user" if role == "context" else role, "content": text}, {"events": [here]}, []])
        elif event.kind == "call":
            j = seen.get(id(step), 0)
            seen[id(step)] = j + 1
            cx = (_ext(step).get("calls") or {}).get(event.call_id) or {}
            args = _arguments((step.get("tool_calls") or [])[j], cx, counts)
            role, label = side(event.role, event.actor)
            if role != "assistant":
                counts["projected"] += 1
                text = "[%s] tool call%s %s" % (label or role, " " + event.name if event.name else "", _json_text(args))
                turns.append([{"role": "user", "content": text}, {"events": [here]}, []])
                calls[event.call_id] = (None, 0, label or role, event.name)
                current = None
                continue
            if current is None or current[0] != id(step):  # a step whose source had no message
                msg = {"role": "assistant"}
                if label is not None:
                    msg["name"] = label
                msg["content"] = None
                turns.append([msg, {"events": []}, []])
                current = (id(step), len(turns) - 1)
            turn = turns[current[1]]
            turn[0].setdefault("tool_calls", []).append({
                "id": event.call_id, "type": "function",
                "function": {"name": event.name, "arguments": _json_text(args) if as_json else args}})
            turn[1]["events"].append(here)
            calls[event.call_id] = (current[1], len(turn[0]["tool_calls"]) - 1, label, event.name)
        else:
            current = None
            text = _text(event.content, counts)
            found = calls.get(event.call_id) if event.call_id is not None else None
            if found is not None and found[0] is not None:  # after the assistant turn that issued the call
                t = found[0]
                answered.add((t, found[1]))
                if t != len(turns) - 1:
                    counts["results_moved"] += 1
                turns[t][2].append(({"role": "tool", "tool_call_id": event.call_id, "content": text},
                                    {"events": [here]}))
                continue
            if found is None:
                counts["results_without_call"] += 1
                role, label = side(step.get("source"), event.actor)
            else:
                role, label = "context", found[2]
            name = event.name or (found[3] if found is not None else None)
            if role == "context":
                counts["projected"] += 1
                text = "[%s] tool result%s: %s" % (label, " " + name if name else "", text)
            turns.append([{"role": "user", "content": text}, {"events": [here]}, []])

    messages: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    if not any(e.role == "system" for _s, e in timeline):
        owner_id = perspective if perspective is not None else sole
        owner = next((a for a in roster if a.get("id") == owner_id), None) if owner_id is not None else None
        instructions = (owner or {}).get("instructions")
        if instructions is not None and instructions != "":
            messages.append({"role": "system", "content": _text(instructions, counts)})
            items.append({"instructions": {"trajectory": run.id, "agent": owner.get("id")}})
    for t, (msg, item, results) in enumerate(turns):
        if msg["role"] == "assistant":
            entry["assistant_turns"] += 1
            if not (msg["content"] is None and msg.get("tool_calls")):
                msg["content"] = _text(msg["content"], counts)
            counts["unanswered_calls"] += sum((t, k) not in answered for k in range(len(msg.get("tool_calls", []))))
        messages.append(msg)
        items.append(item)
        for result in results:
            messages.append(result[0])
            items.append(result[1])
    row: Dict[str, Any] = {"messages": messages}
    tools = _tools(run, counts)
    if tools:
        row["tools"] = tools
    entry["messages"] = items
    return row, entry


def _rows(runs: Union[Run, Iterable[Run]], perspective: Optional[str], as_json: bool,
          entries: Optional[List[Dict[str, Any]]]) -> Iterator[Dict[str, Any]]:
    """The rows, one run at a time; each run's report entry is kept only when ``entries`` collects them."""
    single = isinstance(runs, Run)
    count = 0
    for index, run in enumerate(_iter_runs(runs)):
        row, entry = _row(run, perspective, as_json)
        if row is None and single:
            what = "a config record" if entry["skipped"] == "config" else "a trajectory with no events"
            raise ValueError(f"run {run.id!r} is {what}, which gives no chat row")
        entry["input"] = index
        if row is not None:
            entry["row"] = count
            count += 1
        if entries is not None:
            entries.append(entry)
        if row is not None:
            yield row


def to_chat(runs: Union[Run, Iterable[Run]], path: Union[str, Path, None] = None, *, perspective: Optional[str] = None,
            arguments: str = "json", report: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
    """Chat rows ``{"messages": [...], "tools": [...]}``, one per trajectory: a list, or with ``path`` UTF-8 JSON Lines
    written there and ``None`` returned.

    ``runs`` is a ``Run`` or an iterable of them, read once. A config record or a trajectory with no events gives no
    row: alone it raises ``ValueError``, and in an iterable it is skipped and reported. A subagent's trajectory is a
    run of its own: pass ``run.subagents``. For a chat template that takes argument objects, use
    ``row = ap.to_chat(run, arguments="dict")[0]`` and ``tokenizer.apply_chat_template(row["messages"],
    tools=row.get("tools"))``.

    - Messages follow ``run.events()``. System and user steps keep their roles and places. Without a system step, the
      instructions of the perspective agent, or of the run's only agent, open the row as a system message.
    - Agents: with ``perspective=None`` every agent speaks as ``assistant``, with its id as ``name`` when the run has
      more than one. ``perspective="X"`` keeps agent X as the assistant and turns every other agent's messages, calls
      and results into user text that starts with ``[id]``; the rows cannot show which of them X actually saw. A
      simulated user (the roster agent named ``user_simulator`` in tau2-bench) is user text in both modes, and an
      agent step with no actor in a run of several agents is ``[agent]`` text under a perspective.
    - A tool result follows the assistant turn that issued its call, matched to the latest earlier call with its id.
      A result with no such call becomes a user message; a call with no result stays on its turn.
    - Arguments: the parsed object, ``{"_raw": <source>}`` where the source text did not parse or was null, ``{}``
      where the source had none; ``arguments="json"`` writes it as JSON text (OpenAI fine-tuning, Hugging Face
      datasets) and ``"dict"`` as the object (chat templates that take a dict, Invariant).
    - Content: a string as it is; content parts as their text, one per line, with any other part as JSON text; other
      values as JSON text. Null content stays only on an assistant turn with calls, and is ``""`` elsewhere, as is a
      result the source lacks.
    - ``tools``: the agent's tool definitions and those of ``run.tools``, as OpenAI function tools, each identical
      definition once; the key is left out when there are none.

    When ``report`` is a dict, it is cleared and ``report["runs"]`` gets one entry per input run: ``input`` and
    ``row`` (None when skipped) indexes, ``id``, ``source``, ``skipped`` (None, "config" or "no events"),
    ``assistant_turns``, ``counts`` of every change above (``results_moved`` counts a result placed before turns the
    source recorded ahead of it, ``unanswered_calls`` the row's assistant calls with no result, and ``null_content``
    the nulls written as ``""``), and ``messages``, parallel to the row's messages. Each holds ``events``: the
    ``position`` in ``run.events()`` and the ``address`` of every event the message came from (an assistant turn's
    message, then its calls). Instructions copied from the roster hold ``instructions``: the trajectory and agent.
    """
    if arguments not in ("json", "dict"):
        raise ValueError(f'arguments must be "json" or "dict", got {arguments!r}')
    if report is not None and not isinstance(report, dict):
        raise TypeError(f"report must be a dict, got {type(report).__name__}")
    entries: Optional[List[Dict[str, Any]]] = None
    if report is not None:
        report.clear()
        report["runs"] = entries = []
    rows = _rows(runs, perspective, arguments == "json", entries)
    if path is None:
        return list(rows)
    dest = Path(path)
    # write beside the destination and move it in at the end, so an error partway through (a run without the
    # perspective agent) leaves an earlier file as it was
    tmp = dest.with_name(".%s.%s.tmp" % (dest.name, uuid.uuid4().hex))
    try:
        with open(tmp, "x", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(tmp, dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return None
