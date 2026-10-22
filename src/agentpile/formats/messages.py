"""OpenAI-style message lists: the shared parser behind most trajectory datasets.

``messages_to_trajectory`` turns ``[{role, content, tool_calls, ...}, ...]``
into an ATIF v1.7 trajectory:

- ``system``/``developer`` messages become system steps, ``user`` messages
  user steps, and ``assistant`` messages agent steps carrying their tool calls.
- A ``tool`` message becomes an observation result on the agent step that
  issued the call with the same ``tool_call_id``. Calls and results are paired
  by id only. A result whose id is absent or matches no earlier call is kept
  unpaired (``source_call_id`` null) on the latest agent step, never matched
  by name or position.
- Where ATIF needs a stand-in, the source value is kept in the extension:
  ``message_state`` for a null or absent content, ``arguments_raw`` for
  arguments that are not a JSON object (an explicit null included; any other
  such value also sets ``arguments_state: "unparsed"``, which says ATIF holds
  ``{}`` for it),
  ``arguments_state: "missing"`` for a call whose source had no ``arguments``
  member at all (an absent member, a null and an empty object stay three
  different records), ``message_raw`` / ``content_raw`` for content parts ATIF
  cannot hold, ``id_generated`` for a call the source gave no id.
  ``source_ref`` and ``result_refs`` are message indices, so the source order
  is recoverable even when a result arrives after a later turn;
  ``Run.events()`` follows it.
- Content given as a list of parts maps to ATIF content parts: a text part
  (``{"type": "text", "text": ...}``) is kept, and a part ATIF v1.7 cannot hold
  (an OpenAI ``image_url``, an audio or file part) becomes a text stand-in that
  names the part type, while the whole original list stays in the extension.
- Tool definitions (``tool_definitions``) go to ``agent.tool_definitions`` and
  to the public tool view, one entry per definition with the same definition
  and its ``atif_index``. A call carries the entry's ``tool_id`` only when
  exactly one definition has the call's function name.
- Message keys the parser does not map are kept in ``source_fields``.

The file reader accepts ``{"messages": [...]}`` documents (JSON or JSONL) and
a JSON file whose top level is the message list itself.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from .._io import _is_jsonl, _open_text, iter_documents
from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run

name = "openai-messages"

ROLE_TO_SOURCE = {"system": "system", "developer": "system", "user": "user", "assistant": "agent"}
TOOL_ROLES = ("tool", "function")
_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "function_call", "name", "reasoning_content"}
_RESULT_KEYS = {"role", "content", "tool_call_id", "name"}


def _parse_arguments(raw: Any):
    """ATIF arguments must be an object: return (arguments, raw value or MISSING_RAW)."""
    if isinstance(raw, dict):
        return raw, _NO_RAW
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}, raw
        if isinstance(parsed, dict):
            return parsed, _NO_RAW
    return {}, raw


_NO_RAW = object()
_ABSENT = object()  # the call had no arguments member at all, which is not the same as null


def _first_present(tc: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in tc:
            return tc[key]
    return _ABSENT


def _split_call(tc: Dict[str, Any]):
    """(id, function name, raw arguments or ``_ABSENT``) from the call shapes datasets use."""
    fn = tc.get("function")
    if isinstance(fn, dict):  # OpenAI: {id, type, function: {name, arguments}}
        return tc.get("id"), fn.get("name"), fn["arguments"] if "arguments" in fn else _ABSENT
    if isinstance(fn, str):  # AgentDojo: {id, function, args}
        return tc.get("id"), fn, _first_present(tc, "args", "arguments")
    return tc.get("id"), tc.get("name"), _first_present(tc, "arguments", "args")


# -- content parts -----------------------------------------------------------------

IMAGE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")


def _is_atif_image_source(source: Any) -> bool:
    return (isinstance(source, dict) and set(source) == {"media_type", "path"}
            and source["media_type"] in IMAGE_MEDIA_TYPES and isinstance(source["path"], str))


def _stand_in(part: Any) -> Dict[str, Any]:
    kind = part.get("type") if isinstance(part, dict) else None
    label = kind if isinstance(kind, str) and kind else "unknown"
    return {"type": "text", "text": f"[{label} part: original in agentpile extension]"}


def _atif_part(part: Any):
    """(ATIF content part, whether it equals the source part) for one source content part."""
    if isinstance(part, dict):
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            return {"type": "text", "text": part["text"]}, set(part) == {"type", "text"}
        if part.get("type") == "image" and set(part) == {"type", "source"} and _is_atif_image_source(part["source"]):
            return part, True
    elif isinstance(part, str):  # a bare string in a content list is text
        return {"type": "text", "text": part}, False
    return _stand_in(part), False


def _parts(content: List[Any]):
    """(ATIF content, original list or ``_NO_RAW``) for a content list.

    Text parts map to ATIF text parts; any other part becomes a text stand-in
    naming its type. When any part changed, the whole source list is returned
    for the extension, so the record keeps exactly what the source gave.
    """
    out, exact = [], True
    for part in content:
        converted, same = _atif_part(part)
        out.append(converted)
        exact = exact and same
    if exact:
        return content, _NO_RAW
    return out, content


# -- tool definitions -----------------------------------------------------------------

def _function_name(definition: Any) -> Optional[str]:
    if not isinstance(definition, dict):
        return None
    fn = definition.get("function") if isinstance(definition.get("function"), dict) else definition
    name = fn.get("name")
    return name if isinstance(name, str) and name else None


def _tool_entries(defs: Any, tools: Dict[str, Dict[str, Any]]):
    """Give every ATIF tool definition a public tool entry; return ``name -> tool id`` for names defined once.

    Entries the caller already supplied (and their ``atif_index``) are kept as
    they are, and then no name is linked to a call here: the caller owns the
    tool identity. A definition with no entry gets one holding a copy of the
    same definition and its ``atif_index``; a name used by two definitions links
    no call, because the source does not say which one a call used.
    """
    by_name: Dict[str, str] = {}
    if not isinstance(defs, list):
        return by_name
    covered = {t.get("atif_index") for t in tools.values() if isinstance(t, dict)}
    own = not tools
    names = [_function_name(d) for d in defs]
    counts = Counter(n for n in names if n)
    for index, definition in enumerate(defs):
        if index in covered or not isinstance(definition, dict):
            continue
        name = names[index]
        tid = name if name else f"tool#{index}"
        if tid in tools:
            tid = f"{name or 'tool'}#{index}"
        while tid in tools:
            tid += "'"
        tools[tid] = {"name": name or "", "agent_or_server": None,
                      "type": definition.get("type") if isinstance(definition.get("type"), str) else None,
                      "definition": copy.deepcopy(definition), "definition_format": "openai", "atif_index": index,
                      "permission": "unknown", "permission_from": "unknown"}
        if own and name and counts[name] == 1:
            by_name[name] = tid
    return by_name


def _content(msg: Dict[str, Any], sx: Dict[str, Any]) -> Any:
    if "content" not in msg:
        sx["message_state"] = "missing"
        return ""
    content = msg["content"]
    if content is None:
        sx["message_state"] = "null"
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted, raw = _parts(content)
        if raw is not _NO_RAW:
            sx["message_raw"] = raw
        return converted
    sx["source_fields"] = dict(sx.get("source_fields") or {}, content=content)
    return json.dumps(content, ensure_ascii=False)


def messages_to_trajectory(
    messages: List[Dict[str, Any]],
    trajectory_id: Optional[str] = None,
    agent_name: str = "unknown",
    agent_version: str = "unknown",
    model_name: Optional[str] = None,
    tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ext: Optional[Dict[str, Any]] = None,
    actor: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    source_of: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    pair_by_position: bool = False,
) -> Dict[str, Any]:
    """Build an ATIF trajectory dict from a message list.

    ``ext`` is the trajectory-level agentpile extension (source, task,
    outcome, labels, agents, tools, dataset); ``version`` is filled in.
    ``actor(message)`` names the agent behind a message, for datasets that
    record it; by default the message's ``name`` field is used for agent steps.
    ``source_of(message)`` overrides the ATIF step source for a message (for
    example, a simulated user that calls tools must be an agent step, since
    ATIF allows tool calls on agent steps only); the original role is kept.
    ``pair_by_position`` is for sources whose tool results carry no call id:
    the k-th id-less result after a call-bearing message answers its k-th
    call, and each such pairing is marked ``paired_by: "position"``.
    """
    ext = dict(ext or {})
    tools = dict(ext.get("tools") or {})
    by_name = _tool_entries(tool_definitions, tools)
    steps: List[Dict[str, Any]] = []
    call_step: Dict[str, int] = {}  # tool call id -> index in steps
    last_agent: Optional[int] = None
    open_calls: List[str] = []  # ids of the latest call-bearing step not yet answered by position

    def new_step(source: str, message: Any, sx: Dict[str, Any]) -> Dict[str, Any]:
        step = {"step_id": len(steps) + 1, "source": source, "message": message}
        steps.append(step)
        if sx:
            step["extra"] = {EXT: sx}
        return step

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role in TOOL_ROLES:
            cid = msg.get("tool_call_id")
            by_position = False
            if cid is None and pair_by_position and open_calls:
                cid = open_calls.pop(0)
                by_position = True
            target = call_step.get(cid) if cid is not None else None
            paired = target is not None
            if not paired:
                target = last_agent
            if target is None:  # a tool output before any agent turn: keep it as its own step
                sx = {"source_ref": i, "role": role}
                new_step("user", _content(msg, sx), sx)
                continue
            step = steps[target]
            fields = {k: v for k, v in msg.items() if k not in _RESULT_KEYS or k == "name"}
            content = msg.get("content")
            if "content" not in msg or content is None:  # ATIF content cannot be null
                fields["content_state"] = "missing" if "content" not in msg else "null"
                content = ""
            elif isinstance(content, list):
                content, raw_parts = _parts(content)
                if raw_parts is not _NO_RAW:
                    fields["content_raw"] = raw_parts
            elif not isinstance(content, str):
                fields["content_raw"] = content
                content = json.dumps(content, ensure_ascii=False)
            obs = step.setdefault("observation", {"results": []})
            obs["results"].append({"source_call_id": cid if paired else None, "content": content})
            sx = step.setdefault("extra", {}).setdefault(EXT, {})
            sx.setdefault("result_refs", []).append(i)
            if role != "tool":
                fields["role"] = role
            if by_position:
                fields["paired_by"] = "position"
            elif not paired and cid is not None:
                fields["tool_call_id"] = cid  # kept, though it matches no earlier call
            sx.setdefault("result_fields", []).append(fields or None)
            continue

        default = ROLE_TO_SOURCE.get(role, "user")
        source = (source_of(msg) if source_of else None) or default
        sx: Dict[str, Any] = {"source_ref": i}
        if role not in ROLE_TO_SOURCE or role == "developer" or source != default:
            sx["role"] = role
        message = _content(msg, sx)
        step = new_step(source, message, sx)
        if source == "agent":
            who = actor(msg) if actor else msg.get("name")
            if who is not None:
                sx["actor"] = who
            if isinstance(msg.get("reasoning_content"), str):
                step["reasoning_content"] = msg["reasoning_content"]
            raw_calls = list(msg.get("tool_calls") or [])
            if not raw_calls and isinstance(msg.get("function_call"), dict):  # legacy single call
                raw_calls = [{"function": msg["function_call"]}]
            calls, cx_all = [], {}
            for k, tc in enumerate(raw_calls):
                cid, fname, raw_args = _split_call(tc)
                cx: Dict[str, Any] = {}
                if not isinstance(cid, str) or not cid or cid in cx_all:
                    if cid is not None:
                        cx["source_id"] = cid
                    cid = f"ap-{step['step_id']}-{k}"
                    cx["id_generated"] = True
                if raw_args is _ABSENT:  # no arguments member: ATIF gets {}, the extension says it was absent
                    args = {}
                    cx["arguments_state"] = "missing"
                else:
                    args, raw = _parse_arguments(raw_args)
                    if raw is not _NO_RAW:
                        cx["arguments_raw"] = raw
                        if raw is not None:  # {} stands in for a value that is not an object (an explicit null is no failed parse)
                            cx["arguments_state"] = "unparsed"
                extra_keys = {k2: v for k2, v in tc.items() if k2 not in ("id", "type", "function", "args", "arguments", "name")}
                if extra_keys:
                    cx["source_fields"] = extra_keys
                calls.append({"tool_call_id": cid, "function_name": fname if isinstance(fname, str) else "",
                              "arguments": args})
                if not isinstance(fname, str):
                    cx["function_name_raw"] = fname
                elif fname in by_name:
                    cx["tool_id"] = by_name[fname]
                cx_all[cid] = cx
                if not cx.get("id_generated") or pair_by_position:
                    call_step[cid] = len(steps) - 1
            if pair_by_position:
                open_calls = [c["tool_call_id"] for c in calls]
            if calls:
                step["tool_calls"] = calls
                if any(cx_all.values()):
                    sx["calls"] = {c: v for c, v in cx_all.items() if v}
            last_agent = len(steps) - 1
        else:
            if msg.get("name") is not None:
                sx["name"] = msg["name"]
            if pair_by_position:
                open_calls = []  # a new non-agent turn closes the positional window
        unknown = {k: v for k, v in msg.items() if k not in _MESSAGE_KEYS}
        if unknown:
            sx["source_fields"] = dict(sx.get("source_fields") or {}, **unknown)

    if not steps:  # ATIF requires one step; an empty conversation is a missing message, not a fabricated one
        new_step("system", "", {"message_state": "missing", "empty_source": True})

    agent: Dict[str, Any] = {"name": agent_name, "version": agent_version}
    if model_name is not None:
        agent["model_name"] = model_name
    if tool_definitions is not None:
        agent["tool_definitions"] = tool_definitions
    traj: Dict[str, Any] = {"schema_version": ATIF_VERSION, "session_id": trajectory_id,
                            "trajectory_id": trajectory_id, "agent": agent, "steps": steps}
    if tools:
        ext["tools"] = tools
    ext["version"] = EXT_VERSION
    ext.setdefault("source", {"format": name})
    traj["extra"] = {EXT: ext}
    return traj


# -- file reader ---------------------------------------------------------------

def _is_message(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("role"), str)


def _looks_like_conversation(doc: Any) -> bool:
    return (isinstance(doc, dict) and isinstance(doc.get("messages"), list)
            and bool(doc["messages"]) and all(_is_message(m) for m in doc["messages"][:5]))


def _whole_file(path: Path) -> Any:
    if _is_jsonl(path):
        return None
    try:
        with _open_text(path) as fh:  # a .json.gz file reads like its .json
            return json.load(fh)
    except (ValueError, OSError, UnicodeDecodeError):
        return None


def sniff(source: Any) -> bool:
    if isinstance(source, Path):
        if not source.is_file():
            return False
        whole = _whole_file(source)
        if isinstance(whole, list):
            return bool(whole) and (_is_message(whole[0]) or _looks_like_conversation(whole[0]))
        if whole is not None:
            return _looks_like_conversation(whole)
        first = next(iter_documents(source), None) if source.is_file() else None
        return _looks_like_conversation(first)
    if isinstance(source, list):
        return bool(source) and (_is_message(source[0]) or _looks_like_conversation(source[0]))
    return _looks_like_conversation(source)


def _documents(source: Any) -> Iterator[Any]:
    if isinstance(source, Path):
        whole = _whole_file(source)
        if isinstance(whole, list) and whole and _is_message(whole[0]):
            yield {"messages": whole}
            return
        if whole is not None:
            yield from (whole if isinstance(whole, list) else [whole])
            return
        yield from iter_documents(source)
        return
    if isinstance(source, list) and source and _is_message(source[0]):
        yield {"messages": source}
        return
    yield from (source if isinstance(source, list) else [source])


def read(source: Any, **options: Any) -> Iterator[Run]:
    if options:
        raise TypeError(f"openai-messages reader takes no options, got {sorted(options)}")
    for k, doc in enumerate(_documents(source)):
        tid = doc.get("id")
        tid = str(tid) if tid is not None else f"conversation-{k}"
        defs = doc.get("tools") if isinstance(doc.get("tools"), list) else None
        # a "tools" member that is not a list stays in dataset as given (null included)
        rest = {key: v for key, v in doc.items()
                if key not in ("messages", "id") and not (key == "tools" and defs is not None)}
        ext = {"source": {"format": name, "record_key": tid, "reader": name, "lost_fields": []}}
        if rest:
            ext["dataset"] = rest
        yield Run(messages_to_trajectory(doc["messages"], trajectory_id=tid,
                                         tool_definitions=defs, ext=ext), TRAJECTORY)
