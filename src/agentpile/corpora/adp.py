"""ADP (neulab/agent-data-collection): the legacy ``std`` rows of the Agent Data Protocol collection.

One run per row of ``<config>/full_std.jsonl``. A row is ``{id, content[], details{}}``: a flat list of typed
items (``text_observation``, ``message_action``, ``api_action``, ``code_action``, ``web_observation``) with no
roles, ids, timestamps or agent names. The raw and ``sft_*`` splits are not read. The default load reads the
std split of every supported config (``config=`` selects one). A config is supported when its std rows all become
valid ATIF (agentpile and Harbor), no value is read as a failure label, and the rows read equal the manifest
rows; the manifest lists the 14 supported configs and, for every other config, why it is unsupported (four
declared configs and five undeclared directories whose std rows could not be counted). Row ids repeat across
configs, so a run id is ``<config>/<id>``.

Items become ATIF steps in list order; step ``source_ref`` is the item index and ``source_fields.class_`` the
item's class. Where ADP converts its own rows to ATIF, this loader follows it: ``adp_trajectory_to_atif``,
``code_action_to_tool_call`` (schema/atif.py) and ``backfill_adjacent_tool_call_links`` (schema/tool_call_links.py)
in github.com/neulab/agent-data-protocol at commit 396133a9357a5a0041c39eb26e3642e93f23473f, the last commit of
PR 256 that holds them (main dropped the legacy schema in its squash merge), and ``web_observation_step``
(scripts/legacy_atif.py) at main 040a279b46b2388ae42b43449f8645b9781c7bf7. The manifest records the same
references.

- ``message_action``: an agent step with ``message`` = content. Its ``description`` stays in ``source_fields``
  (it is the source's optional note, not a model reasoning field the source names as such).
- ``api_action`` / ``code_action``: an agent step with ``message`` = description (null or missing is kept as
  ``message_state``) and one tool call. An ``api_action`` calls ``function`` with ``kwargs`` verbatim (many
  values are JSON-encoded strings and are not decoded). A ``code_action`` follows ADP's own ADP-to-ATIF code
  (``code_action_to_tool_call``): language ``bash``/``sh``/``shell`` becomes ``execute_bash`` with
  ``{"command": content}``, ``python``/``py``/``python3`` becomes ``execute_ipython_cell`` with
  ``{"code": content}``, every other language becomes ``execute_code`` with ``{"language", "content"}``; the
  source language stays in the call's ``source_fields``.
- An observation that directly follows an ``api_action`` or ``code_action`` is that call's result, attached to
  the same step. The source has no call ids, so this pairing is ADP's own adjacency rule
  (``backfill_adjacent_tool_call_links``): every such result is flagged ``paired_by: "adp_sequence"`` in
  ``result_fields`` and every call id is generated (``id_generated``), in ADP's format ``call_000001``, ... per
  linked pair in row order, and ``call_<step id>`` for an action with no observation after it (the last item of
  a synatra row). The result keeps the observation's class and its raw ``source``; ADP's own output relabels a
  linked ``user`` source as ``environment``, which is not done here. An ``api_action`` whose item has no
  ``kwargs`` member keeps ``arguments_state: "missing"`` in the call extension, a null ``kwargs`` keeps
  ``arguments_raw: null``, and ``{}`` is the empty object, so the three stay different. Any other ``kwargs``
  that is not an object (a list, a string) is kept in ``arguments_raw`` and marks the call ``arguments_state: "unparsed"``.
- Any other observation has no call to attach to and becomes its own step: ATIF source ``agent`` when the item
  says ``source: "agent"`` (390 items), else ``user`` (``source`` is ``user`` or ``environment``; ATIF has no
  environment source, and ADP's own output makes an environment-sourced one an empty agent step instead). The
  item's ``source`` is kept as the step ``role``.
- ``web_observation``: the message or result content is ``html``, else ``axtree``, else ``url`` (ADP's
  ``web_observation_step`` order; ``content_from`` names the field used, null when all are empty). The other web
  fields stay under ``web`` (``html``, ``axtree``, ``url``, ``image_observation``, ``viewport_size``) without a
  copy of the field used as content. Screenshot paths in ``image_observation`` point at files the dataset does
  not ship.

``details`` stays whole in ``dataset`` and is not interpreted, except: ``task_description`` (synatra) is the
task text; ``model_name`` (nebius) is the ATIF agent model; ``tools`` (swe-gym, a JSON string of OpenAI-style
function definitions) becomes ``agent.tool_definitions`` plus extension tools that hold the same definition and
its ``atif_index`` (permission unknown), and a call whose function name equals the name of exactly one declared
tool carries its ``tool_id`` (including the generated ``execute_bash``, which is the declared tool's name; a name
two declared tools share links no call, because the row does not say which one the call used); ``resolved`` (swe-gym, the string
"True") sets ``outcome.success``; ``resolved``, ``feedback``, ``polarity`` and ``exit_status`` are echoed in
``outcome.raw``. ADP has no failure labels, so ``labels`` is always empty: every ``resolved`` and ``feedback``
value in the supported std files is the success value, and ``exit_status`` (``submitted``, ``submitted
(exit_context)``, ``submitted (exit_format)``) records how an episode ended, not whether it failed. The agent
name and version are "unknown" because no row names an agent.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..record import ATIF_VERSION, EXT, EXT_VERSION, TRAJECTORY, Run
from ._base import Dataset, manifest as load_manifest

ACTIONS = ("api_action", "code_action")
OBSERVATIONS = ("text_observation", "web_observation")
BASH = ("bash", "sh", "shell")
PYTHON = ("python", "py", "python3")
OUTCOME_FIELDS = ("resolved", "feedback", "polarity", "exit_status")
_WEB_FIELDS = ("html", "axtree", "url")


@lru_cache(maxsize=1)
def _manifest() -> Dict[str, Any]:
    return load_manifest("adp")  # the base class re-reads the file on every access, too slow for 200k rows


def _cls(item: Any) -> Optional[str]:
    return item.get("class_") if isinstance(item, dict) else None


def _text(item: Dict[str, Any], key: str, fields: Dict[str, Any], raw_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """ATIF text for a source string field: (text, "null"/"missing" when the source had no string)."""
    if key not in item:
        return "", "missing"
    value = item[key]
    if value is None:
        return "", "null"
    if isinstance(value, str):
        return value, None
    fields[raw_key or key] = value  # a non-string value stays as the source gave it
    return json.dumps(value, ensure_ascii=False), None


def _call(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """(function name, arguments, call extension) of an api_action or code_action."""
    cx: Dict[str, Any] = {}
    if item["class_"] == "api_action":
        name = item.get("function")
        if "kwargs" not in item:  # no member at all: ATIF gets {}, the extension says it was absent
            args = {}
            cx["arguments_state"] = "missing"
        else:
            kwargs = item["kwargs"]
            args = kwargs if isinstance(kwargs, dict) else {}
            if not isinstance(kwargs, dict):
                cx["arguments_raw"] = kwargs
                if kwargs is not None:  # {} stands in for a value that is not an object (an explicit null is no failed parse)
                    cx["arguments_state"] = "unparsed"
    else:  # ADP's code_action_to_tool_call
        language, content = item.get("language"), item.get("content")
        if language in BASH:
            name, args = "execute_bash", {"command": content}
        elif language in PYTHON:
            name, args = "execute_ipython_cell", {"code": content}
        else:
            name, args = "execute_code", {"language": language, "content": content}
        cx["source_fields"] = {"class_": "code_action", "language": language}
    if not isinstance(name, str):
        cx["function_name_raw"] = name
        name = ""
    return name, args, cx


def _web(item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """(content, fields) of a web_observation: the first non-empty of html, axtree, url, and the rest verbatim."""
    used = next((k for k in _WEB_FIELDS if isinstance(item.get(k), str) and item[k]), None)
    content = item[used] if used else ""
    web = {k: item.get(k) for k in _WEB_FIELDS + ("image_observation", "viewport_size") if k != used}
    return content, {"class_": "web_observation", "content_from": used, "web": web}


def _tools(details: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """ATIF tool definitions and extension tools from ``details.tools`` (swe-gym: a JSON string); none otherwise."""
    raw = details.get("tools")
    if not isinstance(raw, str):
        return None, {}
    try:
        defs = json.loads(raw)
    except ValueError:
        return None, {}
    if not isinstance(defs, list) or not defs or not all(isinstance(d, dict) for d in defs):
        return None, {}
    tools: Dict[str, Dict[str, Any]] = {}
    for index, d in enumerate(defs):
        fn = d.get("function") if isinstance(d.get("function"), dict) else d
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            return None, {}  # all or nothing; the raw string stays in dataset.details
        tid = name if name not in tools else f"{name}#{index}"
        tools[tid] = {"name": name, "agent_or_server": None,
                      "type": d.get("type") if isinstance(d.get("type"), str) else None,
                      "definition": copy.deepcopy(d), "definition_format": "openai-function", "atif_index": index,
                      "permission": "unknown", "permission_from": "unknown"}
    return defs, tools


class Adp(Dataset):
    name = "adp"
    format = "adp-std-jsonl"
    incremental = True

    @property
    def manifest(self) -> Dict[str, Any]:
        return _manifest()

    def records(self, root: Path, config: Optional[str] = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"adp takes config= only, got {sorted(options)}")
        supported = self.manifest["supported_configs"]
        names = [c["config"] for c in supported]
        if config is not None and config not in names:
            why = next((u.get("reason") for u in self.manifest["unsupported"] if u.get("config") == config), None)
            raise ValueError(f"config must be one of {names}, got {config!r}" + (f" (unsupported: {why})" if why else ""))
        for entry in supported:
            if config is not None and entry["config"] != config:
                continue
            rel, row = entry["file"], 0
            with self.ready(root / rel).open("r", encoding="utf-8") as fh:
                for number, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"{rel} line {number} is not JSON") from exc
                    if not isinstance(raw, dict):
                        raise ValueError(f"{rel} line {number} is not a JSON object")
                    rid = raw.get("id")
                    yield (f"{entry['config']}/{rid if rid is not None else 'row-' + str(row)}", raw,
                           {"file": rel, "config": entry["config"], "split": "std", "row": row})
                    row += 1

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        items = raw.get("content") if isinstance(raw.get("content"), list) else []
        details_raw = raw.get("details")
        details = details_raw if isinstance(details_raw, dict) else {}
        tool_defs, tools = _tools(details)
        name_count = Counter(t["name"] for t in tools.values())
        tool_ids = {t["name"]: tid for tid, t in tools.items() if name_count[t["name"]] == 1}  # a shared name is ambiguous

        # ADP's adjacency rule: an action followed directly by an observation is that call's result.
        linked: Dict[int, str] = {}
        for i, item in enumerate(items):
            if _cls(item) in ACTIONS and i + 1 < len(items) and _cls(items[i + 1]) in OBSERVATIONS:
                linked[i] = f"call_{len(linked) + 1:06d}"

        steps: List[Dict[str, Any]] = []
        used = set(linked.values())
        consumed = set()
        for i, item in enumerate(items):
            if i in consumed:
                continue
            cls = _cls(item)
            step_id = len(steps) + 1
            fields: Dict[str, Any] = {"class_": cls}
            sx: Dict[str, Any] = {"source_ref": i, "source_fields": fields}
            if cls in ACTIONS:
                cid = linked.get(i)
                if cid is None:
                    cid = f"call_{step_id:06d}"
                    while cid in used:
                        cid += "u"
                used.add(cid)
                fname, args, cx = _call(item)
                cx.update({"id_generated": True, "source_ref": i})
                if fname in tool_ids:
                    cx["tool_id"] = tool_ids[fname]
                message, state = _text(item, "description", fields)
                step = {"step_id": step_id, "source": "agent", "message": message,
                        "tool_calls": [{"tool_call_id": cid, "function_name": fname, "arguments": args}]}
                sx["calls"] = {cid: cx}
                if i in linked:
                    consumed.add(i + 1)
                    obs = items[i + 1]
                    if obs["class_"] == "web_observation":
                        content, rfields = _web(obs)
                    else:
                        rfields = {"class_": "text_observation", "source": obs.get("source")}
                        content, content_state = _text(obs, "content", rfields, raw_key="content_raw")
                        if obs.get("name") is not None:
                            rfields["name"] = obs["name"]
                        if content_state:
                            rfields["content_state"] = content_state
                    rfields["paired_by"] = "adp_sequence"
                    step["observation"] = {"results": [{"source_call_id": cid, "content": content}]}
                    sx["result_refs"] = [i + 1]
                    sx["result_fields"] = [rfields]
            elif cls == "message_action":
                message, state = _text(item, "content", fields)
                if "description" in item:
                    fields["description"] = item["description"]
                step = {"step_id": step_id, "source": "agent", "message": message}
            elif cls == "text_observation":
                source = item.get("source")
                message, state = _text(item, "content", fields)
                step = {"step_id": step_id, "source": "agent" if source == "agent" else "user", "message": message}
                if isinstance(source, str):
                    sx["role"] = source
                if item.get("name") is not None:
                    sx["name"] = item["name"]
            elif cls == "web_observation":
                message, wfields = _web(item)
                fields.update(wfields)
                state = None
                step = {"step_id": step_id, "source": "user", "message": message}
            else:
                raise ValueError(f"{key}: content[{i}] has unsupported class_ {cls!r}")
            if state:
                sx["message_state"] = state
            step["extra"] = {EXT: sx}
            steps.append(step)
        if not steps:  # ATIF requires one step; an empty row is a missing message, not a fabricated one
            steps.append({"step_id": 1, "source": "system", "message": "",
                          "extra": {EXT: {"message_state": "missing", "empty_source": True}}})

        task_text = details.get("task_description")
        resolved = details.get("resolved")
        success = None  # swe-gym stores the flag as the string "True"; anything else is not read as an outcome
        if isinstance(resolved, bool):
            success = resolved
        elif resolved in ("True", "False"):
            success = resolved == "True"
        source = self.source(key, where)
        source["row"] = where.get("row")
        ext = {
            "version": EXT_VERSION,
            "source": source,
            "task": {"id": None, "text": task_text if isinstance(task_text, str) else None},
            "outcome": {"success": success, "score": None,
                        "raw": {k: details[k] for k in OUTCOME_FIELDS if k in details} or None},
            "labels": [],
            "agents": [],
            "tools": tools,
            "dataset": {k: v for k, v in raw.items() if k != "content"},
        }
        agent: Dict[str, Any] = {"name": "unknown", "version": "unknown"}
        if isinstance(details.get("model_name"), str):
            agent["model_name"] = details["model_name"]
        if tool_defs is not None:
            agent["tool_definitions"] = tool_defs
        traj = {"schema_version": ATIF_VERSION, "session_id": key, "trajectory_id": key, "agent": agent,
                "steps": steps, "extra": {EXT: ext}}
        return Run(traj, TRAJECTORY)


dataset = Adp()
