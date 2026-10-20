"""Structural checks for run records, with the standard library only.

``validate(run)`` returns a list of problems, empty when the record is sound.
It checks the ATIF v1.7 constraints a reader can break (sequential step ids,
step sources, non-null messages, the shape of content parts, results that point
at a call on the same step), the bundled extension schema, and that every ATIF
``agent.tool_definitions`` entry belongs to exactly one extension tool whose
public definition equals it. The development extra additionally validates the
ATIF part against the pinned Harbor models.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List

from .record import ATIF_VERSION, CONFIG, EXT, Run

STEP_SOURCES = ("system", "user", "agent")
# Field sets of the pinned Harbor models (harbor 0.22.0, ATIF v1.7), which reject unknown keys.
TRAJECTORY_KEYS = {"schema_version", "session_id", "trajectory_id", "agent", "steps", "notes", "final_metrics",
                   "continued_trajectory_ref", "extra", "subagent_trajectories"}
STEP_KEYS = {"step_id", "timestamp", "source", "message", "model_name", "reasoning_effort", "reasoning_content",
             "tool_calls", "observation", "metrics", "llm_call_count", "is_copied_context", "extra"}
AGENT_ONLY = {"tool_calls", "metrics", "model_name", "reasoning_effort", "reasoning_content"}
IMAGE_MEDIA_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")
PART_KEYS = {"type", "text", "source"}
SOURCE_KEYS = {"media_type", "path"}


@lru_cache(maxsize=1)
def schema() -> Dict[str, Any]:
    """The bundled agentpile v0 JSON Schema."""
    text = resources.files("agentpile").joinpath("schemas/agentpile-v0.schema.json").read_text(encoding="utf-8")
    return json.loads(text)


# -- a small JSON Schema subset: type, enum, required, properties,
#    additionalProperties, items and local $ref. It covers the bundled schema;
#    it is not a general validator.

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _check(value: Any, sch: Dict[str, Any], path: str, errors: List[str]) -> None:
    if "$ref" in sch:
        name = sch["$ref"].rsplit("/", 1)[-1]
        sch = schema()["$defs"][name]
    if "type" in sch:
        types = sch["type"] if isinstance(sch["type"], list) else [sch["type"]]
        if not any(_TYPES[t](value) for t in types):
            errors.append(f"{path}: expected {' or '.join(types)}, got {type(value).__name__}")
            return
    if "enum" in sch and value not in sch["enum"]:
        errors.append(f"{path}: {value!r} is not one of {sch['enum']}")
        return
    if isinstance(value, dict):
        for key in sch.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required field {key!r}")
        props = sch.get("properties", {})
        extra = sch.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                _check(item, props[key], f"{path}.{key}", errors)
            elif isinstance(extra, dict):
                _check(item, extra, f"{path}.{key}", errors)
            elif extra is False:
                errors.append(f"{path}: unexpected field {key!r}")
    if isinstance(value, list) and "items" in sch:
        for i, item in enumerate(value):
            _check(item, sch["items"], f"{path}[{i}]", errors)


def check_schema(value: Any, definition: str, path: str = "$") -> List[str]:
    """Check ``value`` against one definition of the bundled schema."""
    errors: List[str] = []
    _check(value, schema()["$defs"][definition], path, errors)
    return errors


def _check_part(part: Any, path: str, errors: List[str]) -> None:
    """One ATIF v1.7 content part: a text part or an image part, with no other members."""
    if not isinstance(part, dict):
        errors.append(f"{path}: ATIF content part must be an object, got {type(part).__name__}")
        return
    kind = part.get("type")
    if kind not in ("text", "image"):
        errors.append(f"{path}.type: {kind!r} is not an ATIF content part type (text or image)")
        return
    for key in part:
        if key not in PART_KEYS:
            errors.append(f"{path}: unknown ATIF content part field {key!r}")
    if kind == "text":
        if not isinstance(part.get("text"), str):
            errors.append(f"{path}.text: a text part needs a string")
        if part.get("source") is not None:
            errors.append(f"{path}.source: not allowed on a text part")
        return
    if part.get("text") is not None:
        errors.append(f"{path}.text: not allowed on an image part")
    source = part.get("source")
    if not isinstance(source, dict):
        errors.append(f"{path}.source: an image part needs a source object")
        return
    for key in source:
        if key not in SOURCE_KEYS:
            errors.append(f"{path}.source: unknown field {key!r}")
    if source.get("media_type") not in IMAGE_MEDIA_TYPES:
        errors.append(f"{path}.source.media_type: {source.get('media_type')!r} is not one of {list(IMAGE_MEDIA_TYPES)}")
    if not isinstance(source.get("path"), str):
        errors.append(f"{path}.source.path: required string is missing")


def _check_content(content: Any, path: str, errors: List[str]) -> None:
    """A string, or a list of ATIF content parts."""
    if isinstance(content, list):
        for k, part in enumerate(content):
            _check_part(part, f"{path}[{k}]", errors)


def _check_trajectory(traj: Dict[str, Any], path: str, errors: List[str]) -> None:
    if traj.get("schema_version") != ATIF_VERSION:
        errors.append(f"{path}.schema_version: expected {ATIF_VERSION!r}, got {traj.get('schema_version')!r}")
    agent = traj.get("agent")
    if not isinstance(agent, dict):
        errors.append(f"{path}.agent: required object is missing")
    else:
        for key in ("name", "version"):
            if not isinstance(agent.get(key), str):
                errors.append(f"{path}.agent.{key}: required string is missing")
    for key in traj:
        if key not in TRAJECTORY_KEYS:
            errors.append(f"{path}: unknown ATIF field {key!r}")
    steps = traj.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append(f"{path}.steps: ATIF requires at least one step")
        steps = []
    for i, step in enumerate(steps):
        sp = f"{path}.steps[{i}]"
        if step.get("step_id") != i + 1:
            errors.append(f"{sp}.step_id: expected {i + 1}, got {step.get('step_id')!r}")
        if step.get("source") not in STEP_SOURCES:
            errors.append(f"{sp}.source: {step.get('source')!r} is not one of {list(STEP_SOURCES)}")
        if not isinstance(step.get("message"), (str, list)):
            errors.append(f"{sp}.message: ATIF requires a string or content parts, got {type(step.get('message')).__name__}")
        else:
            _check_content(step["message"], f"{sp}.message", errors)
        for key in step:
            if key not in STEP_KEYS:
                errors.append(f"{sp}: unknown ATIF field {key!r}")
            elif key in AGENT_ONLY and step.get("source") != "agent":
                errors.append(f"{sp}.{key}: ATIF allows it on agent steps only")
        call_ids = []
        for j, call in enumerate(step.get("tool_calls") or []):
            cp = f"{sp}.tool_calls[{j}]"
            if not isinstance(call.get("tool_call_id"), str):
                errors.append(f"{cp}.tool_call_id: required string is missing")
            if not isinstance(call.get("function_name"), str):
                errors.append(f"{cp}.function_name: required string is missing")
            if not isinstance(call.get("arguments"), dict):
                errors.append(f"{cp}.arguments: ATIF requires an object")
            call_ids.append(call.get("tool_call_id"))
        if len(set(call_ids)) != len(call_ids):
            errors.append(f"{sp}.tool_calls: duplicate tool_call_id on one step")
        for j, res in enumerate((step.get("observation") or {}).get("results") or []):
            ref = res.get("source_call_id")
            if ref is not None and ref not in call_ids:
                errors.append(f"{sp}.observation.results[{j}].source_call_id: {ref!r} matches no call on this step")
            content = res.get("content")
            if content is not None and not isinstance(content, (str, list)):
                errors.append(f"{sp}.observation.results[{j}].content: ATIF requires a string, content parts or null, "
                              f"got {type(content).__name__}")
            else:
                _check_content(content, f"{sp}.observation.results[{j}].content", errors)
        sx = (step.get("extra") or {}).get(EXT)
        if sx is not None:
            _check(sx, schema()["$defs"]["step_ext"], f"{sp}.extra.{EXT}", errors)
            for cid, cx in (sx.get("calls") or {}).items():
                if cid not in call_ids:
                    errors.append(f"{sp}.extra.{EXT}.calls: {cid!r} matches no call on this step")
                if isinstance(cx, dict) and "arguments_state" in cx:
                    if cx["arguments_state"] != "missing":
                        errors.append(f"{sp}.extra.{EXT}.calls[{cid!r}].arguments_state: {cx['arguments_state']!r} is not 'missing'")
                    elif "arguments_raw" in cx:
                        errors.append(f"{sp}.extra.{EXT}.calls[{cid!r}]: arguments_state 'missing' and arguments_raw "
                                      "cannot both be set (absent is not null)")
    ext = (traj.get("extra") or {}).get(EXT)
    if ext is not None:
        _check(ext, schema()["$defs"]["trajectory_ext"], f"{path}.extra.{EXT}", errors)
        _check_tool_ids(traj, ext, path, errors)
    subs = traj.get("subagent_trajectories") or []
    sub_ids = [s.get("trajectory_id") for s in subs]
    if any(not isinstance(i, str) for i in sub_ids) or len(set(sub_ids)) != len(sub_ids):
        errors.append(f"{path}.subagent_trajectories: each needs a unique trajectory_id")
    for k, sub in enumerate(subs):
        _check_trajectory(sub, f"{path}.subagent_trajectories[{k}]", errors)


def _check_tool_ids(traj: Dict[str, Any], ext: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Each ATIF tool definition belongs to exactly one extension tool, through ``atif_index``.

    The public definition of that tool must equal the ATIF definition it
    indexes: a null, a missing definition or an edited one is an error, so the
    two views of one tool cannot drift apart.
    """
    tools = ext.get("tools") or {}
    defs = (traj.get("agent") or {}).get("tool_definitions") or []
    indexes = [t.get("atif_index") for t in tools.values() if t.get("atif_index") is not None]
    if sorted(indexes) != list(range(len(defs))):
        errors.append(f"{path}.extra.{EXT}.tools: atif_index values {sorted(indexes)} do not cover "
                      f"agent.tool_definitions[0..{len(defs) - 1}] exactly once")
    for tid, tool in tools.items():
        index = tool.get("atif_index")
        if index is None or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(defs):
            continue
        if tool.get("definition") != defs[index]:
            errors.append(f"{path}.extra.{EXT}.tools[{tid!r}].definition: differs from agent.tool_definitions[{index}]")


def validate(run: Any) -> List[str]:
    """Problems with a ``Run`` or a raw record; an empty list means none were found."""
    if not isinstance(run, Run):
        run = Run(run)
    errors: List[str] = []
    if run.kind == CONFIG:
        _check(run.record, schema()["$defs"]["config"], "$", errors)
    else:
        _check_trajectory(run.record, "$", errors)
    return errors
