"""Standard-library helpers shared by the framework integrations (no framework or OpenTelemetry import)."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from ..record import CONFIG_SCHEMA

_DROP = object()
_MAX_DEPTH = 8


def jsonable(value: Any, _depth: int = 0) -> Any:
    """A JSON-safe copy of ``value``, or the ``_DROP`` marker for what cannot be copied (callables, handles, cycles)."""
    if _depth > _MAX_DEPTH:
        return _DROP
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else _DROP
    if isinstance(value, enum.Enum):
        return jsonable(value.value, _depth + 1)
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            item = jsonable(v, _depth + 1)
            if item is not _DROP:
                out[str(k)] = item
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [jsonable(v, _depth + 1) for v in value]
        return [i for i in items if i is not _DROP]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable({f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}, _depth + 1)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"), _depth + 1)
        except Exception:
            return _DROP
    if isinstance(value, type):
        return getattr(value, "__name__", _DROP)
    return _DROP


def clean(value: Any) -> Any:
    """``jsonable`` that returns ``None`` instead of the drop marker."""
    out = jsonable(value)
    return None if out is _DROP else out


def function_definition(name: str, description: Optional[str], parameters: Optional[Dict[str, Any]],
                        strict: Optional[bool] = None) -> Dict[str, Any]:
    """A tool definition in the OpenAI function-schema shape (the shape ATIF's ``agent.tool_definitions`` uses)."""
    fn: Dict[str, Any] = {"name": name}
    if description is not None:
        fn["description"] = description
    fn["parameters"] = clean(parameters) if parameters is not None else {"type": "object", "properties": {}}
    if strict is not None:
        fn["strict"] = bool(strict)
    return {"type": "function", "function": fn}


def new_config(fmt: str, record_key: Optional[str], lost: List[str], dataset: Dict[str, Any]) -> Dict[str, Any]:
    """An empty ``agentpile-config/0`` record for ``fmt``; the caller fills agents and tools."""
    return {
        "schema": CONFIG_SCHEMA,
        "source": {"format": fmt, "dataset": None, "revision": None, "record_key": record_key,
                   "reader": "agentpile.capture", "reader_version": "0", "lost_fields": lost},
        "task": {"id": None, "text": None},
        "agents": [],
        "tools": {},
        "labels": [],
        "dataset": dataset,
    }


def unique(base: str, taken: Mapping) -> str:
    """``base``, or ``base#2``, ``base#3`` ... when ``base`` is already a key of ``taken``."""
    if base not in taken:
        return base
    n = 2
    while f"{base}#{n}" in taken:
        n += 1
    return f"{base}#{n}"
