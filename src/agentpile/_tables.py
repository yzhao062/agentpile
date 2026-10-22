"""Flat tables of runs, for pandas, csv and the plotting tools people already use.

``table(runs, level)`` returns one list of plain dicts per level (runs, events or
labels). Every value is a scalar (``None``, bool, int, float or str), and a value
that is not one (a dict, a list) is JSON text, so the rows go straight into
``pandas.DataFrame``, ``csv.DictWriter`` or a database. Every row of a level has
the same keys. ``to_pandas`` builds the DataFrame.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from .record import MISSING, Event, Run, _ext

LEVELS = ("runs", "events", "labels")

RUN_COLUMNS = ["run_id", "kind", "dataset", "config", "revision", "file", "format", "task_id", "n_steps",
               "n_messages", "n_calls", "n_results", "n_agents", "n_tools", "n_labels"]
EVENT_COLUMNS = ["run_id", "index", "kind", "role", "actor", "step", "call", "result", "source_call_id", "name",
                 "content", "error", "content_state", "id_generated", "name_stand_in", "paired_by"]
LABEL_COLUMNS = ["run_id", "name", "value", "target_kind", "trajectory", "step", "call", "result", "span", "agent",
                 "tool", "label_source"]

COLUMNS = {"runs": RUN_COLUMNS, "events": EVENT_COLUMNS, "labels": LABEL_COLUMNS}
INT_COLUMNS = {
    "runs": ["n_steps", "n_messages", "n_calls", "n_results", "n_agents", "n_tools", "n_labels"],
    "events": ["index", "step", "result"],
    "labels": ["step", "result"],
}


def _cell(value: Any) -> Any:
    """A table cell: a scalar as it is, ``None`` for a value the source did not record, anything else as JSON text."""
    if value is MISSING:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _text_cell(value: Any) -> Optional[str]:
    """Event content as a cell: text as it is, ``None`` for null or absent, anything else as JSON text."""
    if value is None or value is MISSING:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _iter_runs(runs: Union[Run, Iterable[Run]]) -> Iterator[Run]:
    if isinstance(runs, Run):
        yield runs
        return
    try:
        items = iter(runs)
    except TypeError:
        raise TypeError(f"expected a Run or an iterable of Run objects, got {type(runs).__name__}") from None
    for item in items:
        if not isinstance(item, Run):
            raise TypeError(f"expected a Run or an iterable of Run objects, found {type(item).__name__} in it")
        yield item


def _run_row(run: Run) -> Dict[str, Any]:
    src = run.source
    counts = Counter(event.kind for event in run.events())
    row: Dict[str, Any] = {
        "run_id": run.id, "kind": run.kind, "dataset": src.get("dataset"), "config": src.get("config"),
        "revision": src.get("revision"), "file": src.get("file"), "format": src.get("format"),
        "task_id": run.task.get("id"), "n_steps": len(run.steps), "n_messages": counts["message"],
        "n_calls": counts["call"], "n_results": counts["result"], "n_agents": len(run.agents),
        "n_tools": len(run.tools), "n_labels": len(run.labels),
    }
    for key, value in run.outcome.items():
        row["outcome." + key] = _cell(value)
    return row


def _event_markers(event: Event, sx: Dict[str, Any]) -> Tuple[Optional[str], bool, bool, Optional[str]]:
    """``(content_state, id_generated, name_stand_in, paired_by)`` as the record marks them for one event.

    ``content_state`` is ``"null"`` or ``"missing"`` where the source value was null or absent, ``"unparsed"``
    for call arguments that were not a JSON object, ``"rebuilt"`` for arguments the reader reconstructed as
    ``{}``, and ``"raw"`` where ATIF could not hold the source content and the event carries the source value.
    """
    state: Optional[str] = None
    generated = stand_in = False
    paired: Optional[str] = None
    raw = False
    if event.kind == "message":
        raw = "message_raw" in sx
    elif event.kind == "call":
        cx = (sx.get("calls") or {}).get(event.call_id) or {}
        generated = bool(cx.get("id_generated"))
        stand_in = "function_name_raw" in cx
        paired = cx.get("paired_by")
        if cx.get("arguments_state") == "unparsed":
            state = "unparsed"
        elif "arguments_stand_in" in cx:
            state = "rebuilt"
    else:
        fields = sx.get("result_fields") or []
        i = event.address.get("result")
        rf = (fields[i] if isinstance(i, int) and i < len(fields) else None) or {}
        raw = "content_raw" in rf
        paired = rf.get("paired_by")
    if event.content is MISSING:
        state = "missing"
    elif event.content is None:
        state = "null"
    elif state is None and raw:
        state = "raw"
    return state, generated, stand_in, paired


def _event_rows(run: Run) -> List[Dict[str, Any]]:
    steps = {step.get("step_id"): step for step in run.steps}
    rows: List[Dict[str, Any]] = []
    for index, event in enumerate(run.events()):
        address = event.address
        sx = _ext(steps.get(address.get("step")))
        state, generated, stand_in, paired = _event_markers(event, sx)
        rows.append({
            "run_id": run.id, "index": index, "kind": event.kind, "role": event.role, "actor": event.actor,
            "step": address.get("step"), "call": address.get("call"), "result": address.get("result"),
            "source_call_id": event.call_id if event.kind == "result" else None, "name": event.name,
            "content": _text_cell(event.content), "error": _cell(event.error), "content_state": state,
            "id_generated": generated, "name_stand_in": stand_in, "paired_by": paired,
        })
    return rows


def _label_rows(run: Run) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label in run.labels:
        target = label.get("target") or {}
        address = target.get("address") or {}
        rows.append({  # run_id is the record holding the label; trajectory is the one it targets (a subagent's, say)
            "run_id": run.id, "name": label.get("name"), "value": _cell(label.get("value")),
            "target_kind": target.get("kind"), "trajectory": address.get("trajectory") or run.id,
            "step": address.get("step"), "call": address.get("call"),
            "result": address.get("result"), "span": address.get("span"), "agent": address.get("agent"),
            "tool": address.get("tool"), "label_source": label.get("label_source"),
        })
    return rows


def _uniform(rows: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    """Rows that share one set of keys: ``columns`` first, then any other key in order of first appearance."""
    keys = list(columns)
    seen = set(keys)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return [{key: row.get(key) for key in keys} for row in rows]


def table(runs: Union[Run, Iterable[Run]], level: str = "events") -> List[Dict[str, Any]]:
    """Flat rows for one run or many, at one level; the standard library only.

    ``runs`` is a ``Run`` or an iterable of them (a generator from ``load`` is read once).

    - ``"runs"``: one row per run, with its source, the sizes of the record (``n_steps``, ``n_messages``,
      ``n_calls``, ``n_results``, ``n_agents``, ``n_tools``, ``n_labels``) and the outcome fields the
      record carries as ``outcome.*`` (a nested value is JSON text; a run without one has ``None``).
    - ``"events"``: one row per event of ``run.events()`` in source order. ``step``, ``call`` and ``result``
      come from the event address, and a result also names its call in ``source_call_id``. Content is text,
      with a non-text value as JSON; ``error`` is a scalar or JSON text. Stand-ins are marked in
      ``content_state`` (``null``, ``missing``, ``unparsed``, ``rebuilt`` or ``raw``, as the record states
      them), ``id_generated`` and ``name_stand_in`` (flags of a call), and ``paired_by`` (the rule by which
      a reader tied a result, or a call, to a place the source does not state).
    - ``"labels"``: one row per label with its value (JSON text if not a scalar), ``target_kind``, the
      target address fields and ``label_source``.

    A config record has a row at the ``runs`` and ``labels`` levels and no events.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(repr(name) for name in LEVELS)}, got {level!r}")
    build = {"runs": lambda run: [_run_row(run)], "events": _event_rows, "labels": _label_rows}[level]
    rows: List[Dict[str, Any]] = []
    for run in _iter_runs(runs):
        rows.extend(build(run))
    return _uniform(rows, COLUMNS[level]) if level == "runs" else rows


def to_pandas(runs: Union[Run, Iterable[Run]], level: str = "events"):
    """``table(runs, level)`` as a ``pandas.DataFrame``; counts and positions are nullable ``Int64`` columns."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("to_pandas needs the pandas extra: pip install 'agentpile[pandas]'") from exc
    rows = table(runs, level)
    frame = pd.DataFrame(rows, columns=list(rows[0]) if rows else COLUMNS[level])
    for column in INT_COLUMNS[level]:
        frame[column] = pd.to_numeric(frame[column]).astype("Int64")
    return frame
