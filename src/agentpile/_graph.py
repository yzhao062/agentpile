"""One trajectory as a ``networkx.MultiDiGraph``, for graph tools and plots.

Each event is a node keyed by its address (``step:3``, ``step:2/call:call_1``,
``step:2/result:0``). Edges follow the source order (``next``), tie a result to
its call (``result_of``) and, when the caller names dependency rules, carry the
entries ``derive_deps`` finds (``dep``).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ._tables import _text_cell
from .deps import Rule, derive_deps
from .record import CONFIG, Run, _ext


def _node_key(address: Dict[str, Any]) -> Optional[str]:
    """The node key of a record address, or ``None`` when it names no step."""
    step = address.get("step")
    if step is None:
        return None
    if address.get("call") is not None:
        return f"step:{step}/call:{address['call']}"
    if address.get("result") is not None:
        return f"step:{step}/result:{address['result']}"
    return f"step:{step}"


def _truncate(text: Optional[str], limit: Optional[int]) -> Optional[str]:
    if text is None or limit is None or len(text) <= limit:
        return text
    return text[:max(limit - 3, 0)] + "..."


def _free_key(graph: Any, u: str, v: str, base: str) -> str:
    """``base``, or ``base#2``, ``base#3`` and so on when that edge key is taken between ``u`` and ``v``."""
    key, n = base, 1
    while graph.has_edge(u, v, key):
        n += 1
        key = f"{base}#{n}"
    return key


def to_networkx(run: Run, rules: Union[None, str, Rule, Iterable[Union[str, Rule]]] = None, *,
                max_content: Optional[int] = 200):
    """The events of one trajectory as a ``networkx.MultiDiGraph``.

    **Nodes** are the events of ``run.events()``, keyed by address: ``step:3`` for a message,
    ``step:2/call:call_1`` for a call and ``step:2/result:0`` for a result. A step the source recorded with no
    message gets a node of kind ``step`` under its own key, so every step address (a label or a dependency)
    lands on a node. Attributes: ``kind``, ``role``, ``actor``, ``name``, ``step`` (the step id), ``content``
    (text, cut to ``max_content`` characters; ``None`` keeps all of it) and ``labels`` (copies of the labels that
    target the node, directly or through a span the record maps to it).

    **Edges** carry ``kind``. A ``next`` edge joins consecutive nodes in source order (its key is ``next``).
    A ``result_of`` edge runs from a result to the call it answers (key ``result_of``) and has ``paired_by``
    when the record marks the pairing as inferred. With ``rules`` (names or rule functions, as for
    ``derive_deps``) each dependency entry is one ``dep`` edge from ``src`` to ``dst`` (``dst`` depends on
    ``src``), keyed ``<rule>_v<version>``, with the entry's ``rule``, ``rule_version``, ``resource`` and, for
    resource rules, ``extractor``. Loading infers no dependency, so without ``rules`` the graph has none.

    ``graph.graph`` holds ``run_id`` and ``labels``, the labels that sit on no node (those of the run, an agent
    or a tool, and any label whose span the record does not map to an event). Subagent trajectories are not
    part of the graph, as ``Run.events`` leaves them out. A config record has no steps and raises ValueError.
    """
    if not isinstance(run, Run):
        raise TypeError(f"to_networkx takes a Run, got {type(run).__name__}")
    if run.kind == CONFIG:
        raise ValueError("to_networkx takes trajectory records; a config record has no steps to draw as a graph")
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("to_networkx needs the graph extra: pip install 'agentpile[graph]'") from exc

    graph = nx.MultiDiGraph(run_id=run.id, labels=[])
    timeline = run._timeline()
    with_message = {step.get("step_id") for step, event in timeline if event is not None and event.kind == "message"}
    previous: Optional[str] = None
    answers: List[Tuple[str, str, Optional[str]]] = []  # (result node, call node, paired_by) for when all nodes exist
    for step, event in timeline:
        sid = step.get("step_id")
        if event is None:
            if sid in with_message:
                continue
            key = _node_key({"step": sid})
            attrs = {"kind": "step", "role": step.get("source"), "actor": _ext(step).get("actor"), "name": None,
                     "content": None}
        else:
            key = _node_key(event.address)
            attrs = {"kind": event.kind, "role": event.role, "actor": event.actor, "name": event.name,
                     "content": _truncate(_text_cell(event.content), max_content)}
        base, n = key, 1
        while key in graph:  # a record that repeats a call id on one step keeps every event
            n += 1
            key = f"{base}#{n}"
        graph.add_node(key, step=sid, labels=[], **attrs)
        if event is not None and event.kind == "result" and event.call_id is not None:
            fields = _ext(step).get("result_fields") or []
            i = event.address["result"]
            paired = ((fields[i] if i < len(fields) else None) or {}).get("paired_by")
            answers.append((key, _node_key({"step": sid, "call": event.call_id}), paired))
        if previous is not None:
            graph.add_edge(previous, key, key="next", kind="next")
        previous = key
    for result, call, paired in answers:
        if call in graph:
            extra = {"paired_by": paired} if paired else {}
            graph.add_edge(result, call, key="result_of", kind="result_of", **extra)

    def locate(address: Any) -> Optional[str]:
        address = address or {}
        if address.get("trajectory") not in (None, run.id):
            return None
        key = _node_key(address)
        return key if key in graph else None

    spans = run.spans
    for label in run.labels:
        target = label.get("target") or {}
        address = target.get("address") or {}
        if target.get("kind") == "span":
            address = (spans.get(address.get("span")) or {}).get("address")
        key = locate(address)
        (graph.nodes[key]["labels"] if key is not None else graph.graph["labels"]).append(copy.deepcopy(label))

    if rules is not None:
        if isinstance(rules, str) or callable(rules):
            rules = [rules]
        for dep in derive_deps(run, rules):
            src, dst = locate(dep.get("src")), locate(dep.get("dst"))
            if src is None or dst is None:
                raise ValueError(f"dependency {dep!r} names an address this run has no node for")
            name = f"{dep.get('rule')}_v{dep.get('rule_version')}"
            key = graph.add_edge(src, dst, key=_free_key(graph, src, dst, name))
            graph[src][dst][key].update({k: copy.deepcopy(v) for k, v in dep.items() if k not in ("src", "dst")})
            graph[src][dst][key]["kind"] = "dep"
    return graph
