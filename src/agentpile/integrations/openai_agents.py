"""OpenAI Agents SDK: configuration records and capture.

Capture adds one small trace processor of agentpile's own to the SDK's
process-wide processor list and removes that processor again on exit. The
processor forwards each SDK trace and span event to the OpenInference
``OpenInferenceTracingProcessor``, bound to a tracer of the capture's own
provider, which writes the OpenInference span attributes (``openinference.span.kind``,
``llm.*``, ``tool.*``). The host's processors (the default OpenAI exporter,
its own observers) are neither replaced nor reordered, so they receive every
event natively, exactly once.

Which traces the capture takes. The SDK has one processor list for the whole
process, so the processor sees every trace that has events while the block is
open, selected agents or not. It forwards a trace to the OpenInference processor
only when an agent span of that trace names a selected agent (an agent passed to
``capture()``): a run the selected agent starts, a run that hands off to it, and
a run that calls it as a tool all qualify, and the whole trace is taken, so the
other agents of such a run appear in it. A trace whose agent spans name no
selected agent is dropped when it ends, and the configuration is never attached
to it. The SDK's spans name an agent by its ``name`` and carry nothing else, so
two agents with one name cannot be told apart, and a run of an unselected agent
that shares a selected agent's name is taken. The events of a trace are held
until its first selected agent span shows up (a trace opens before any agent span
exists) and are then passed on in their original order, with the root span opened
at the time the trace really started; a trace that never shows one has its held
events discarded when it ends, and the capture holds them only for that long. A
trace that began before the block opened is taken from its first event inside
the block, without the root span the SDK announced before the capture was there.

Why not ``OpenAIAgentsInstrumentor.instrument()`` plus a restored snapshot of
the processor list: ``instrument()`` is a process-wide singleton. When the host
has already instrumented, a second call does nothing, so the capture would see
no span, and ``uninstrument()`` would then remove the host's own patches. The
default ``exclusive_processor=True`` also replaces the whole processor list,
and ``uninstrument()`` restores nothing, so the capture would have to restore
the list itself, which loses any processor another thread registered in the
window. The processor route touches only the one list entry the capture added.
What the route gives up: the instrumentor's patch of the SDK's tool execution
step, which adds ``tool.description`` and ``tool.parameters`` to function
spans. ``config_from`` records the full tool definitions instead.
"""

from __future__ import annotations

import contextlib
import inspect
import threading
import time
import warnings
from typing import Any, Dict, Iterator, List, Optional, Set

import agents
from agents.handoffs import Handoff, handoff
from agents.tracing import TracingProcessor, add_trace_processor, get_trace_provider

from .._capture import CaptureError
from ._common import clean, function_definition, new_config, unique

FORMAT = "openai-agents"
SCOPE = "openinference.instrumentation.openai_agents"  # the scope name the OpenInference instrumentor itself uses


def _framework_version() -> str:
    try:
        from importlib import metadata

        return metadata.version("openai-agents")
    except Exception:
        return "unknown"


# -- configuration -------------------------------------------------------------

def _model_name(model: Any) -> Any:
    if model is None or isinstance(model, str):
        return model
    name = getattr(model, "model", None)
    return name if isinstance(name, str) and name else type(model).__name__


def _entry(agent_id: str, name: str, kind: str, definition: Any, fmt: str) -> Dict[str, Any]:
    return {"name": name, "agent_or_server": agent_id, "type": kind, "definition": definition,
            "definition_format": fmt, "atif_index": None, "permission": "unknown", "permission_from": "unknown"}


def _reachable(roots: List[Any]) -> List[Any]:
    """The given agents and every agent a handoff reaches, each once, in discovery order."""
    seen: Dict[int, Any] = {}
    queue = list(roots)
    while queue:
        agent = queue.pop(0)
        if id(agent) in seen:
            continue
        seen[id(agent)] = agent
        for h in getattr(agent, "handoffs", None) or []:
            target = h if isinstance(h, agents.Agent) else (h._agent_ref() if getattr(h, "_agent_ref", None) else None)
            if target is not None:
                queue.append(target)
    return list(seen.values())


def config_record(roots: List[Any]) -> Dict[str, Any]:
    """An ``agentpile-config/0`` record for Agents SDK agents and the agents their handoffs reach."""
    lost: List[str] = []
    members = _reachable(roots)
    ids: Dict[str, Any] = {}
    id_of: Dict[int, str] = {}
    for agent in members:
        aid = unique(agent.name, ids)
        ids[aid] = agent
        id_of[id(agent)] = aid

    out_agents: List[Dict[str, Any]] = []
    tools: Dict[str, Dict[str, Any]] = {}
    extras: Dict[str, Dict[str, Any]] = {}
    for agent in members:
        aid = id_of[id(agent)]
        tool_ids: List[str] = []
        handoff_ids: List[str] = []
        for tool in getattr(agent, "tools", None) or []:
            name = getattr(tool, "name", None) or type(tool).__name__
            tid = unique(f"{aid}.{name}", tools)
            if getattr(tool, "_is_agent_tool", False):
                lost.append(f"agents.{aid}.tools[{name}]: wraps another agent (as_tool), whose configuration is "
                            "recorded only if that agent is passed in as well")
            if isinstance(tool, agents.FunctionTool):
                definition = function_definition(tool.name, tool.description, tool.params_json_schema,
                                                 tool.strict_json_schema)
                tools[tid] = _entry(aid, tool.name, "function", definition, "openai")
            else:  # a hosted tool (web search, file search, code interpreter, ...): its settings, not a function schema
                tools[tid] = _entry(aid, name, "hosted", {"type": type(tool).__name__, "config": clean(tool)},
                                    "openai-agents")
            tool_ids.append(tid)
        for h in getattr(agent, "handoffs", None) or []:
            spec = handoff(h) if isinstance(h, agents.Agent) else h
            if not isinstance(spec, Handoff):
                lost.append(f"agents.{aid}.handoffs: an entry of type {type(h).__name__} is not recorded")
                continue
            target = h if isinstance(h, agents.Agent) else (spec._agent_ref() if spec._agent_ref else None)
            if target is not None and id(target) in id_of:
                handoff_ids.append(id_of[id(target)])
            else:
                handoff_ids.append(spec.agent_name)
                lost.append(f"agents.{aid}.handoffs[{spec.agent_name}]: the target agent object is not reachable, "
                            "so its configuration is not recorded")
            tid = unique(f"{aid}.{spec.tool_name}", tools)
            tools[tid] = _entry(aid, spec.tool_name, "handoff",
                                function_definition(spec.tool_name, spec.tool_description, spec.input_json_schema,
                                                    spec.strict_json_schema), "openai")
            tool_ids.append(tid)
        for server in getattr(agent, "mcp_servers", None) or []:
            label = getattr(server, "name", None) or type(server).__name__
            lost.append(f"agents.{aid}.mcp_servers[{label}]: the tool list is read from the server at run time, "
                        "not recorded here")
        instructions = getattr(agent, "instructions", None)
        if callable(instructions):
            lost.append(f"agents.{aid}.instructions: a dynamic instruction function is not evaluated")
            instructions = None
        elif not isinstance(instructions, str):
            instructions = None
        out_agents.append({"id": aid, "name": agent.name, "instructions": instructions,
                           "model": _model_name(getattr(agent, "model", None)), "tools": tool_ids,
                           "handoffs": handoff_ids})
        settings = clean(getattr(agent, "model_settings", None))
        extra = {
            "handoff_description": getattr(agent, "handoff_description", None),
            "output_type": clean(getattr(agent, "output_type", None)),
            "tool_use_behavior": clean(getattr(agent, "tool_use_behavior", None)),
            "model_settings": {k: v for k, v in settings.items() if v is not None} if isinstance(settings, dict) else None,
        }
        extra = {k: v for k, v in extra.items() if v not in (None, {}, [])}
        if extra:
            extras[aid] = extra

    record = new_config(FORMAT, out_agents[0]["id"] if out_agents else None, lost,
                        {"framework": FORMAT, "framework_version": _framework_version(), "agents": extras})
    record["agents"] = out_agents
    record["tools"] = tools
    return record


# -- capture -------------------------------------------------------------------

def _trace_config() -> Any:
    """An OpenInference trace config that hides nothing (the library default reads environment variables)."""
    from openinference.instrumentation import TraceConfig

    params = inspect.signature(TraceConfig.__init__).parameters
    return TraceConfig(**{name: False for name in params if name.startswith("hide_")})


def _agent_names(span: Any) -> tuple:
    """The agent names a span records: an agent span names its agent, other spans none that identify a participant."""
    data = getattr(span, "span_data", None)
    if getattr(data, "type", None) == "agent":
        return (getattr(data, "name", None),)
    return ()


class _RootClock:
    """The tracer the OpenInference processor gets: it opens a replayed trace's root span at the trace's real start time.

    A trace is forwarded only after its first selected agent span, so the root
    span would otherwise open later than the spans under it.
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer
        self._local = threading.local()

    @contextlib.contextmanager
    def root_started_at(self, nanoseconds: int) -> Iterator[None]:
        self._local.start_ns = nanoseconds
        try:
            yield
        finally:
            self._local.start_ns = None

    def start_span(self, *args: Any, **kwargs: Any) -> Any:
        start_ns = getattr(self._local, "start_ns", None)
        if start_ns is not None and kwargs.get("start_time") is None:
            kwargs["start_time"] = start_ns
        return self._tracer.start_span(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)


class _Held:
    """The events of one trace in which no selected agent has appeared yet, in the order they arrived."""

    __slots__ = ("started_ns", "events")

    def __init__(self, trace: Any = None) -> None:
        self.started_ns = time.time_ns()
        self.events: List[Any] = [] if trace is None else [("on_trace_start", trace)]


_TAKEN = object()  # the state of a trace in which a selected agent takes part
_DROPPED = object()  # the state of a trace that ended without one; its late events are ignored


class _Processor(TracingProcessor):
    """The capture's entry in the SDK's processor list; inert once the capture has closed.

    ``names`` holds the names of the selected agents (the set grows as bindings
    enter). A trace is held from its first event; the first agent span that
    names a selected agent marks the trace as taken and replays everything held,
    in order, into the OpenInference processor. Every later event of a taken
    trace goes straight through. A trace that ends while still held is dropped.
    """

    def __init__(self, inner: TracingProcessor, session: Any, names: Set[str], clock: _RootClock) -> None:
        self._inner = inner
        self._session = session
        self._names = names
        self._clock = clock
        self._lock = threading.Lock()
        # trace id -> _Held, _TAKEN or _DROPPED; a taken or dropped trace stays listed so that a late event is routed right
        self._traces: Dict[str, Any] = {}
        self.active = True

    def _live(self) -> bool:
        return self.active and not self._session.closed

    def close(self) -> None:
        """Stop forwarding and let go of every held event."""
        self.active = False
        with self._lock:
            self._traces.clear()

    def _selects(self, obj: Any) -> bool:
        return any(isinstance(name, str) and name in self._names for name in _agent_names(obj))

    def _replay(self, held: _Held) -> None:
        """Pass the held events to the OpenInference processor; one failing event does not lose the rest.

        The OpenInference processor makes each span it opens the current
        OpenTelemetry span and expects to close it in the same context. The
        spans that opened before the trigger belong to contexts that are gone
        (the run may have moved into a tool's task since), so their hold on the
        current context is released at once; the trigger keeps it.
        """
        failure: Optional[Exception] = None
        last = len(held.events) - 1
        for index, (kind, obj) in enumerate(held.events):
            try:
                if kind == "on_trace_start":
                    with self._clock.root_started_at(held.started_ns):
                        self._inner.on_trace_start(obj)
                else:
                    getattr(self._inner, kind)(obj)
                    if kind == "on_span_start" and index < last:
                        self._release_context(obj)
            except Exception as exc:  # the SDK logs a processor's failure; raise the first once the replay is done
                failure = failure or exc
        held.events = []
        if failure is not None:
            raise failure

    def _release_context(self, span: Any) -> None:
        tokens = getattr(self._inner, "_tokens", None)
        token = tokens.pop(span.span_id, None) if isinstance(tokens, dict) else None
        if token is not None:
            from opentelemetry.context import detach

            detach(token)

    def _route(self, kind: str, obj: Any, trace_id: str) -> None:
        if not self._live():
            return
        with self._lock:
            state = self._traces.get(trace_id)
            if state is _DROPPED:
                return
            if state is None:
                state = self._traces[trace_id] = _Held()  # a trace that began before the capture: held from here on
            if state is not _TAKEN:
                state.events.append((kind, obj))
                if kind == "on_trace_end":
                    self._traces[trace_id] = _DROPPED  # no selected agent took part in it
                elif self._selects(obj):
                    self._traces[trace_id] = _TAKEN
                    self._replay(state)
                return
        getattr(self._inner, kind)(obj)

    def on_trace_start(self, trace: Any) -> None:
        if self._live():
            with self._lock:
                self._traces[trace.trace_id] = _Held(trace)

    def on_trace_end(self, trace: Any) -> None:
        self._route("on_trace_end", trace, trace.trace_id)

    def on_span_start(self, span: Any) -> None:
        self._route("on_span_start", span, span.trace_id)

    def on_span_end(self, span: Any) -> None:
        self._route("on_span_end", span, span.trace_id)

    def shutdown(self) -> None:  # the SDK shuts every processor down at exit; the capture's has nothing to release
        return None

    def force_flush(self) -> None:
        return None


def _remove_processor(provider: Any, processor: Any) -> bool:
    """Take one processor out of the SDK's list, leaving every other entry (and its order) alone."""
    multi = getattr(provider, "_multi_processor", None)
    current = getattr(multi, "_processors", None)
    if current is None:
        return False
    provider.set_processors([p for p in current if p is not processor])
    return True


class Binding:
    """Attach the capture's processor to the SDK once per capture, however many agents it covers."""

    def __init__(self, agent: Any, session: Any) -> None:
        self.agent = agent
        self.session = session
        self._entered = False

    def enter(self) -> None:
        shared = self.session.shared.setdefault(FORMAT, {"users": 0, "processor": None, "provider": None,
                                                         "names": set()})
        name = getattr(self.agent, "name", None)
        if isinstance(name, str):
            shared["names"].add(name)  # the processor reads this set, so it covers every agent of the capture
        if shared["users"] == 0:
            provider = get_trace_provider()
            refresh = getattr(provider, "_refresh_disabled_flag", None)
            if callable(refresh):
                refresh()
            if getattr(provider, "_disabled", False):
                raise CaptureError(
                    "OpenAI Agents SDK tracing is switched off (OPENAI_AGENTS_DISABLE_TRACING or "
                    "set_tracing_disabled), so no span would be recorded. Turning it on for the capture would also "
                    "start the SDK's default exporter, which sends traces to OpenAI; enable tracing yourself, with "
                    "your own processors in place, before capturing.")
            try:  # needed to capture, not to read a configuration
                from openinference.instrumentation import OITracer
                from openinference.instrumentation.openai_agents._processor import OpenInferenceTracingProcessor
                from openinference.instrumentation.openai_agents.version import __version__ as oi_version
            except ImportError as exc:
                raise CaptureError(
                    f"capturing an Agents SDK run needs the OpenInference instrumentor ({exc}); install it with: "
                    "pip install 'agentpile[openai-agents]'") from exc
            otel_provider = self.session.tracer_provider()
            tracer = OITracer(otel_provider.get_tracer(SCOPE, oi_version), config=_trace_config())
            clock = _RootClock(tracer)
            processor = _Processor(OpenInferenceTracingProcessor(clock), self.session, shared["names"], clock)
            add_trace_processor(processor)
            shared.update(processor=processor, provider=provider)
        shared["users"] += 1
        self._entered = True

    def exit(self) -> None:
        if not self._entered:
            return
        self._entered = False
        shared = self.session.shared[FORMAT]
        shared["users"] -= 1
        if shared["users"] > 0:
            return
        shared["processor"].close()  # inert at once, even if it cannot be removed below
        if not _remove_processor(shared["provider"], shared["processor"]):
            warnings.warn("agentpile.capture: the SDK's processor list could not be read, so the capture's processor "
                          "stays registered (inert) until the process ends", RuntimeWarning, stacklevel=3)
