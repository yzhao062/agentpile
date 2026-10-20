"""Pydantic AI: configuration records and capture.

Capture works on the agent's instrumentation setting, the one place Pydantic
AI reads which tracer an agent run uses. On entry the setting is replaced by a
copy of the host's own setting (or by a fresh one when the host had none) that
points at the capture's tracer provider and records content. A host that had
its own provider keeps receiving every span: the capture provider forwards
each span to the host provider's span processor, because one setting binds one
tracer. On exit the original setting is put back, unless the host changed it
in the meantime.

Three places can hold the host's setting, and each is handled:

- ``agent.instrument = InstrumentationSettings(...)``
- an ``Instrumentation`` capability passed to the agent (it wins over the
  attribute, so its ``settings`` field is the one swapped)
- ``Agent.instrument_all(...)`` (class level; the per-agent attribute set by
  the capture takes precedence over it for the run, and the class value is
  never touched)

Which runs the capture takes. Only a run of a selected agent (an agent passed to
``capture()``) goes through the capture's tracer, because the swap is made on
that agent alone: an unselected agent keeps whatever tracer it had, and its runs
are neither stored nor given a configuration. The one place where the setting is
shared is an ``Instrumentation`` capability instance passed to several agents;
the capture then installs a per-run hook on that instance (``_PerAgentRuns``) so
that each run takes the settings of its own agent, the capture's for a selected
agent and the host's for any other. A selected agent calling an unselected one
from a tool does not capture the callee's spans, which stay with the callee's own
tracer; select the callee as well to record them.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Tuple

import pydantic_ai
from pydantic_ai import InstrumentationSettings
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.toolsets import FunctionToolset

from ._common import function_definition, new_config, unique

FORMAT = "pydantic-ai"


def _framework_version() -> str:
    try:
        from importlib import metadata

        return metadata.version("pydantic-ai-slim")
    except Exception:
        return str(getattr(pydantic_ai, "__version__", "unknown"))


# -- configuration -------------------------------------------------------------

def _agent_name(agent: Any) -> str:
    # an unnamed agent is reported as "agent" in its spans, the same fallback
    return getattr(agent, "name", None) or "agent"


def _model_name(model: Any) -> Any:
    if model is None or isinstance(model, str):
        return model
    for attr in ("model_id", "model_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


def _instructions(agent: Any, lost: List[str], agent_id: str) -> Any:
    parts: List[str] = [p for p in getattr(agent, "_system_prompts", ()) if isinstance(p, str)]
    for sourced in getattr(agent, "_instructions", ()):
        text = getattr(sourced, "instruction", None)
        if isinstance(text, str) and not getattr(sourced, "dynamic", False):
            parts.append(text)
        else:
            lost.append(f"agents.{agent_id}.instructions: a dynamic instruction function is not evaluated")
    if getattr(agent, "_system_prompt_functions", None) or getattr(agent, "_system_prompt_dynamic_functions", None):
        lost.append(f"agents.{agent_id}.system_prompt: a dynamic system prompt function is not evaluated")
    if getattr(agent, "_cap_instructions", None):
        lost.append(f"agents.{agent_id}.instructions: instructions contributed by capabilities are not recorded")
    return "\n\n".join(parts) if parts else None


def _tool_entry(agent_id: str, definition: Any, kind: str) -> Dict[str, Any]:
    return {
        "name": definition.name,
        "agent_or_server": agent_id,
        "type": kind,
        "definition": function_definition(definition.name, definition.description,
                                          definition.parameters_json_schema, getattr(definition, "strict", None)),
        "definition_format": "openai",
        "atif_index": None,
        "permission": "unknown",
        "permission_from": "unknown",
    }


def config_record(agents: List[Any]) -> Dict[str, Any]:
    """An ``agentpile-config/0`` record for Pydantic AI agents."""
    lost: List[str] = []
    ids: Dict[str, Any] = {}
    out_agents: List[Dict[str, Any]] = []
    tools: Dict[str, Dict[str, Any]] = {}
    for agent in agents:
        agent_id = unique(_agent_name(agent), ids)
        ids[agent_id] = agent
        tool_ids: List[str] = []
        for toolset in agent.toolsets:
            if isinstance(toolset, FunctionToolset):
                for name, tool in toolset.tools.items():
                    tid = unique(f"{agent_id}.{name}", tools)
                    tools[tid] = _tool_entry(agent_id, tool.tool_def, getattr(tool.tool_def, "kind", None) or "function")
                    tool_ids.append(tid)
            else:
                label = getattr(toolset, "label", None) or type(toolset).__name__
                lost.append(f"agents.{agent_id}.toolsets[{label}]: the tool list is read at run time, not recorded here")
        output = getattr(agent, "_output_toolset", None)
        for definition in getattr(output, "_tool_defs", None) or []:
            tid = unique(f"{agent_id}.{definition.name}", tools)
            tools[tid] = _tool_entry(agent_id, definition, "output")
            tool_ids.append(tid)
        out_agents.append({
            "id": agent_id,
            "name": getattr(agent, "name", None),
            "instructions": _instructions(agent, lost, agent_id),
            "model": _model_name(getattr(agent, "model", None)),
            "tools": tool_ids,
            "handoffs": [],
        })
    record = new_config(FORMAT, out_agents[0]["id"] if out_agents else None, lost,
                        {"framework": FORMAT, "framework_version": _framework_version()})
    record["agents"] = out_agents
    record["tools"] = tools
    return record


# -- capture -------------------------------------------------------------------

_ABSENT = object()


def _identities(agent: Any) -> List[int]:
    """The ids of an agent and of the agents it wraps: a run of a wrapper reaches the capability as the wrapped agent."""
    ids: List[int] = []
    while agent is not None and id(agent) not in ids:
        ids.append(id(agent))
        agent = getattr(agent, "wrapped", None)
    return ids


class _PerAgentRuns:
    """The ``for_run`` of an ``Instrumentation`` capability while captures hold it: each run takes its own agent's settings.

    A capture swaps the capability's ``settings``, but one capability instance
    can be passed to several agents, and the swap alone would send the runs of
    agents the capture was not given through the capture's tracer. Pydantic AI
    copies the capability for every run through ``for_run``, so this wrapper
    sets the copy's settings from the agent running it: the capture's for a
    selected agent, the host's for any other. It is installed on the capability
    instance, removed again when the last selected agent leaves, and then the
    instance is as it was found.
    """

    def __init__(self, carrier: Any) -> None:
        self.carrier = carrier
        self.host = carrier.settings  # what every agent sharing the capability ran with before the capture
        self.previous = carrier.__dict__.get("for_run", _ABSENT)
        self._inner = carrier.for_run
        self.ours: Dict[int, Any] = {}  # id of a selected agent (and of what it wraps) -> the settings its runs take

    async def __call__(self, ctx: Any) -> Any:
        run = await self._inner(ctx)
        run.settings = self.ours.get(id(getattr(ctx, "agent", None)), self.host)
        return run

    def add(self, agent: Any, ours: Any) -> None:
        for key in _identities(agent):
            self.ours[key] = ours
        self.carrier.settings = ours
        self.carrier.for_run = self

    def remove(self, agent: Any) -> None:
        mine = None
        for key in _identities(agent):
            mine = self.ours.pop(key, mine)
        carrier = self.carrier
        if carrier.settings is mine:  # a setting the host replaced during the capture is left alone
            carrier.settings = next(iter(self.ours.values()), self.host)
        if self.ours or carrier.__dict__.get("for_run") is not self:
            return
        if self.previous is _ABSENT:
            del carrier.__dict__["for_run"]
        else:
            carrier.for_run = self.previous


class Binding:
    """Swap one agent's instrumentation for the capture's, and swap it back."""

    def __init__(self, agent: Any, session: Any) -> None:
        self.agent = agent
        self.session = session
        self._undo: List[Tuple[Any, str, Any, Any]] = []  # (owner, attribute, original, ours)
        self._selectors: List[_PerAgentRuns] = []

    def _host_settings(self, carriers: List[Any]) -> Any:
        """The settings the agent runs with before the capture: a capability's, the agent's, or the class default."""
        if carriers:
            held = carriers[0].__dict__.get("for_run")
            return held.host if isinstance(held, _PerAgentRuns) else carriers[0].settings
        own = self.agent.instrument
        if own is None:
            own = getattr(type(self.agent), "_instrument_default", False)
        return own

    def enter(self) -> None:
        from opentelemetry import trace as trace_api  # the SDK-side imports wait until a capture needs them
        from opentelemetry.metrics import NoOpMeterProvider

        from ._otel import host_span_processor

        agent = self.agent
        leaves: List[Any] = []
        agent.root_capability.apply(leaves.append)
        carriers = [c for c in leaves if isinstance(c, Instrumentation)]
        host = self._host_settings(carriers)

        forward_to, resource = None, None
        if isinstance(host, InstrumentationSettings):
            tracer = host.tracer
        elif host is True:  # instrument_all(True): the global provider, which the capture only reads
            tracer = trace_api.get_tracer_provider().get_tracer("pydantic-ai", pydantic_ai.__version__)
        else:
            tracer = None
        if tracer is not None:
            kind, forward_to, resource = host_span_processor(tracer)
            if kind == "unknown":
                warnings.warn(
                    "agentpile.capture: the agent's existing tracer is not an OpenTelemetry SDK tracer, so its "
                    "exporters will not receive the spans of captured runs", RuntimeWarning, stacklevel=4)

        if forward_to is not None and isinstance(host, InstrumentationSettings) and not host.include_content:
            warnings.warn(
                "agentpile.capture: the agent's own instrumentation has content recording switched off; the capture "
                "records content, and the spans forwarded to the host's exporters carry it for the length of the "
                "capture", RuntimeWarning, stacklevel=4)

        provider = self.session.tracer_provider(resource=resource, forward_to=forward_to)
        if isinstance(host, InstrumentationSettings):
            ours = copy.copy(host)  # keeps the host's meter, version and content flags
            ours.tracer = provider.get_tracer("pydantic-ai", pydantic_ai.__version__)
            ours.include_content = True
        else:
            ours = InstrumentationSettings(tracer_provider=provider, meter_provider=NoOpMeterProvider(),
                                           include_content=True)

        if carriers:
            for carrier in carriers:
                held = carrier.__dict__.get("for_run")
                selector = held if isinstance(held, _PerAgentRuns) else _PerAgentRuns(carrier)
                selector.add(agent, ours)
                self._selectors.append(selector)
        else:
            self._undo.append((agent, "instrument", agent.instrument, ours))
            agent.instrument = ours

    def exit(self) -> None:
        for selector in reversed(self._selectors):
            selector.remove(self.agent)
        self._selectors = []
        for owner, attr, original, ours in reversed(self._undo):
            if getattr(owner, attr) is ours:  # leave a setting the host replaced during the capture alone
                setattr(owner, attr, original)
        self._undo = []
