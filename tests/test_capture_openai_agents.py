"""Gate 5 for the OpenAI Agents SDK: a fake model and real tools, a host trace processor, and a lifecycle check.

No network and no key: a Model subclass plays the model and records its response the way the SDK's own models do.
The tests skip when the Agents SDK, the OpenInference instrumentor or the OpenTelemetry SDK is absent.
"""

import json
import time

import pytest

pytest.importorskip("agents.tracing")
pytest.importorskip("openinference.instrumentation.openai_agents")
pytest.importorskip("opentelemetry.sdk.trace")

from agents import Agent, Runner, function_tool  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.tracing import TracingProcessor, get_trace_provider, response_span, set_trace_processors  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from opentelemetry import trace as otel_trace  # noqa: E402

import agentpile as ap  # noqa: E402
from agentpile._capture import CaptureError  # noqa: E402


# -- fixtures --------------------------------------------------------------------

class FakeModel(Model):
    """Replies from a script, one entry per model call; an Exception entry is raised instead."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing,
                           *, previous_response_id=None, conversation_id=None, prompt=None):
        step = self.script[self.calls]
        self.calls += 1
        with response_span(disabled=tracing.is_disabled()) as span:
            if isinstance(step, Exception):
                raise step
            response = Response(id=f"resp_{self.calls}", created_at=time.time(), model="fake-model", object="response",
                                instructions=system_instructions, output=step, parallel_tool_calls=False,
                                tool_choice="auto", tools=[])
            if tracing.include_data():
                span.span_data.response = response
                span.span_data.input = input
        return ModelResponse(output=step, usage=Usage(), response_id=response.id)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


def call(name, arguments, call_id):
    return ResponseFunctionToolCall(id=call_id, call_id=call_id, type="function_call", name=name,
                                    arguments=json.dumps(arguments))


def message(text):
    return ResponseOutputMessage(id="msg_1", role="assistant", status="completed", type="message",
                                 content=[ResponseOutputText(type="output_text", text=text, annotations=[])])


class HostCounter(TracingProcessor):
    """The host's own trace processor: counts events, remembers ids, notes a shutdown."""

    def __init__(self):
        self.traces_started = self.traces_ended = self.spans_started = self.spans_ended = 0
        self.span_ids = []
        self.shutdowns = 0

    def on_trace_start(self, trace):
        self.traces_started += 1

    def on_trace_end(self, trace):
        self.traces_ended += 1

    def on_span_start(self, span):
        self.spans_started += 1

    def on_span_end(self, span):
        self.spans_ended += 1
        self.span_ids.append(span.span_id)

    def shutdown(self):
        self.shutdowns += 1

    def force_flush(self):
        return None

    @property
    def events(self):
        """Finished SDK spans and traces: the unit a capture turns into one OpenTelemetry span each."""
        return self.spans_ended + self.traces_ended


class Team:
    """triage (a real tool and a handoff) -> billing, each with its own scripted model."""

    def __init__(self, fail_on_second_call=False):
        self.executed = []

        @function_tool
        def get_weather(city: str) -> str:
            """Return the weather for a city."""
            self.executed.append(city)
            return f"sunny in {city}"

        self.billing = Agent(name="billing", instructions="Handle billing.",
                             handoff_description="Billing questions.", model=FakeModel([[message("billing done")]]))
        second = RuntimeError("the model call failed") if fail_on_second_call else [call("transfer_to_billing", {}, "call_h1")]
        self.triage = Agent(name="triage", instructions="Route requests.", tools=[get_weather],
                            handoffs=[self.billing],
                            model=FakeModel([[call("get_weather", {"city": "Paris"}, "call_w1")], second]))

    def reset(self):
        self.billing.model.calls = 0
        self.triage.model.calls = 0

    def run(self):
        self.reset()
        return Runner.run_sync(self.triage, "What is the weather in Paris, and who handles billing?")


@pytest.fixture
def sdk():
    """A clean processor list holding only the host's counter; the SDK's real state is put back afterwards."""
    provider = get_trace_provider()
    saved = (provider._multi_processor._processors, provider._manual_disabled, provider._env_disabled,
             provider._disabled)
    host = HostCounter()
    provider._manual_disabled = False
    provider._env_disabled = False
    provider._disabled = False
    set_trace_processors([host])
    yield host
    provider._multi_processor._processors = saved[0]
    provider._manual_disabled, provider._env_disabled, provider._disabled = saved[1:]


def processors():
    return tuple(get_trace_provider()._multi_processor._processors)


def span_list(doc):
    return [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]


def attrs(span):
    out = {}
    for a in span["attributes"]:
        value = a["value"]
        out[a["key"]] = value.get("stringValue", value.get("intValue", value.get("boolValue", value.get("doubleValue"))))
    return out


def ids(doc):
    return {s["spanId"] for s in span_list(doc)}


def stored(cap):
    """Spans the capture's own store holds right now; unlike cap.otlp this is not frozen when the block ends."""
    return len(cap._session.exporter.get_finished_spans())


GLOBAL_TRACER_PROVIDER = otel_trace.get_tracer_provider()  # as found at import


def assert_no_global_provider():
    """Capture never installs a tracer provider of its own as the global one."""
    assert otel_trace.get_tracer_provider() is GLOBAL_TRACER_PROVIDER


# -- what the document holds -------------------------------------------------------

def test_document_holds_calls_ids_agent_names_and_config(sdk):
    team = Team()
    with ap.capture(agent=team.triage) as cap:
        result = team.run()
    assert result.final_output == "billing done"
    assert team.executed == ["Paris"]  # the tool really ran

    spans = span_list(cap.otlp)
    by_kind = {}
    for s in spans:
        by_kind.setdefault(attrs(s)["openinference.span.kind"], []).append(s)

    agents = {attrs(s)["agent.name"]: attrs(s) for s in by_kind["AGENT"] if "agent.name" in attrs(s)}
    assert set(agents) == {"triage", "billing"}
    assert agents["billing"]["graph.node.parent_id"] == "triage"  # the handoff is visible as the parent agent

    (tool,) = [s for s in by_kind["TOOL"] if attrs(s).get("tool.name") == "get_weather"]
    assert json.loads(attrs(tool)["input.value"]) == {"city": "Paris"}
    assert attrs(tool)["output.value"] == "sunny in Paris"
    assert any(s["name"] == "handoff to billing" for s in by_kind["TOOL"])

    llm = [attrs(s) for s in by_kind["LLM"]]
    assert len(llm) == 3
    first = llm[0]
    assert first["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call_w1"
    assert first["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "get_weather"
    assert json.loads(first["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"]) == {"city": "Paris"}
    assert first["llm.model_name"] == "fake-model"
    assert json.loads(first["output.value"])["instructions"] == "Route requests."  # the system prompt the model saw
    second = llm[1]  # the result of the first call comes back with the same id
    assert second["llm.input_messages.3.message.tool_call_id"] == "call_w1"
    assert second["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == "call_h1"

    # the encoding: hex ids, string nanosecond times, one trace, parents inside the document
    for s in spans:
        assert len(s["traceId"]) == 32 and len(s["spanId"]) == 16
        assert isinstance(s["startTimeUnixNano"], str) and int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"])
    assert len({s["traceId"] for s in spans}) == 1
    assert all(s["parentSpanId"] in ids(cap.otlp) for s in spans if "parentSpanId" in s)
    assert [s["scope"]["name"] for rs in cap.otlp["resourceSpans"] for s in rs["scopeSpans"]] == [
        "openinference.instrumentation.openai_agents"]
    assert json.loads(json.dumps(cap.otlp, allow_nan=False)) == cap.otlp
    assert cap.span_count == sdk.events == 12
    assert_no_global_provider()

    cfg = cap.config
    assert ap.validate(cfg) == []
    assert {a["name"] for a in cfg.agents} == {"triage", "billing"}


# -- lifecycle ------------------------------------------------------------------------

def test_lifecycle_host_keeps_observing_and_state_is_restored(sdk):
    team = Team()
    host = sdk
    host_list = processors()
    assert host_list == (host,)
    other = otel_trace.get_tracer_provider()

    # (1) the host's own processor, before any capture
    team.run()
    per_run = host.events
    assert per_run == 12 and host.spans_started == host.spans_ended and host.traces_started == 1

    # (2) two captures back to back
    caps = []
    for n in (1, 2):
        before = host.events
        with ap.capture(team.triage) as cap:
            assert len(processors()) == 2 and processors()[0] is host  # the host's entry stays first, untouched
            team.run()
        assert host.events - before == per_run  # the host received every event of the captured run too
        assert cap.span_count == per_run  # the capture got its own run, no more
        assert len(ids(cap.otlp)) == per_run
        assert processors() == host_list  # the capture's processor is gone, the list is as it was
        caps.append(cap)
    first, second = caps
    assert ids(first.otlp).isdisjoint(ids(second.otlp))
    assert {s["traceId"] for s in span_list(first.otlp)}.isdisjoint({s["traceId"] for s in span_list(second.otlp)})
    snapshot = [json.dumps(c.otlp, sort_keys=True) for c in caps]

    # (3) a run outside any capture: the host still observes, the finished captures receive nothing
    before = host.events
    team.run()
    assert host.events - before == per_run
    assert [json.dumps(c.otlp, sort_keys=True) for c in caps] == snapshot
    assert [stored(c) for c in caps] == [per_run, per_run]

    # (4) a capture that exits by exception
    before = host.events
    with pytest.raises(ZeroDivisionError):
        with ap.capture(team.triage) as failed:
            team.run()
            1 / 0
    assert host.events - before == per_run and failed.span_count == per_run
    assert processors() == host_list
    team.run()
    assert failed.span_count == per_run and stored(failed) == per_run

    # the failure can also come from the run itself
    broken = Team(fail_on_second_call=True)
    before = host.events
    with pytest.raises(RuntimeError, match="the model call failed"):
        with ap.capture(broken.triage) as crashed:
            broken.run()
    assert processors() == host_list
    assert crashed.span_count == host.events - before > 0
    errors = [s for s in span_list(crashed.otlp) if s["status"]["code"] == 2]
    assert errors  # the failed model call is recorded as an error span

    # exactly once, everywhere, and nothing global changed
    assert len(set(host.span_ids)) == len(host.span_ids)
    assert host.spans_started == host.spans_ended and host.traces_started == host.traces_ended
    assert host.shutdowns == 0
    assert otel_trace.get_tracer_provider() is other
    assert_no_global_provider()
    assert team.executed == ["Paris"] * 6


def test_a_processor_registered_during_the_capture_survives_it(sdk):
    team = Team()
    late = HostCounter()
    with ap.capture(team.triage):
        get_trace_provider().register_processor(late)
        team.run()
    assert processors() == (sdk, late)
    assert late.events > 0


def test_a_host_that_already_instruments_with_openinference_keeps_its_instrumentation(sdk):
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor, _tool_schemas
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace import TracerProvider

    def patched():
        return sorted(m.__name__ for m, a in _tool_schemas.find_tool_execution_bindings()
                      if type(getattr(m, a)).__name__ == "FunctionWrapper")

    host_spans = InMemorySpanExporter()
    host_provider = TracerProvider(shutdown_on_exit=False)
    host_provider.add_span_processor(SimpleSpanProcessor(host_spans))
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=host_provider, exclusive_processor=False)
    try:
        team = Team()
        patches = patched()
        assert patches  # the host's instrumentor patched the SDK's tool execution step
        listed = processors()
        with ap.capture(team.triage) as cap:
            team.run()
        assert cap.span_count == sdk.events == 12  # the capture still gets its own spans
        assert len(host_spans.get_finished_spans()) == 12  # and the host's instrumentation its own, once
        assert patched() == patches and processors() == listed  # nothing of the host's was undone
    finally:
        instrumentor.uninstrument()


def test_two_agents_share_one_processor(sdk):
    team = Team()
    before = processors()
    with ap.capture([team.triage, team.billing]) as cap:
        assert len(processors()) == len(before) + 1
        team.run()
    assert cap.span_count == sdk.events  # each span once, not once per agent
    assert processors() == before


def test_a_closed_capture_is_inert_even_while_its_processor_is_still_listed(sdk):
    team = Team()
    with ap.capture(team.triage) as cap:
        team.run()
    spans = stored(cap)
    # re-register the capture's processor by hand: it must not record, because the capture is closed
    leftover = cap._session.shared["openai-agents"]["processor"]
    get_trace_provider().register_processor(leftover)
    team.run()
    assert stored(cap) == spans


def test_tracing_switched_off_is_reported_and_leaves_state_alone(sdk):
    team = Team()
    provider = get_trace_provider()
    provider.set_disabled(True)
    before = processors()
    with pytest.raises(CaptureError, match="tracing is switched off"):
        with ap.capture(team.triage):
            raise AssertionError("must not enter")
    assert processors() == before


def test_misuse_is_reported(sdk):
    team = Team()
    cap = ap.capture(team.triage)
    with pytest.raises(CaptureError, match="not been entered"):
        cap.otlp
    with cap:
        pass
    assert cap.span_count == 0 and processors() == (sdk,)
    with pytest.raises(CaptureError):
        cap.run


# -- which runs a capture takes --------------------------------------------------------

def solo(name, text):
    """One agent with a one-reply model: a run of it is a trace of five SDK events."""
    return Agent(name=name, instructions=f"{name} instructions.", model=FakeModel([[message(text)]]))


def run_solo(agent):
    agent.model.calls = 0
    return Runner.run_sync(agent, "hello")


def agent_names(doc):
    return {attrs(s)["agent.name"] for s in span_list(doc) if "agent.name" in attrs(s)}


def root_starts_first(doc):
    spans = span_list(doc)
    (root,) = [s for s in spans if "parentSpanId" not in s]
    assert int(root["startTimeUnixNano"]) <= min(int(s["startTimeUnixNano"]) for s in spans)


@pytest.mark.parametrize("order", [("selected", "unrelated"), ("unrelated", "selected")])
def test_an_unrelated_agent_run_in_the_window_is_not_captured(sdk, order):
    agents = {"selected": solo("selected", "selected done"), "unrelated": solo("unrelated", "unrelated done")}
    run_solo(agents["selected"])
    per_run = sdk.events
    assert per_run == 5

    before = sdk.events
    with ap.capture(agent=agents["selected"]) as cap:
        for name in order:
            run_solo(agents[name])
    assert sdk.events - before == 2 * per_run  # the host observed both runs

    assert cap.span_count == per_run  # the capture holds one run's events, none of the unrelated run's
    assert agent_names(cap.otlp) == {"selected"}
    assert "unrelated" not in json.dumps(cap.otlp).lower()  # not as an agent, an instruction or a reply
    assert len({s["traceId"] for s in span_list(cap.otlp)}) == 1
    root_starts_first(cap.otlp)

    assert len(cap.runs) == 1
    run = cap.run  # exactly one run, so the single-run accessor works
    assert ap.validate(run) == []
    assert [a["name"] for a in run.agents] == ["selected"]  # no agent joined the roster from the configuration
    assert [a["name"] for a in cap.config.agents] == ["selected"]
    assert run.agents[0]["instructions"] == "selected instructions."
    assert "config_agents_not_attached" not in run.dataset["capture"]
    assert processors() == (sdk,)
    assert_no_global_provider()


def test_agents_selected_together_each_get_their_run(sdk):
    one, two, other = solo("one", "one done"), solo("two", "two done"), solo("other", "other done")
    with ap.capture([one, two]) as cap:
        for agent in (one, other, two):
            run_solo(agent)
    assert cap.span_count == 10 and agent_names(cap.otlp) == {"one", "two"}
    assert sorted(r.agents[0]["name"] for r in cap.runs) == ["one", "two"]
    assert all(len(r.agents) == 1 for r in cap.runs)


def test_a_run_that_hands_off_to_a_selected_agent_is_captured_whole(sdk):
    team = Team()
    with ap.capture(team.billing) as cap:  # triage is not selected, it only hands off to billing
        team.run()
    assert cap.span_count == sdk.events == 12
    run = cap.run
    assert {a["name"] for a in run.agents} == {"triage", "billing"}  # the roster is what the spans name
    assert [a["name"] for a in cap.config.agents] == ["billing"]
    (billing,) = [a for a in run.agents if a["name"] == "billing"]
    assert billing["instructions"] == "Handle billing."
    assert "config_agents_not_attached" not in run.dataset["capture"]


def test_a_selected_agent_that_hands_off_takes_the_whole_run_with_the_target_in_the_roster(sdk):
    team = Team()
    with ap.capture(team.triage) as cap:
        team.run()
    run = cap.run
    assert cap.span_count == sdk.events == 12
    assert {a["name"] for a in run.agents} == {"triage", "billing"}
    root_starts_first(cap.otlp)


def test_a_run_that_calls_a_selected_agent_as_a_tool_is_captured_whole(sdk, caplog):
    inner = solo("inner", "inner done")
    orchestrator = Agent(name="orchestrator", instructions="Orchestrate.", model=FakeModel(
        [[call("ask_inner", {"input": "hi"}, "call_i1")], [message("orchestrator done")]]),
        tools=[inner.as_tool(tool_name="ask_inner", tool_description="Ask the inner agent.")])
    caplog.set_level("ERROR")
    with ap.capture(inner) as cap:  # the orchestrator is not selected
        Runner.run_sync(orchestrator, "go")
    assert cap.span_count == sdk.events == 12
    assert {s["traceId"] for s in span_list(cap.otlp)} == {span_list(cap.otlp)[0]["traceId"]}
    assert agent_names(cap.otlp) == {"orchestrator", "inner"}
    root_starts_first(cap.otlp)
    run = cap.run
    (nested,) = run.record["subagent_trajectories"]
    (agent,) = nested["extra"]["agentpile"]["agents"]
    assert (agent["name"], agent["instructions"]) == ("inner", "inner instructions.")  # its configuration reached its own roster
    assert [a["name"] for a in run.agents] == ["orchestrator"]
    assert "orchestrator" not in {a["name"] for a in cap.config.agents}
    # spans opened before the selected agent showed up were replayed in another task: nothing is left to detach there
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == []


def test_an_unselected_trace_is_dropped_when_it_ends_and_holds_nothing(sdk):
    selected, unrelated = solo("selected", "s"), solo("unrelated", "u")
    with ap.capture(selected) as cap:
        run_solo(unrelated)
        processor = cap._session.shared["openai-agents"]["processor"]
        assert not [s for s in processor._traces.values() if hasattr(s, "events")]  # no held event outlives its trace
        assert cap._session.started == 0  # the capture's tracer was never asked for a span
        run_solo(selected)
    assert cap.span_count == 5
    assert processor._traces == {}  # and the closed capture lets go of its bookkeeping


def test_a_trace_that_began_before_the_capture_is_taken_from_its_first_event_inside_it(sdk):
    from agents import trace

    selected = solo("selected", "selected done")
    outer = trace("outer workflow")
    outer.start(mark_as_current=True)
    try:
        with ap.capture(selected) as cap:
            run_solo(selected)
    finally:
        outer.finish(reset_current=True)
    assert agent_names(cap.otlp) == {"selected"}
    assert "outer workflow" not in json.dumps(cap.otlp)  # the start of that trace was announced before the capture
    assert len(cap.runs) == 1


def test_runs_from_several_threads_are_sorted_by_agent(sdk):
    import asyncio
    import threading

    chosen = [solo("chosen1", "c1"), solo("chosen2", "c2")]
    others = [solo("other1", "o1"), solo("other2", "o2")]
    barrier = threading.Barrier(4)
    rounds = 5

    def work(agent):
        barrier.wait()
        for _ in range(rounds):
            agent.model.calls = 0
            asyncio.run(Runner.run(agent, "hello"))

    threads = [threading.Thread(target=work, args=(agent,)) for agent in chosen + others]
    with ap.capture(chosen) as cap:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert sdk.events == 4 * rounds * 5
    assert cap.span_count == 2 * rounds * 5
    assert agent_names(cap.otlp) == {"chosen1", "chosen2"}
    runs = cap.runs
    assert len(runs) == 2 * rounds and all(len(r.agents) == 1 for r in runs)
    assert sorted(r.agents[0]["name"] for r in runs) == ["chosen1"] * rounds + ["chosen2"] * rounds


class _Inner:
    """Stands in for the OpenInference processor and records what it is handed."""

    def __init__(self, fail_on=None):
        self.seen = []
        self.fail_on = fail_on

    def _note(self, kind, obj):
        self.seen.append((kind, getattr(obj, "span_id", None) or obj.trace_id))
        if self.fail_on == len(self.seen):
            raise RuntimeError("inner failed")

    def on_trace_start(self, trace):
        self._note("trace_start", trace)

    def on_trace_end(self, trace):
        self._note("trace_end", trace)

    def on_span_start(self, span):
        self._note("span_start", span)

    def on_span_end(self, span):
        self._note("span_end", span)


def _sdk_span(trace_id, span_id, kind="function", name="x"):
    from types import SimpleNamespace

    return SimpleNamespace(trace_id=trace_id, span_id=span_id, span_data=SimpleNamespace(type=kind, name=name))


def _processor(inner, names):
    from types import SimpleNamespace

    from agentpile.integrations.openai_agents import _Processor, _RootClock

    clock = _RootClock(SimpleNamespace(start_span=lambda *a, **k: None))
    return _Processor(inner, SimpleNamespace(closed=False), set(names), clock)


def test_routing_holds_replays_in_order_then_passes_through_and_drops():
    from types import SimpleNamespace

    inner = _Inner()
    p = _processor(inner, {"chosen"})
    t1, t2 = SimpleNamespace(trace_id="t1"), SimpleNamespace(trace_id="t2")

    p.on_trace_start(t1)
    p.on_span_start(_sdk_span("t1", "a", "task"))
    p.on_trace_start(t2)
    p.on_span_start(_sdk_span("t2", "b", "agent", "unrelated"))
    assert inner.seen == []  # nothing yet: no selected agent in either trace
    p.on_span_start(_sdk_span("t1", "c", "agent", "chosen"))
    assert inner.seen == [("trace_start", "t1"), ("span_start", "a"), ("span_start", "c")]  # replayed in order
    p.on_span_end(_sdk_span("t1", "c", "agent", "chosen"))
    p.on_trace_end(t1)
    assert inner.seen[-2:] == [("span_end", "c"), ("trace_end", "t1")]  # now straight through
    p.on_span_end(_sdk_span("t1", "late"))
    assert inner.seen[-1] == ("span_end", "late")  # a late event of a taken trace still goes through

    p.on_trace_end(t2)
    p.on_span_end(_sdk_span("t2", "after"))
    assert all(ident not in ("b", "t2", "after") for _, ident in inner.seen)  # the unselected trace never reached inner


def test_routing_one_failing_replayed_event_does_not_lose_the_others():
    from types import SimpleNamespace

    inner = _Inner(fail_on=2)
    p = _processor(inner, {"chosen"})
    p.on_trace_start(SimpleNamespace(trace_id="t"))
    p.on_span_start(_sdk_span("t", "a", "task"))
    with pytest.raises(RuntimeError, match="inner failed"):
        p.on_span_start(_sdk_span("t", "c", "agent", "chosen"))
    assert [kind for kind, _ in inner.seen] == ["trace_start", "span_start", "span_start"]  # the third was still delivered


def test_routing_ignores_everything_once_closed():
    from types import SimpleNamespace

    inner = _Inner()
    p = _processor(inner, {"chosen"})
    p.on_trace_start(SimpleNamespace(trace_id="t"))
    p.close()
    p.on_span_start(_sdk_span("t", "c", "agent", "chosen"))
    p.on_span_start(_sdk_span("fresh", "d", "agent", "chosen"))
    assert inner.seen == [] and p._traces == {}


# -- cap.run (needs the otlp-json reader, which is written separately) ---------------------

def test_run_is_built_from_the_document_with_the_config_attached(sdk):
    if "otlp-json" not in ap.readers:
        pytest.skip("the otlp-json reader is not registered in this build yet")
    team = Team()
    with ap.capture(team.triage) as cap:
        team.run()
    run = cap.run
    assert run.kind == "trajectory"
    assert ap.validate(run) == []
    by_name = {t["name"]: t for t in run.tools.values()}
    assert by_name["get_weather"]["definition"]["function"]["parameters"]["required"] == ["city"]
    assert {a["name"] for a in run.agents} >= {"triage", "billing"}
    ids_called = [e.call_id for e in run.events() if e.kind == "call"]
    assert "call_w1" in ids_called
