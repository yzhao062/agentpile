"""Gate 5 for Pydantic AI: a fake model and real tools, a host that already observes its agent, and a lifecycle check.

No network and no key: FunctionModel plays the model. The tests skip when Pydantic AI or the OpenTelemetry SDK is absent.
"""

import asyncio
import json
import os

import pytest

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

pytest.importorskip("pydantic_ai")
pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry import metrics as otel_metrics  # noqa: E402
from opentelemetry import trace as otel_trace  # noqa: E402
from opentelemetry.metrics import NoOpMeterProvider  # noqa: E402
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider  # noqa: E402
from pydantic_ai import Agent, InstrumentationSettings  # noqa: E402
from pydantic_ai.capabilities import Instrumentation  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402

import agentpile as ap  # noqa: E402
from agentpile._capture import CaptureError  # noqa: E402


# -- fixtures --------------------------------------------------------------------

class Counting(SpanProcessor):
    """The host's observer: counts and remembers every span it is handed, and notes a shutdown."""

    def __init__(self):
        self.started = 0
        self.ended = 0
        self.ended_ids = []
        self.shutdowns = 0

    def on_start(self, span, parent_context=None):
        self.started += 1

    def on_end(self, span):
        self.ended += 1
        self.ended_ids.append(format(span.context.span_id, "016x"))

    def shutdown(self):
        self.shutdowns += 1

    def force_flush(self, timeout_millis=30000):
        return True


def host_settings(counter):
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(counter)
    return InstrumentationSettings(tracer_provider=provider, meter_provider=NoOpMeterProvider(), version=5)


class World:
    """An agent with a real tool, a scripted model, and (optionally) a host that observes it."""

    def __init__(self, style, name="weather_agent", fail=False):
        self.style = style
        self.executed = []
        self.counter = Counting() if style != "none" else None
        self.settings = host_settings(self.counter) if self.counter else None

        def model_fn(messages, info):
            if len(messages) == 1:
                tool = "explode" if fail else "get_weather"
                return ModelResponse(parts=[ToolCallPart(tool, {"city": "Paris"}, tool_call_id="call_w1")])
            return ModelResponse(parts=[TextPart("It is sunny in Paris.")])

        kwargs = dict(name=name, instructions="Answer weather questions.")
        if style == "capability":
            kwargs["capabilities"] = [Instrumentation(settings=self.settings)]
        self.agent = Agent(FunctionModel(model_fn), **kwargs)
        if style == "attribute":
            self.agent.instrument = self.settings
        if style == "instrument_all":
            Agent.instrument_all(self.settings)

        @self.agent.tool_plain
        def get_weather(city: str) -> str:
            """Return the weather for a city."""
            self.executed.append(city)
            return f"sunny in {city}"

        @self.agent.tool_plain
        def explode(city: str) -> str:
            """A tool that fails."""
            raise RuntimeError("the tool failed")

    def close(self):
        if self.style == "instrument_all":
            Agent.instrument_all(False)

    def setting(self):
        """What the agent would run with: the capability's settings, else the agent attribute."""
        if self.style == "capability":
            leaves = []
            self.agent.root_capability.apply(leaves.append)
            return next(c for c in leaves if isinstance(c, Instrumentation)).settings
        return self.agent.instrument

    def run(self):
        return self.agent.run_sync("What is the weather in Paris?")


@pytest.fixture(params=["attribute", "capability", "instrument_all"])
def hosted(request):
    world = World(request.param)
    yield world
    world.close()


def span_list(doc):
    return [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]


def unwrap(value):
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "boolValue" in value:
        return value["boolValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    raise AssertionError(f"unexpected value {value}")


def attrs(span):
    return {a["key"]: unwrap(a["value"]) for a in span["attributes"]}


def ids(doc):
    return {s["spanId"] for s in span_list(doc)}


def stored(cap):
    """Spans the capture's own store holds right now; unlike cap.otlp this is not frozen when the block ends."""
    return len(cap._session.exporter.get_finished_spans())


GLOBAL_PROVIDERS = (otel_trace.get_tracer_provider(), otel_metrics.get_meter_provider())  # as found at import


def assert_no_global_provider():
    """Capture never installs a provider of its own as the global tracer or meter provider."""
    assert (otel_trace.get_tracer_provider(), otel_metrics.get_meter_provider()) == GLOBAL_PROVIDERS


# -- what the document holds -------------------------------------------------------

def test_document_holds_calls_ids_agent_names_and_config():
    world = World("attribute")
    with ap.capture(agent=world.agent) as cap:
        result = world.run()
    assert result.output == "It is sunny in Paris."
    assert world.executed == ["Paris"]  # the tool really ran

    spans = span_list(cap.otlp)
    by_name = {s["name"]: s for s in spans}
    assert sorted(s["name"] for s in spans).count("chat function:model_fn:") == 2
    assert {"invoke_agent weather_agent", "execute_tool get_weather"} <= set(by_name)

    tool = attrs(by_name["execute_tool get_weather"])
    assert tool["gen_ai.tool.name"] == "get_weather"
    assert tool["gen_ai.tool.call.id"] == "call_w1"
    assert json.loads(tool["gen_ai.tool.call.arguments"]) == {"city": "Paris"}
    assert tool["gen_ai.tool.call.result"] == "sunny in Paris"
    assert attrs(by_name["invoke_agent weather_agent"])["gen_ai.agent.name"] == "weather_agent"

    chats = [s for s in spans if s["name"].startswith("chat ")]
    first = attrs(chats[0])
    definitions = json.loads(first["gen_ai.tool.definitions"])  # the agent's tool configuration, as the model saw it
    assert definitions[0]["name"] == "get_weather"
    assert definitions[0]["parameters"]["properties"] == {"city": {"type": "string"}}
    assert json.loads(first["gen_ai.system_instructions"])[0]["content"] == "Answer weather questions."
    call = json.loads(first["gen_ai.output.messages"])[0]["parts"][0]
    assert (call["type"], call["id"], call["name"]) == ("tool_call", "call_w1", "get_weather")

    # the encoding: hex ids, string nanosecond times, one trace, parents inside the document
    assert {s["traceId"] for s in spans} == {spans[0]["traceId"]} and len(spans[0]["traceId"]) == 32
    for s in spans:
        assert len(s["spanId"]) == 16 and int(s["spanId"], 16) >= 0
        assert isinstance(s["startTimeUnixNano"], str) and int(s["endTimeUnixNano"]) >= int(s["startTimeUnixNano"])
    roots = [s for s in spans if "parentSpanId" not in s]
    assert [s["name"] for s in roots] == ["invoke_agent weather_agent"]
    assert all(s["parentSpanId"] in ids(cap.otlp) for s in spans if "parentSpanId" in s)
    assert json.loads(json.dumps(cap.otlp, allow_nan=False)) == cap.otlp
    assert cap.span_count == 4
    assert_no_global_provider()

    cfg = cap.config
    assert ap.validate(cfg) == []
    assert cfg.agents[0]["name"] == "weather_agent" and "weather_agent.get_weather" in cfg.tools


def test_capture_records_the_failure_of_a_tool_and_still_restores():
    world = World("attribute", fail=True)
    original = world.setting()
    with pytest.raises(Exception, match="the tool failed"):
        with ap.capture(world.agent) as cap:
            world.run()
    assert world.setting() is original
    spans = {s["name"]: s for s in span_list(cap.otlp)}
    assert spans["execute_tool explode"]["status"]["code"] == 2  # ERROR
    assert any(e["name"] == "exception" for e in spans["execute_tool explode"]["events"])
    assert world.counter.ended == cap.span_count  # the host saw the same spans


# -- lifecycle ------------------------------------------------------------------------

def test_lifecycle_host_keeps_observing_and_state_is_restored(hosted):
    host, agent = hosted.counter, hosted.agent
    providers = (otel_trace.get_tracer_provider(), otel_metrics.get_meter_provider())
    original = hosted.setting()

    # (1) the host's own observer, before any capture
    before = host.ended
    hosted.run()
    per_run = host.ended - before
    assert per_run == 4 and host.started == host.ended

    # (2) two captures back to back
    docs = []
    for _ in range(2):
        before = host.ended
        with ap.capture(agent) as cap:
            assert hosted.setting() is not original  # the capture's setting is in place for the run
            hosted.run()
        assert host.ended - before == per_run  # the host kept receiving, once per span
        assert cap.span_count == per_run  # the capture got its own run, no more
        assert len(ids(cap.otlp)) == per_run
        assert hosted.setting() is original  # restored
        docs.append(cap)
    first, second = docs
    assert ids(first.otlp).isdisjoint(ids(second.otlp))
    assert {s["traceId"] for s in span_list(first.otlp)}.isdisjoint({s["traceId"] for s in span_list(second.otlp)})
    snapshot = [json.dumps(c.otlp, sort_keys=True) for c in docs]

    # (3) one run outside any capture: the host still observes, the finished captures see nothing
    before = host.ended
    hosted.run()
    assert host.ended - before == per_run
    assert [json.dumps(c.otlp, sort_keys=True) for c in docs] == snapshot
    assert [stored(c) for c in docs] == [per_run, per_run]

    # (4) a capture that exits by exception
    before = host.ended
    with pytest.raises(ZeroDivisionError):
        with ap.capture(agent) as failed:
            hosted.run()
            1 / 0
    assert host.ended - before == per_run
    assert failed.span_count == per_run
    assert hosted.setting() is original
    before = host.ended
    hosted.run()
    assert host.ended - before == per_run
    assert stored(failed) == per_run

    # exactly once, everywhere; the host never lost its provider and nothing global changed
    assert len(set(host.ended_ids)) == len(host.ended_ids)
    for cap in docs + [failed]:
        assert ids(cap.otlp) <= set(host.ended_ids)
    assert host.shutdowns == 0
    assert (otel_trace.get_tracer_provider(), otel_metrics.get_meter_provider()) == providers
    assert_no_global_provider()
    assert hosted.executed == ["Paris"] * 6


def test_lifecycle_without_a_host():
    world = World("none")
    assert world.agent.instrument is None
    with ap.capture(world.agent) as cap:
        world.run()
    assert world.agent.instrument is None
    assert cap.span_count == 4
    world.run()  # uninstrumented again: nothing more reaches the finished capture
    assert cap.span_count == 4 and stored(cap) == 4
    assert Agent._instrument_default is False


def test_the_hosts_own_settings_object_is_left_as_it_was(hosted):
    settings = hosted.settings
    flags = (settings.include_content, settings.tracer, settings.version)
    settings.include_content = False
    with pytest.warns(RuntimeWarning, match="content"):  # the host's exporters will receive content it switched off
        with ap.capture(hosted.agent) as cap:
            hosted.run()
    assert settings.include_content is False and settings.tracer is flags[1]
    # content is recorded for the capture even though the host has it off
    assert any("gen_ai.tool.call.result" in attrs(s) for s in span_list(cap.otlp))


def test_two_agents_in_one_capture_and_a_failing_entry_rolls_back():
    a, b = World("attribute", name="agent_a"), World("none", name="agent_b")
    orig_a = a.setting()
    with ap.capture([a.agent, b.agent]) as cap:
        a.run()
        b.run()
    assert cap.span_count == 8
    assert a.setting() is orig_a and b.agent.instrument is None
    names = {attrs(s).get("gen_ai.agent.name") for s in span_list(cap.otlp)}
    assert {"agent_a", "agent_b"} <= names

    class Broken(Agent):
        @property
        def root_capability(self):
            raise RuntimeError("cannot bind")

    broken = Broken(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("x")])), name="broken")
    with pytest.raises(RuntimeError, match="cannot bind"):
        with ap.capture([a.agent, broken]):
            raise AssertionError("must not enter")
    assert a.setting() is orig_a  # the agent bound before the failure was put back


def test_a_finished_capture_ignores_a_straggling_tracer_but_the_host_still_gets_the_span():
    world = World("attribute")
    with ap.capture(world.agent) as cap:
        ours = world.setting()
        world.run()
    assert stored(cap) == 4
    before = world.counter.ended
    world.agent.instrument = ours  # a run that still holds the capture's tracer, such as a stream left open
    world.run()
    assert stored(cap) == 4  # the closed capture collects nothing more
    assert world.counter.ended - before == 4  # and the forwarding to the host goes on


def test_async_capture():
    world = World("attribute")
    original = world.setting()

    async def main():
        async with ap.capture(world.agent) as cap:
            await world.agent.run("What is the weather in Paris?")
        return cap

    cap = asyncio.run(main())
    assert cap.span_count == 4 and world.setting() is original


def test_unreadable_host_tracer_is_reported_not_hidden():
    world = World("attribute")
    world.settings.tracer = object()  # not an OpenTelemetry SDK tracer
    with pytest.warns(RuntimeWarning, match="will not receive"):
        with ap.capture(world.agent) as cap:
            world.run()
    assert cap.span_count == 4
    assert world.counter.ended == 0


def test_a_setting_the_host_replaced_during_the_capture_is_kept():
    world = World("attribute")
    replacement = host_settings(Counting())
    with ap.capture(world.agent):
        world.agent.instrument = replacement
    assert world.agent.instrument is replacement


def test_a_missing_opentelemetry_sdk_is_reported_with_the_extra_and_touches_nothing(monkeypatch):
    import sys

    import agentpile.integrations as integrations

    monkeypatch.delitem(sys.modules, "agentpile.integrations._otel", raising=False)
    monkeypatch.delattr(integrations, "_otel", raising=False)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", None)  # makes `import opentelemetry.sdk.resources` fail
    world = World("attribute")
    original = world.setting()
    with pytest.raises(CaptureError, match=r"pip install 'agentpile\[pydantic-ai\]'$"):  # the SDK comes with it
        with ap.capture(world.agent):
            raise AssertionError("must not enter")
    assert world.setting() is original


def test_misuse_is_reported():
    world = World("none")
    cap = ap.capture(world.agent)
    with pytest.raises(CaptureError, match="not been entered"):
        cap.otlp
    with cap:
        pass
    with pytest.raises(CaptureError, match="entered once"):
        with cap:
            pass
    assert cap.span_count == 0
    with pytest.raises(CaptureError):  # nothing ran, so there is no run to build
        cap.run


# -- which runs a capture takes ---------------------------------------------------------

def agent_names(doc):
    return {attrs(s)["gen_ai.agent.name"] for s in span_list(doc) if "gen_ai.agent.name" in attrs(s)}


@pytest.mark.parametrize("style", ["none", "attribute", "capability", "instrument_all"])
def test_an_unrelated_agent_run_in_the_window_is_not_captured(style):
    selected = World(style, name="selected_agent")
    unrelated = World(style, name="unrelated_agent")
    try:
        with ap.capture(selected.agent) as cap:
            selected.run()
            unrelated.run()
    finally:
        selected.close()
        unrelated.close()
    assert unrelated.executed == ["Paris"] and selected.executed == ["Paris"]  # both really ran
    assert cap.span_count == 4  # one agent run, four spans
    assert agent_names(cap.otlp) == {"selected_agent"}
    assert "unrelated_agent" not in json.dumps(cap.otlp)
    assert len(cap.runs) == 1
    assert [a["name"] for a in cap.run.agents] == ["selected_agent"]  # the configuration added no agent to the roster
    assert [a["name"] for a in cap.config.agents] == ["selected_agent"]
    if style in ("attribute", "capability"):
        assert unrelated.counter.ended == 4  # the unrelated agent's own host kept seeing its spans


def test_a_shared_instrumentation_capability_does_not_pull_in_the_agents_that_share_it():
    host = Counting()
    settings = host_settings(host)
    shared = Instrumentation(settings=settings)
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    selected = Agent(model, name="selected_agent", capabilities=[shared])
    unrelated = Agent(model, name="unrelated_agent", capabilities=[shared])

    with ap.capture(selected) as cap:
        assert shared.settings is not settings and "for_run" in shared.__dict__  # taken over for the block
        selected.run_sync("hi")
        unrelated.run_sync("hi")
    assert cap.span_count == 2 and agent_names(cap.otlp) == {"selected_agent"}
    assert [a["name"] for a in cap.run.agents] == ["selected_agent"]
    assert host.ended == 4  # the host saw both runs, once per span
    assert len(set(host.ended_ids)) == 4 and ids(cap.otlp) <= set(host.ended_ids)
    assert shared.settings is settings and "for_run" not in shared.__dict__  # and the capability is as it was found
    unrelated.run_sync("hi")
    assert host.ended == 6 and cap.span_count == 2


def test_agents_that_share_one_capability_and_two_overlapping_captures_restore_it_in_any_order():
    host = Counting()
    settings = host_settings(host)
    shared = Instrumentation(settings=settings)
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    one = Agent(model, name="one", capabilities=[shared])
    two = Agent(model, name="two", capabilities=[shared])
    three = Agent(model, name="three", capabilities=[shared])

    with ap.capture([one, two]) as both:  # two selected agents on one capability, one bystander
        for agent in (one, two, three):
            agent.run_sync("hi")
    assert both.span_count == 4 and agent_names(both.otlp) == {"one", "two"}
    assert len(set(host.ended_ids)) == len(host.ended_ids) == 6  # no span reached the host twice
    assert shared.settings is settings and "for_run" not in shared.__dict__

    first, second = ap.capture(one), ap.capture(two)
    first.__enter__()
    second.__enter__()
    one.run_sync("hi")
    two.run_sync("hi")
    three.run_sync("hi")
    first.__exit__(None, None, None)  # the older capture leaves first
    two.run_sync("hi")
    second.__exit__(None, None, None)
    assert agent_names(first.otlp) == {"one"} and agent_names(second.otlp) == {"two"}
    assert first.span_count == 2 and second.span_count == 4
    assert shared.settings is settings and "for_run" not in shared.__dict__


def test_a_wrapper_agent_on_a_shared_capability_is_captured_and_its_neighbour_is_not():
    from pydantic_ai.agent import WrapperAgent

    shared = Instrumentation(settings=host_settings(Counting()))
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    inner = Agent(model, name="inner_agent", capabilities=[shared])
    neighbour = Agent(model, name="neighbour_agent", capabilities=[shared])
    wrapper = WrapperAgent(inner)  # its runs reach the capability as the wrapped agent
    with ap.capture(wrapper) as cap:
        wrapper.run_sync("hi")
        neighbour.run_sync("hi")
    assert cap.span_count == 2 and agent_names(cap.otlp) == {"inner_agent"}


def test_a_selected_agent_calling_an_unselected_one_records_the_caller_only():
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("sub says hi")]))
    sub = Agent(model, name="sub_agent", instructions="Help the caller.")

    def caller_model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("ask_sub", {}, tool_call_id="call_s1")])
        return ModelResponse(parts=[TextPart("done")])

    caller = Agent(FunctionModel(caller_model), name="caller_agent")

    @caller.tool_plain
    async def ask_sub() -> str:
        return (await sub.run("hi")).output

    with ap.capture(caller) as only_caller:
        caller.run_sync("go")
    assert only_caller.span_count == 4 and agent_names(only_caller.otlp) == {"caller_agent"}
    assert "subagent_trajectories" not in only_caller.run.record

    with ap.capture([caller, sub]) as both:
        caller.run_sync("go")
    assert agent_names(both.otlp) == {"caller_agent", "sub_agent"}
    (nested,) = both.run.record["subagent_trajectories"]
    (entry,) = nested["extra"]["agentpile"]["agents"]
    assert (entry["name"], entry["instructions"]) == ("sub_agent", "Help the caller.")
    assert [a["name"] for a in both.run.agents] == ["caller_agent"]


# -- cap.run (needs the otlp-json reader, which is written separately) ---------------------

def test_run_is_built_from_the_document_with_the_config_attached():
    if "otlp-json" not in ap.readers:
        pytest.skip("the otlp-json reader is not registered in this build yet")
    world = World("attribute")
    with ap.capture(world.agent) as cap:
        world.run()
    run = cap.run
    assert run.kind == "trajectory"
    assert ap.validate(run) == []
    by_name = {t["name"]: t for t in run.tools.values()}
    assert by_name["get_weather"]["definition"]["function"]["parameters"]["properties"] == {"city": {"type": "string"}}
    (agent,) = [a for a in run.agents if a["name"] == "weather_agent"]
    assert agent["instructions"] == "Answer weather questions."
    calls = [e for e in run.events() if e.kind == "call"]
    assert [(e.name, e.call_id) for e in calls] == [("get_weather", "call_w1")]


def test_an_untested_framework_version_is_built_only_on_request(monkeypatch):
    if "otlp-json" not in ap.readers:
        pytest.skip("the otlp-json reader is not registered in this build yet")
    from agentpile.formats import otlp

    monkeypatch.setitem(otlp._TESTED, "pydantic-ai/genai", ("0.0.0",))  # the installed version is now untested
    world = World("attribute")
    with ap.capture(world.agent) as cap:
        world.run()
    with pytest.raises(ap.UnsupportedVersion, match="allow_untested_version"):
        cap.run
    world = World("attribute")
    with ap.capture(world.agent, allow_untested_version=True) as cap:
        world.run()
    run = cap.run
    assert run.source["dialect_tested"] is False and any("not tested" in f for f in run.source["lost_fields"])
    assert [(e.name, e.call_id) for e in run.events() if e.kind == "call"] == [("get_weather", "call_w1")]
    (agent,) = [a for a in run.agents if a["name"] == "weather_agent"]
    assert agent["instructions"] == "Answer weather questions."  # the configuration is still attached
