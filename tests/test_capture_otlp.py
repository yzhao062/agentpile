"""The standard-library parts of capture: the OTLP JSON encoder, config attachment and run building.

These tests use stand-in spans and need neither OpenTelemetry nor an agent framework.
"""

import contextlib
import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

import agentpile as ap
from agentpile import integrations
from agentpile._capture import CaptureError, READER, attach_config, build_runs, count_spans, encode_otlp

_ABSENT = object()


def span(name="s", trace=0xA1, sid=1, parent=None, start=10, end=20, kind="INTERNAL", attrs=None, events=(),
         links=(), status=("UNSET", None), resource=None, scope=("scope-a", "1.0"), dropped=0):
    return SimpleNamespace(
        name=name,
        context=SimpleNamespace(trace_id=trace, span_id=sid, trace_state=None),
        get_span_context=lambda: SimpleNamespace(trace_id=trace, span_id=sid, trace_state=None),
        parent=None if parent is None else SimpleNamespace(trace_id=trace, span_id=parent),
        start_time=start, end_time=end, kind=SimpleNamespace(name=kind),
        attributes=dict(attrs or {}), events=list(events), links=list(links),
        status=SimpleNamespace(status_code=SimpleNamespace(name=status[0]), description=status[1]),
        resource=resource or SimpleNamespace(attributes={"service.name": "svc"}, schema_url=""),
        instrumentation_scope=SimpleNamespace(name=scope[0], version=scope[1], schema_url="", attributes=None),
        dropped_attributes=dropped, dropped_events=0, dropped_links=0)


def values(doc):
    return [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]


# -- encoder ---------------------------------------------------------------------

def test_document_shape_and_ids():
    doc = encode_otlp([span(trace=0xABC, sid=0x1F, parent=0x2)])
    assert list(doc) == ["resourceSpans"]
    (rs,) = doc["resourceSpans"]
    assert rs["resource"]["attributes"] == [{"key": "service.name", "value": {"stringValue": "svc"}}]
    (ss,) = rs["scopeSpans"]
    assert ss["scope"] == {"name": "scope-a", "version": "1.0"}
    (s,) = ss["spans"]
    assert s["traceId"] == "0" * 29 + "abc" and len(s["traceId"]) == 32
    assert s["spanId"] == "0" * 14 + "1f" and s["parentSpanId"] == "0" * 15 + "2"
    assert s["startTimeUnixNano"] == "10" and s["endTimeUnixNano"] == "20"
    assert s["kind"] == 1 and s["status"] == {"code": 0}


def test_root_span_has_no_parent_field_and_kind_and_status_map():
    root = encode_otlp([span(kind="CLIENT", status=("ERROR", "boom"))])
    (s,) = values(root)
    assert "parentSpanId" not in s
    assert s["kind"] == 3
    assert s["status"] == {"code": 2, "message": "boom"}
    (ok,) = values(encode_otlp([span(status=("OK", None))]))
    assert ok["status"] == {"code": 1}


def test_attribute_value_encoding():
    attrs = {"s": "x", "b": True, "i": 7, "neg": -3, "f": 1.5, "arr": ["a", "b"], "nums": (1, 2), "raw": b"\x00\xff"}
    (s,) = values(encode_otlp([span(attrs=attrs)]))
    got = {a["key"]: a["value"] for a in s["attributes"]}
    assert got["s"] == {"stringValue": "x"}
    assert got["b"] == {"boolValue": True}  # a bool is not an int
    assert got["i"] == {"intValue": "7"} and got["neg"] == {"intValue": "-3"}
    assert got["f"] == {"doubleValue": 1.5}
    assert got["arr"] == {"arrayValue": {"values": [{"stringValue": "a"}, {"stringValue": "b"}]}}
    assert got["nums"] == {"arrayValue": {"values": [{"intValue": "1"}, {"intValue": "2"}]}}
    assert got["raw"] == {"bytesValue": "AP8="}


def test_non_finite_and_oversized_numbers_stay_valid_json():
    attrs = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "big": 2 ** 70}
    doc = encode_otlp([span(attrs=attrs)])
    text = json.dumps(doc, allow_nan=False)  # raises on a bare NaN
    got = {a["key"]: a["value"] for a in json.loads(text)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]}
    assert got["nan"] == {"doubleValue": "NaN"}
    assert got["inf"] == {"doubleValue": "Infinity"} and got["ninf"] == {"doubleValue": "-Infinity"}
    assert got["big"] == {"stringValue": str(2 ** 70)}  # beyond int64: the digits survive


def test_events_links_and_dropped_counts():
    event = SimpleNamespace(name="exception", timestamp=15, attributes={"exception.type": "ValueError"},
                            dropped_attributes=0)
    link = SimpleNamespace(context=SimpleNamespace(trace_id=0x5, span_id=0x6, trace_state=None), attributes={"k": 1})
    (s,) = values(encode_otlp([span(events=[event], links=[link], dropped=3)]))
    assert s["events"] == [{"timeUnixNano": "15", "name": "exception",
                            "attributes": [{"key": "exception.type", "value": {"stringValue": "ValueError"}}]}]
    assert s["links"][0]["traceId"] == "0" * 31 + "5" and s["links"][0]["spanId"] == "0" * 15 + "6"
    assert s["links"][0]["attributes"] == [{"key": "k", "value": {"intValue": "1"}}]
    assert s["droppedAttributesCount"] == 3


def test_spans_group_by_resource_and_scope_and_order_by_start():
    other = SimpleNamespace(attributes={"service.name": "other"}, schema_url="")
    spans = [span("late", sid=3, start=30), span("early", sid=1, start=10),
             span("elsewhere", sid=4, start=5, resource=other), span("scoped", sid=2, start=20, scope=("scope-b", "2"))]
    doc = encode_otlp(spans)
    # groups appear in the order of their earliest span
    assert [rs["resource"]["attributes"][0]["value"]["stringValue"] for rs in doc["resourceSpans"]] == ["other", "svc"]
    svc = doc["resourceSpans"][1]["scopeSpans"]
    assert [ss["scope"]["name"] for ss in svc] == ["scope-a", "scope-b"]
    assert [s["name"] for s in svc[0]["spans"]] == ["early", "late"]
    assert count_spans(doc) == 4


def test_empty_capture_is_an_empty_document():
    assert encode_otlp([]) == {"resourceSpans": []}
    assert count_spans({"resourceSpans": []}) == 0


# -- importing capture must stay light ---------------------------------------------

def test_importing_the_package_loads_no_framework_or_telemetry_module():
    script = (
        "import sys\n"
        "import agentpile as ap\n"
        "ap.capture, ap.config_from\n"
        "bad = sorted(m for m in ('opentelemetry', 'pydantic_ai', 'agents', 'openinference', 'pydantic') if m in sys.modules)\n"
        "print(','.join(bad))\n")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# -- attaching the configuration ---------------------------------------------------

def tool(name, agent=None, definition=None):
    return {"name": name, "agent_or_server": agent, "type": None, "definition": definition, "definition_format": None,
            "atif_index": None, "permission": "unknown", "permission_from": "unknown"}


def trajectory(agents, tools):
    return {"schema_version": "ATIF-v1.7", "session_id": "t", "trajectory_id": "t",
            "agent": {"name": "a", "version": "0"},
            "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
            "extra": {"agentpile": {"version": "0", "source": {"format": "otlp-json"}, "agents": agents, "tools": tools}}}


def config(agents, tools, lost=()):
    return {"schema": ap.CONFIG_SCHEMA,
            "source": {"format": "test", "lost_fields": list(lost)},
            "agents": agents, "tools": tools, "dataset": {"framework": "test", "framework_version": "1"}}


def agent(aid, tools=(), handoffs=(), **fields):
    base = {"id": aid, "name": aid, "instructions": None, "model": None}
    base.update(fields)
    return dict(base, tools=list(tools), handoffs=list(handoffs))


DEF = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}


def test_attach_fills_what_the_trace_left_empty_and_keeps_what_it_recorded():
    record = trajectory([agent("weather", ["t1"], model="observed-model")], {"t1": tool("get_weather", "weather")})
    cfg = config([agent("weather", ["weather.get_weather"], instructions="Be brief.", model="declared-model")],
                 {"weather.get_weather": tool("get_weather", "weather", DEF) | {"definition_format": "openai",
                                                                                "type": "function"}})
    attach_config(record, cfg)
    ext = record["extra"]["agentpile"]
    (a,) = ext["agents"]
    assert a["instructions"] == "Be brief." and a["model"] == "observed-model"  # the trace's value wins
    assert a["tools"] == ["t1"]  # matched to the trace's tool id, not duplicated
    assert ext["tools"]["t1"]["definition"] == DEF and ext["tools"]["t1"]["definition_format"] == "openai"
    assert set(ext["tools"]) == {"t1"}
    assert ap.validate(record) == []


def test_attach_adds_the_tools_an_agent_of_the_run_never_used():
    record = trajectory([agent("triage", ["t1"])], {"t1": tool("get_weather", "triage")})
    cfg = config([agent("triage", ["triage.get_weather", "triage.transfer"], handoffs=["billing"])],
                 {"triage.get_weather": tool("get_weather", "triage", DEF), "triage.transfer": tool("transfer", "triage", DEF)})
    attach_config(record, cfg)
    ext = record["extra"]["agentpile"]
    (triage,) = ext["agents"]
    assert triage["handoffs"] == ["billing"]  # a configured handoff stays; the target is not made an agent of the run
    assert triage["tools"] == ["t1", "triage.transfer"]  # the tool the trace never called is configuration of an agent that ran
    assert ext["tools"]["triage.transfer"]["agent_or_server"] == "triage"
    assert "config_agents_not_attached" not in ext["dataset"]["capture"]  # every configured agent is in the run
    assert ap.validate(record) == []


def test_attach_never_adds_an_agent_that_is_not_in_the_run():
    record = trajectory([agent("selected", ["t1"])], {"t1": tool("search", "selected")})
    cfg = config(
        [agent("selected", ["selected.search"], handoffs=["billing"]), agent("billing", ["billing.refund"])],
        {"selected.search": tool("search", "selected", DEF), "billing.refund": tool("refund", "billing", DEF)})
    attach_config(record, cfg)
    ext = record["extra"]["agentpile"]
    assert [a["id"] for a in ext["agents"]] == ["selected"]  # billing took no part in this run: no roster entry
    assert set(ext["tools"]) == {"t1"}  # and none of its tools came along
    assert ext["tools"]["t1"]["definition"] == DEF
    assert ext["dataset"]["capture"]["config_agents_not_attached"] == ["billing"]
    assert ap.validate(record) == []


def test_attach_does_not_pick_between_agents_that_share_a_name():
    record = trajectory([agent("a1", name="same"), agent("a2", name="same")], {})
    attach_config(record, config([agent("same", instructions="Do it.")], {}))
    ext = record["extra"]["agentpile"]
    assert [a["instructions"] for a in ext["agents"]] == [None, None]
    assert [a["id"] for a in ext["agents"]] == ["a1", "a2"]
    assert ext["dataset"]["capture"]["config_agents_not_attached"] == ["same"]


def test_attach_reaches_the_roster_of_a_subagent_trajectory():
    sub = trajectory([agent("sub")], {"s1": tool("lookup", "sub")})
    record = trajectory([agent("parent")], {})
    record["subagent_trajectories"] = [sub]
    cfg = config([agent("parent"), agent("sub", ["sub.lookup"], instructions="Look things up."), agent("idle")],
                 {"sub.lookup": tool("lookup", "sub", DEF)})
    attach_config(record, cfg)
    top, nested = record["extra"]["agentpile"], sub["extra"]["agentpile"]
    assert [a["id"] for a in top["agents"]] == ["parent"] and top["tools"] == {}
    assert [(a["id"], a["instructions"]) for a in nested["agents"]] == [("sub", "Look things up.")]
    assert nested["tools"]["s1"]["definition"] == DEF
    assert top["dataset"]["capture"]["config_agents_not_attached"] == ["idle"]  # only "idle" matched nowhere
    assert "dataset" not in nested or "capture" not in nested["dataset"]


def test_attach_matches_an_agent_by_name_when_ids_differ():
    record = trajectory([agent("run-agent-1", name="triage")], {})
    attach_config(record, config([agent("triage", instructions="Route.")], {}))
    (a,) = record["extra"]["agentpile"]["agents"]
    assert a["id"] == "run-agent-1" and a["instructions"] == "Route."


def test_attach_does_not_merge_tools_by_guess():
    # two trace tools called "search" and no owner to tell them apart: the configured one is added, not merged
    record = trajectory([agent("a")], {"x": tool("search", "s1"), "y": tool("search", "s2")})
    attach_config(record, config([agent("a", ["a.search"])], {"a.search": tool("search", "a", DEF)}))
    tools = record["extra"]["agentpile"]["tools"]
    assert tools["x"]["definition"] is None and tools["y"]["definition"] is None
    assert tools["a.search"]["definition"] == DEF


def test_attach_twice_changes_nothing():
    record = trajectory([agent("a", ["t1"])], {"t1": tool("get_weather", "a")})
    cfg = config([agent("a", ["a.get_weather"])], {"a.get_weather": tool("get_weather", "a", DEF)})
    attach_config(record, cfg)
    once = copy.deepcopy(record)
    attach_config(record, cfg)
    assert record == once


def test_attach_records_what_the_config_could_not_read():
    record = trajectory([agent("a")], {})
    attach_config(record, config([agent("a")], {}, lost=["agents.a.instructions: dynamic"]))
    note = record["extra"]["agentpile"]["dataset"]["capture"]
    assert note == {"framework": "test", "framework_version": "1",
                    "config_lost_fields": ["agents.a.instructions: dynamic"]}


# -- building runs from a document --------------------------------------------------

@contextlib.contextmanager
def reader_registered(obj):
    previous = ap.readers._items.get(READER, _ABSENT)
    ap.register(READER, obj, replace=True)
    try:
        yield
    finally:
        if previous is _ABSENT:
            ap.readers.unregister(READER)
        else:
            ap.readers._items[READER] = previous


@contextlib.contextmanager
def reader_missing():
    previous = ap.readers._items.pop(READER, _ABSENT)
    try:
        yield
    finally:
        if previous is not _ABSENT:
            ap.readers._items[READER] = previous


class StubReader:
    @staticmethod
    def sniff(source):
        return False

    @staticmethod
    def read(source, **options):
        yield ap.Run(trajectory([agent("weather", ["t1"])], {"t1": tool("get_weather", "weather")}))


DOC = {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"scope": {"name": "x"}, "spans": [{}]}]}]}


def test_build_runs_without_the_reader_says_so():
    with reader_missing():
        with pytest.raises(CaptureError, match="otlp-json.*not registered"):
            build_runs(DOC)


def test_build_runs_needs_a_span():
    with reader_registered(StubReader):
        with pytest.raises(CaptureError, match="no span was captured"):
            build_runs({"resourceSpans": []})


def test_build_runs_attaches_config_to_every_trajectory():
    cfg = config([agent("weather", ["weather.get_weather"], instructions="Be brief.")],
                 {"weather.get_weather": tool("get_weather", "weather", DEF)})
    with reader_registered(StubReader):
        (run,) = build_runs(DOC, cfg)
    assert run.agents[0]["instructions"] == "Be brief."
    assert run.tools["t1"]["definition"] == DEF
    assert ap.validate(run) == []


def test_build_runs_leaves_a_config_record_alone():
    class ConfigReader(StubReader):
        @staticmethod
        def read(source, **options):
            yield ap.Run({"schema": ap.CONFIG_SCHEMA, "source": {"format": "x"}, "agents": [], "tools": {}})

    with reader_registered(ConfigReader):
        (run,) = build_runs(DOC, config([agent("a")], {}))
    assert run.kind == "config" and run.agents == []


def test_errors_for_unknown_and_empty_input():
    with pytest.raises(TypeError, match="agent is required"):
        ap.capture(None)
    with pytest.raises(TypeError, match="empty list"):
        ap.capture([])
    with pytest.raises(TypeError, match="cannot tell which framework"):
        ap.config_from(object())


def test_the_framework_is_told_from_the_class_hierarchy_without_importing_it():
    pai = type("Agent", (), {"__module__": "pydantic_ai.agent"})
    sdk = type("Agent", (), {"__module__": "agents.agent"})
    assert integrations.framework_of(pai()) == "pydantic-ai"
    assert integrations.framework_of(sdk()) == "openai-agents"
    assert integrations.framework_of(type("Mine", (pai,), {"__module__": "my_app"})()) == "pydantic-ai"


def test_an_uninstalled_framework_is_reported_with_the_extra_that_installs_it(monkeypatch):
    monkeypatch.setitem(integrations.FRAMEWORKS, "pydantic-ai", ("agentpile.integrations.not_there", "pydantic-ai"))
    fake = type("Agent", (), {"__module__": "pydantic_ai.agent"})()
    with pytest.raises(CaptureError, match=r"pip install 'agentpile\[pydantic-ai\]'"):
        with ap.capture(fake):
            raise AssertionError("must not enter")
    with pytest.raises(CaptureError, match=r"pip install 'agentpile\[pydantic-ai\]'"):
        ap.config_from(fake)
