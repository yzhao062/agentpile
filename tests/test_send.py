"""ap.send_otlp against a local HTTP receiver: JSON first, protobuf after a 415, no network."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

import agentpile as ap

MESSAGES = [
    {"role": "user", "content": "Is it raining in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "light rain, 14 C"},
    {"role": "assistant", "content": "Yes, light rain."},
]


class Receiver:
    """A one-endpoint OTLP/HTTP stand-in that records each request and answers with the status for its content type."""

    def __init__(self, status_by_type):
        self.status_by_type = status_by_type
        self.requests = []
        self.headers = []
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                ctype = self.headers["Content-Type"]
                receiver.requests.append((self.path, ctype, body))
                receiver.headers.append(self.headers)  # an email.message.Message: lookups ignore case
                self.send_response(receiver.status_by_type.get(ctype, 400))
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/v1/traces"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def receiver_factory():
    made = []

    def make(status_by_type):
        made.append(Receiver(status_by_type))
        return made[-1]

    yield make
    for receiver in made:
        receiver.close()


def _run():
    return next(ap.load(MESSAGES))


def test_a_json_receiver_gets_the_otlp_json_of_each_run(receiver_factory):
    rx = receiver_factory({"application/json": 200})
    run = _run()
    ids = ap.send_otlp([run, run.with_deps(rules=["tool_call_id"])], rx.endpoint)
    assert len(rx.requests) == 2 and all(r[0] == "/v1/traces" and r[1] == "application/json" for r in rx.requests)
    assert json.loads(rx.requests[0][2]) == json.loads(run.to_otlp())
    assert ids[0] == json.loads(run.to_otlp())["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]
    assert len(ids) == 2 and all(len(i) == 32 for i in ids)


def test_a_single_run_is_accepted_without_a_list(receiver_factory):
    rx = receiver_factory({"application/json": 200})
    assert len(ap.send_otlp(_run(), rx.endpoint)) == 1


def test_extra_headers_go_with_the_request_and_cannot_change_the_content_type(receiver_factory):
    rx = receiver_factory({"application/json": 200})
    both = {"x-mlflow-experiment-id": "0", "Content-Type": "x", "content-type": "text/plain"}
    ap.send_otlp(_run(), rx.endpoint, headers=both)
    assert rx.headers[0]["x-mlflow-experiment-id"] == "0" and rx.requests[0][1] == "application/json"


def test_the_protobuf_retry_also_keeps_its_own_content_type(receiver_factory):
    pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    rx = receiver_factory({"application/x-protobuf": 200, "application/json": 415})
    ap.send_otlp(_run(), rx.endpoint, headers={"Content-Type": "x", "content-type": "text/plain"})
    assert [r[1] for r in rx.requests] == ["application/json", "application/x-protobuf"]


def test_runs_are_sent_as_the_iterable_yields_them(receiver_factory):
    rx = receiver_factory({"application/json": 200})
    seen = []

    def runs():
        for i in range(3):
            seen.append(("yield", i, len(rx.requests)))  # how many posts arrived before this run was asked for
            yield _run()

    assert len(ap.send_otlp(runs(), rx.endpoint)) == 3
    assert seen == [("yield", 0, 0), ("yield", 1, 1), ("yield", 2, 2)]


def test_a_415_switches_to_protobuf_for_this_and_later_runs(receiver_factory):
    trace_pb2 = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    rx = receiver_factory({"application/x-protobuf": 200, "application/json": 415})
    run = _run().with_deps(rules=["tool_call_id"])
    ids = ap.send_otlp([run, _run()], rx.endpoint)
    assert [r[1] for r in rx.requests] == ["application/json", "application/x-protobuf", "application/x-protobuf"]
    sent = trace_pb2.ExportTraceServiceRequest.FromString(rx.requests[1][2])
    spans = [s for rs in sent.resource_spans for ss in rs.scope_spans for s in ss.spans]
    doc = json.loads(run.to_otlp())
    want = [s for rs in doc["resourceSpans"] for ss in rs["scopeSpans"] for s in ss["spans"]]
    assert [s.name for s in spans] == [s["name"] for s in want]
    assert {s.trace_id.hex() for s in spans} == {ids[0]}
    assert [s.span_id.hex() for s in spans] == [s["spanId"] for s in want]
    links = [(lk.trace_id.hex(), lk.span_id.hex()) for s in spans for lk in s.links]
    want_links = [(lk["traceId"], lk["spanId"]) for s in want for lk in s.get("links") or []]
    assert links == want_links and links


def test_another_error_status_is_raised(receiver_factory):
    rx = receiver_factory({"application/json": 500})
    with pytest.raises(HTTPError):
        ap.send_otlp(_run(), rx.endpoint)


def test_protobuf_without_opentelemetry_proto_names_the_extra(receiver_factory, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.protobuf.json_format", None)
    rx = receiver_factory({"application/json": 415})
    with pytest.raises(ImportError, match=r"agentpile\[otel\]"):
        ap.send_otlp(_run(), rx.endpoint)


def test_a_config_record_has_no_spans_to_send(receiver_factory):
    rx = receiver_factory({"application/json": 200})
    config = next(ap.load(str(Path(__file__).parent / "fixtures" / "config_mcp.json")))
    with pytest.raises(ValueError):
        ap.send_otlp(config, rx.endpoint)
    assert rx.requests == []


def test_the_otel_extra_is_protobuf_alone_and_no_extra_outgrows_colab():
    # Google Colab's google-adk caps OpenTelemetry at 1.42.1, so no extra may ask for more than that.
    import re

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r"""^otel = \["opentelemetry-proto>=1\.42\.1; python_version >= '3\.10'"\]$""", pyproject, re.M)
    floors = re.findall(r'"opentelemetry-[a-z-]+>=([0-9.]+)', pyproject)
    assert floors and set(floors) == {"1.42.1"}
