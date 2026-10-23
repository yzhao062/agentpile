"""View agent trajectories in OpenTelemetry observability backends.

agentpile exports any loaded run to standard OTLP JSON (conforming to OpenInference
and OpenTelemetry GenAI semantic conventions) and posts it directly to OTLP/HTTP
receivers including Jaeger, Arize Phoenix, MLflow, and Langfuse.

Two modes:
- Default (offline, CI): exports fixture runs (tau_bench, trail, and OTLP fixtures)
  with run.to_otlp(), checks the ids as OTLP JSON writes them (hex, 32 characters for
  a trace and 16 for a span, not all zero, for spans, parents and links), parents that
  exist, and end times not before start times, parses each document with
  opentelemetry-proto, and asserts lossless round-trip loading. A document with a
  malformed trace id must fail the same check.
- Send mode (--send PRESET[,PRESET...]): posts runs with ap.send_otlp() and queries
  each platform's API to confirm trace ingestion.

Usage:
    pip install -e ".[hf]" -r examples/integrations/view-otlp/requirements.txt
    python examples/integrations/view-otlp/view_otlp.py
    python examples/integrations/view-otlp/view_otlp.py --send jaeger,phoenix,mlflow

Preset endpoint and authentication overrides:
    PHOENIX_ENDPOINT, PHOENIX_QUERY_ENDPOINT (defaults: http://localhost:6006/v1/traces, http://localhost:6006)
    JAEGER_ENDPOINT, JAEGER_QUERY_ENDPOINT   (defaults: http://localhost:4318/v1/traces, http://localhost:16686)
    MLFLOW_ENDPOINT, MLFLOW_TRACKING_URI     (defaults: http://localhost:5000/v1/traces, http://localhost:5000)
    MLFLOW_EXPERIMENT_ID                     (default: 0)
    LANGFUSE_ENDPOINT                        (default: http://localhost:3000/api/public/otel/v1/traces)
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from google.protobuf.json_format import Parse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

import agentpile as ap

HEX = re.compile(r"[0-9a-fA-F]+")


def load_runs(hub: bool = False):
    """Load fixture runs from the repository checkout or from Hugging Face Hub."""
    if hub:
        return list(ap.load("tau_bench", limit=2)) + list(ap.load("trail", limit=2))
    repo = Path(__file__).resolve().parents[3]
    corp = repo / "tests" / "corpora" / "fixtures"
    runs = list(ap.load("tau_bench", root=corp / "tau_bench")) + list(ap.load("trail", root=corp / "trail"))
    for p in sorted((repo / "tests" / "fixtures" / "otlp").glob("*")):
        if p.suffix in (".json", ".jsonl") and p.name not in ("versions.json", "not_a_trace.json"):
            runs += list(ap.load(p))
    return runs


def _spans(doc):
    return [s for rs in doc.get("resourceSpans") or [] for ss in rs.get("scopeSpans") or [] for s in ss.get("spans") or []]


def _hex_id(value, size):
    """An OTLP JSON id: hex of the given length, not all zeros (protobuf's own JSON parser would read it as base64)."""
    return isinstance(value, str) and len(value) == size and bool(HEX.fullmatch(value)) and set(value) != {"0"}


def problems(doc):
    """What keeps a document from being valid OTLP JSON for a trace viewer; empty when nothing does."""
    found, spans = [], _spans(doc)
    ids = {s.get("spanId") for s in spans}
    for s in spans:
        if not (_hex_id(s.get("traceId"), 32) and _hex_id(s.get("spanId"), 16) and s.get("name")):
            found.append(f"span ids or name: {s.get('traceId')!r} {s.get('spanId')!r} {s.get('name')!r}")
        pid = s.get("parentSpanId")
        if pid and not (_hex_id(pid, 16) and pid in ids):
            found.append(f"parent {pid!r} of span {s.get('spanId')!r}")
        if int(s.get("endTimeUnixNano") or 0) < int(s.get("startTimeUnixNano") or 0):
            found.append(f"span {s.get('spanId')!r} ends before it starts")
        for link in s.get("links") or []:
            if not (_hex_id(link.get("traceId"), 32) and _hex_id(link.get("spanId"), 16)):
                found.append(f"link {link!r} of span {s.get('spanId')!r}")
    return found


def as_protobuf_json(doc):
    """The document with its hex ids as base64, the form protobuf's JSON parser reads bytes fields in."""
    doc = json.loads(json.dumps(doc))
    for s in _spans(doc):
        for item in [s] + list(s.get("links") or []):
            for key in ("traceId", "spanId", "parentSpanId"):
                if item.get(key):
                    item[key] = base64.b64encode(bytes.fromhex(item[key])).decode("ascii")
    return doc


def check_offline(runs):
    """Check each run's OTLP export offline: ids, parents and times, a protobuf parse, and a lossless reload."""
    total_spans = 0
    for run in runs:
        doc = json.loads(run.to_otlp())
        assert not problems(doc), f"{run.id}: {problems(doc)[:3]}"
        request = Parse(json.dumps(as_protobuf_json(doc)), ExportTraceServiceRequest())
        parsed = [s for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans]
        assert len(parsed) == len(_spans(doc)) and all(len(s.trace_id) == 16 and len(s.span_id) == 8 for s in parsed)
        total_spans += len(parsed)
        rt = next(ap.load(doc))
        assert json.loads(run.to_json()) == json.loads(rt.to_json()), f"Roundtrip mismatch for {run.id}"
    # control: a malformed trace id (protobuf's JSON parser alone reads "abcd" as base64 and accepts it) is caught
    bad = json.loads(runs[0].to_otlp())
    _spans(bad)[0]["traceId"] = "abcd"
    assert problems(bad), "the id check accepted a malformed trace id"
    return total_spans


def _http_json(url, data=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8") if data is not None else None, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        return {}


def confirmed(ids, found, wait=60.0):
    """How many of the sent traces a backend reports, asking again while it is still writing them (Phoenix and
    Jaeger store spans in the background, so a query right after the POST can miss them)."""
    deadline = time.monotonic() + wait
    while True:
        count = sum(1 for t in ids if found(t))
        if count == len(ids) or time.monotonic() > deadline:
            return count
        time.sleep(2)


def send_preset(preset, runs):
    """Send runs to one backend preset and query its API to confirm ingestion."""
    name = preset.strip().lower()
    if name == "jaeger":
        ep = os.environ.get("JAEGER_ENDPOINT", "http://localhost:4318/v1/traces")
        qbase = os.environ.get("JAEGER_QUERY_ENDPOINT", "http://localhost:16686")
        ids = ap.send_otlp(runs, ep)
        return name, len(ids), confirmed(ids, lambda t: _http_json(f"{qbase}/api/traces/{t}").get("data"))

    if name == "mlflow":
        ep = os.environ.get("MLFLOW_ENDPOINT", "http://localhost:5000/v1/traces")
        qbase = os.environ.get("MLFLOW_TRACKING_URI", ep.replace("/v1/traces", ""))
        exp = os.environ.get("MLFLOW_EXPERIMENT_ID", "0")
        ids = ap.send_otlp(runs, ep, headers={"x-mlflow-experiment-id": exp})
        return name, len(ids), confirmed(ids, lambda t: _http_json(
            f"{qbase}/api/3.0/mlflow/traces/get?trace_id=tr-{t}").get("trace", {}).get("spans"))

    if name == "phoenix":
        ep = os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
        qbase = os.environ.get("PHOENIX_QUERY_ENDPOINT", ep.replace("/v1/traces", ""))
        ids = ap.send_otlp(runs, ep)
        headers = {"Content-Type": "application/json"}
        q = "query($p: ID!, $t: ID!) { node(id: $p) { ... on Project { trace(traceId: $t) { id } } } }"

        def found(t):  # each dataset is its own Phoenix project, created as its first span arrives
            projects = _http_json(f"{qbase}/graphql", {"query": "{ projects { edges { node { id } } } }"}, headers)
            pids = [e["node"]["id"] for e in projects.get("data", {}).get("projects", {}).get("edges", [])]
            return any(_http_json(f"{qbase}/graphql", {"query": q, "variables": {"p": p, "t": t}}, headers)
                       .get("data", {}).get("node", {}).get("trace") for p in pids)

        return name, len(ids), confirmed(ids, found)

    if name == "langfuse":
        ep = os.environ.get("LANGFUSE_ENDPOINT", "http://localhost:3000/api/public/otel/v1/traces")
        pk, sk = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
        if not pk or not sk:
            raise SystemExit("langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY for the project to send to")
        auth = {"Authorization": "Basic " + base64.b64encode(f"{pk}:{sk}".encode("utf-8")).decode("ascii")}
        ids = ap.send_otlp(runs, ep, headers=auth)
        base = ep.replace("/api/public/otel/v1/traces", "")
        return name, len(ids), confirmed(ids, lambda t: _http_json(f"{base}/api/public/traces/{t}", headers=auth).get("id"))

    raise ValueError(f"Unknown preset: {preset}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", action="store_true", help="Load from Hugging Face Hub instead of checkout")
    parser.add_argument("--send", help="Comma-separated backend presets to send traces: phoenix, jaeger, mlflow, langfuse")
    args = parser.parse_args()

    runs = load_runs(hub=args.hub)
    if not args.send:
        spans = check_offline(runs)
        print(f"offline checks passed: {len(runs)} runs, {spans} spans")
        return
    for preset in args.send.split(","):
        name, sent, found = send_preset(preset, runs)
        print(f"{name}: sent {sent} traces, confirmed {found} ingested")
        assert sent and found == sent, f"{name}: confirmed {found} of {sent} traces"


if __name__ == "__main__":
    main()
