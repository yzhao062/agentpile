"""Open runs in an OpenTelemetry trace viewer.

A run exports to OTLP JSON (OpenInference and OpenTelemetry GenAI conventions), and ap.send_otlp posts it to the
OTLP/HTTP receiver of Phoenix, Jaeger, MLflow or Langfuse. The checks of the bundled fixtures read each export back
and, with --send, query the platform for every trace.

    pip install -e ".[hf]" -r examples/integrations/view-otlp/requirements.txt
    python examples/integrations/view-otlp/view_otlp.py                        # the bundled fixtures, with the checks
    python examples/integrations/view-otlp/view_otlp.py tau_bench --limit 20   # your runs: a dataset name or a trace file

Without --send the script only converts the runs and counts their spans. Add --send phoenix, jaeger, mlflow or langfuse to post them to a platform that
is running; the script then prints the URL to open. Start the platform first (Phoenix and MLflow come from
`pip install -r examples/integrations/view-otlp/requirements-ingest.txt`):

    phoenix serve                                                          # http://localhost:6006
    docker run -d -p 4318:4318 -p 16686:16686 jaegertracing/jaeger:2.22.0  # http://localhost:16686
    mlflow server --backend-store-uri sqlite:///mlflow.db                  # http://localhost:5000

Langfuse (http://localhost:3000) also needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY set to a project's keys. To move
a platform off its default address, set <PLATFORM>_ENDPOINT (the traces URL) and <PLATFORM>_URL (the web UI), for example
MLFLOW_ENDPOINT=http://localhost:5001/v1/traces and MLFLOW_URL=http://localhost:5001.
"""

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from itertools import islice
from pathlib import Path

from google.protobuf.json_format import Parse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

import agentpile as ap

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
OTLP_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "otlp"
HEX = re.compile(r"[0-9a-fA-F]+")
PLATFORMS = {  # platform: the OTLP/HTTP endpoint that takes traces, and the web UI that shows them
    "phoenix": ("http://localhost:6006/v1/traces", "http://localhost:6006"),
    "jaeger": ("http://localhost:4318/v1/traces", "http://localhost:16686"),
    "mlflow": ("http://localhost:5000/v1/traces", "http://localhost:5000"),
    "langfuse": ("http://localhost:3000/api/public/otel/v1/traces", "http://localhost:3000"),
}


def urls(platform):
    """The endpoint that takes traces and the web UI, unless <PLATFORM>_ENDPOINT and <PLATFORM>_URL move them."""
    endpoint, ui = PLATFORMS[platform]
    return os.environ.get(f"{platform.upper()}_ENDPOINT", endpoint), os.environ.get(f"{platform.upper()}_URL", ui)


def headers(platform):
    if platform == "mlflow":  # MLflow files traces under an experiment
        return {"x-mlflow-experiment-id": os.environ.get("MLFLOW_EXPERIMENT_ID", "0")}
    if platform == "langfuse":  # Langfuse takes the project's keys as basic auth
        keys = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}"
        return {"Authorization": "Basic " + base64.b64encode(keys.encode("utf-8")).decode("ascii")}
    return {}


def send(runs, platform):
    """The agentpile part: post each run to the platform's OTLP/HTTP endpoint and return the trace ids."""
    return ap.send_otlp(runs, urls(platform)[0], headers=headers(platform))


def spans_of(doc):
    return [s for rs in doc.get("resourceSpans") or [] for ss in rs.get("scopeSpans") or [] for s in ss.get("spans") or []]


def hex_id(value, size):
    """An OTLP JSON id: hex of the given length, not all zeros (protobuf's own JSON parser would read it as base64)."""
    return isinstance(value, str) and len(value) == size and bool(HEX.fullmatch(value)) and set(value) != {"0"}


def problems(doc):
    """What keeps a document from being valid OTLP JSON for a trace viewer; empty when nothing does."""
    found, spans = [], spans_of(doc)
    ids = {s.get("spanId") for s in spans}
    for s in spans:
        if not (hex_id(s.get("traceId"), 32) and hex_id(s.get("spanId"), 16) and s.get("name")):
            found.append(f"span ids or name: {s.get('traceId')!r} {s.get('spanId')!r} {s.get('name')!r}")
        pid = s.get("parentSpanId")
        if pid and not (hex_id(pid, 16) and pid in ids):
            found.append(f"parent {pid!r} of span {s.get('spanId')!r}")
        if int(s.get("endTimeUnixNano") or 0) < int(s.get("startTimeUnixNano") or 0):
            found.append(f"span {s.get('spanId')!r} ends before it starts")
        for link in s.get("links") or []:
            if not (hex_id(link.get("traceId"), 32) and hex_id(link.get("spanId"), 16)):
                found.append(f"link {link!r} of span {s.get('spanId')!r}")
    return found


def as_protobuf_json(doc):
    """The document with its hex ids as base64, the form protobuf's JSON parser reads bytes fields in."""
    doc = json.loads(json.dumps(doc))
    for s in spans_of(doc):
        for item in [s] + list(s.get("links") or []):
            for key in ("traceId", "spanId", "parentSpanId"):
                if item.get(key):
                    item[key] = base64.b64encode(bytes.fromhex(item[key])).decode("ascii")
    return doc


def http_json(url, data=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8") if data is not None else None, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        return {}


def confirmed(platform, ids, wait=60.0):
    """How many of the sent traces the platform reports, asking its API again while it is still writing them (Phoenix
    and Jaeger store spans in the background, so a query right after the POST can miss them)."""
    ui, auth = urls(platform)[1], headers(platform)
    json_type = {"Content-Type": "application/json"}
    query = "query($p: ID!, $t: ID!) { node(id: $p) { ... on Project { trace(traceId: $t) { id } } } }"

    def phoenix(t):  # each dataset is its own Phoenix project, created as its first span arrives
        projects = http_json(f"{ui}/graphql", {"query": "{ projects { edges { node { id } } } }"}, json_type)
        pids = [e["node"]["id"] for e in projects.get("data", {}).get("projects", {}).get("edges", [])]
        return any(http_json(f"{ui}/graphql", {"query": query, "variables": {"p": p, "t": t}}, json_type)
                   .get("data", {}).get("node", {}).get("trace") for p in pids)

    found = {
        "phoenix": phoenix,
        "jaeger": lambda t: http_json(f"{ui}/api/traces/{t}").get("data"),
        "mlflow": lambda t: http_json(f"{ui}/api/3.0/mlflow/traces/get?trace_id=tr-{t}").get("trace", {}).get("spans"),
        "langfuse": lambda t: http_json(f"{ui}/api/public/traces/{t}", headers=auth).get("id"),
    }[platform]
    deadline = time.monotonic() + wait
    while True:
        count = sum(1 for t in ids if found(t))
        if count == len(ids) or time.monotonic() > deadline:
            return count
        time.sleep(2)


def check(runs, platform=None, ids=()):
    """Check each run's OTLP export offline (ids, parents and times, a protobuf parse, a lossless reload); with a
    platform, also ask it for every trace sent."""
    for run in runs:
        doc = json.loads(run.to_otlp())
        assert not problems(doc), f"{run.id}: {problems(doc)[:3]}"
        request = Parse(json.dumps(as_protobuf_json(doc)), ExportTraceServiceRequest())
        parsed = [s for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans]
        assert len(parsed) == len(spans_of(doc)) and all(len(s.trace_id) == 16 and len(s.span_id) == 8 for s in parsed)
        rt = next(ap.load(doc))
        assert json.loads(run.to_json()) == json.loads(rt.to_json()), f"Roundtrip mismatch for {run.id}"
    # control: a malformed trace id (protobuf's JSON parser alone reads "abcd" as base64 and accepts it) is caught
    bad = json.loads(runs[0].to_otlp())
    spans_of(bad)[0]["traceId"] = "abcd"
    assert problems(bad), "the id check accepted a malformed trace id"
    if platform:
        found = confirmed(platform, ids)
        assert ids and found == len(ids), f"{platform}: confirmed {found} of {len(ids)} traces"


def load_runs(sources, limit):
    """The first `limit` runs of each source; a dataset name takes limit= and downloads only that part, a file path does not."""
    return [run for source in sources
            for run in islice(ap.load(source, **({"limit": limit} if source in ap.datasets.names() else {})), limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="*", help="a dataset name or a trace file (default: the bundled fixtures, with the checks)")
    parser.add_argument("--limit", type=int, default=20, help="runs per source (default: 20)")
    parser.add_argument("--send", choices=list(PLATFORMS), help="post the runs to this platform, which must be running")
    args = parser.parse_args()
    if args.source:
        runs = load_runs(args.source, args.limit)
    else:
        runs = [run for name in ("tau_bench", "trail") for run in ap.load(name, root=FIXTURES / name)]
        runs += [run for path in sorted(OTLP_FIXTURES.glob("*.json*")) if path.name != "versions.json"
                 for run in ap.load(path)]
    ids = []
    if args.send:
        ids = send(runs, args.send)
        print(f"sent {len(ids)} traces to {args.send}")
        print(f"open with: {urls(args.send)[1]}")
    else:
        spans = sum(len(spans_of(json.loads(run.to_otlp()))) for run in runs)
        print(f"converted {len(runs)} runs to {spans} OTLP spans; --send PLATFORM posts them")
    if not args.source:
        check(runs, args.send, ids)
        print("all checks passed")


if __name__ == "__main__":
    main()
