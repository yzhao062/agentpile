"""Send runs to an OTLP/HTTP receiver, so a loaded run opens in a tracing UI such as Jaeger or Arize Phoenix."""

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Union

from .record import Run

JSON = "application/json"
PROTOBUF = "application/x-protobuf"


def send_otlp(runs: Union[Run, Iterable[Run]], endpoint: str = "http://localhost:4318/v1/traces",
              headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> List[str]:
    """POST each run's OTLP spans to ``endpoint`` and return the trace ids, one per run.

    The body is OTLP JSON, which Jaeger, MLflow, the OpenTelemetry Collector
    and most OTLP/HTTP receivers accept, so the standard library is enough. A
    receiver that answers 415 (Arize Phoenix takes protobuf only) gets the same
    spans as protobuf from then on, which needs opentelemetry-proto:
    pip install 'agentpile[otel]'. ``headers`` go with every request (MLflow
    wants ``x-mlflow-experiment-id``; a hosted service wants its auth header).
    A config record has no spans and raises ValueError, as ``to_otlp`` does.
    Runs are sent one at a time as an iterable yields them, so
    ``send_otlp(ap.load(name), ...)`` never holds a whole dataset in memory.
    """
    items = (runs,) if isinstance(runs, Run) else runs
    ids = []
    content_type = JSON
    for run in items:
        body = run.to_otlp()
        if content_type == JSON:
            try:
                _post(endpoint, body.encode("utf-8"), JSON, headers, timeout)
            except urllib.error.HTTPError as exc:
                if exc.code != 415:
                    raise
                content_type = PROTOBUF
        if content_type == PROTOBUF:
            _post(endpoint, _protobuf(json.loads(body)), PROTOBUF, headers, timeout)
        ids.append(_trace_id(json.loads(body)))
    return ids


def _post(endpoint: str, data: bytes, content_type: str, headers: Optional[Dict[str, str]], timeout: float) -> None:
    sent = {key: value for key, value in (headers or {}).items() if key.lower() != "content-type"}
    sent["Content-Type"] = content_type
    request = urllib.request.Request(endpoint, data=data, headers=sent, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def _spans(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for resource in doc.get("resourceSpans") or []:
        for scope in resource.get("scopeSpans") or []:
            yield from scope.get("spans") or []


def _trace_id(doc: Dict[str, Any]) -> str:
    return next((s["traceId"] for s in _spans(doc) if s.get("traceId")), "")


def _protobuf(doc: Dict[str, Any]) -> bytes:
    """The OTLP JSON document as an ExportTraceServiceRequest.

    OTLP JSON writes trace and span ids as hex, while protobuf's JSON parser
    reads bytes fields as base64, so the ids (also those in span links) are
    re-encoded first; parsing the hex directly yields ids of the wrong length.
    """
    try:
        from google.protobuf.json_format import ParseDict
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    except ImportError as exc:
        raise ImportError("this receiver takes only protobuf, which needs opentelemetry-proto: "
                          "pip install 'agentpile[otel]'") from exc
    for span in _spans(doc):
        for item in [span] + list(span.get("links") or []):
            for key in ("traceId", "spanId", "parentSpanId"):
                if item.get(key):
                    item[key] = base64.b64encode(bytes.fromhex(item[key])).decode("ascii")
    return ParseDict(doc, ExportTraceServiceRequest(), ignore_unknown_fields=True).SerializeToString()
