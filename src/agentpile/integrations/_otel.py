"""OpenTelemetry plumbing shared by the framework integrations.

This module imports ``opentelemetry`` when it is imported, so it is only ever
imported from inside a function (``capture()`` on entry), never at
``import agentpile``.

A capture owns one ``Session``. The session holds the in-memory exporter that
stores the finished spans, and every ``TracerProvider`` the capture creates
(one per host provider it has to forward to, or one plain provider). It never
touches the global tracer provider, and it never shuts down a provider it did
not create.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

DEFAULT_SERVICE_NAME = "agentpile-capture"

# A span of a long agent conversation carries one attribute per message field
# (OpenInference) or one event per delta, so the SDK default of 128 would drop
# the tail of a long run without any error. Dropped counts still reach the
# document if a limit is hit.
_LIMIT = 4096


class _Collector(SpanProcessor):
    """Stores each finished span once in the session's exporter, until the session is closed."""

    def __init__(self, session: "Session") -> None:
        self._session = session

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if not self._session.closed:
            self._session._count(started=1)

    def on_end(self, span: Any) -> None:
        if self._session.closed:
            return
        self._session._count(ended=1)
        self._session.exporter.export((span,))

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _Forwarder(SpanProcessor):
    """Hands every span of a capture provider to the host's own span processor, so the host sees it once.

    The capture's provider replaces the host's tracer for the agent, and one
    tracer binds one provider. Without this the host's exporters would miss
    every span of a captured run. ``shutdown`` and ``force_flush`` are not
    forwarded: the host owns its processor.
    """

    def __init__(self, host: Any) -> None:
        self._host = host

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        self._host.on_start(span, parent_context=parent_context)

    def _on_ending(self, span: Any) -> None:
        hook = getattr(self._host, "_on_ending", None)
        if hook is not None:
            hook(span)

    def on_end(self, span: Any) -> None:
        self._host.on_end(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _span_limits() -> Any:
    from opentelemetry.sdk.trace import SpanLimits

    for kwargs in (
        dict(max_span_attributes=_LIMIT, max_events=_LIMIT, max_links=_LIMIT, max_event_attributes=_LIMIT,
             max_link_attributes=_LIMIT, max_span_attribute_length=1 << 30),
        dict(max_span_attributes=_LIMIT, max_events=_LIMIT, max_links=_LIMIT, max_event_attributes=_LIMIT,
             max_link_attributes=_LIMIT),
    ):
        try:
            return SpanLimits(**kwargs)
        except TypeError:  # an older SDK without one of the parameters
            continue
    return None


class Session:
    """The span store of one capture, and the factory of its tracer providers."""

    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.closed = False
        self.started = 0
        self.ended = 0
        self.shared: Dict[str, Any] = {}  # per-framework state shared by the bindings of one capture
        self._lock = threading.Lock()
        self._providers: List[TracerProvider] = []

    def _count(self, started: int = 0, ended: int = 0) -> None:
        with self._lock:
            self.started += started
            self.ended += ended

    def tracer_provider(self, resource: Optional[Resource] = None, forward_to: Any = None) -> TracerProvider:
        """A new provider that records into this session and, if ``forward_to`` is given, also feeds that host processor."""
        provider = TracerProvider(
            sampler=ALWAYS_ON,
            resource=resource or Resource.create({"service.name": DEFAULT_SERVICE_NAME}),
            shutdown_on_exit=False,  # no atexit hook: the session closes its own providers
            span_limits=_span_limits(),
        )
        provider.add_span_processor(_Collector(self))
        if forward_to is not None:
            provider.add_span_processor(_Forwarder(forward_to))
        self._providers.append(provider)
        return provider

    def spans(self) -> Tuple[Any, ...]:
        return tuple(self.exporter.get_finished_spans())

    @property
    def unfinished(self) -> int:
        """Spans that started during the capture and had not ended when it was read."""
        with self._lock:
            return max(self.started - self.ended, 0)

    def close(self) -> None:
        """Stop collecting, and shut down the providers this session created (never a host provider)."""
        self.closed = True
        for provider in self._providers:
            try:
                provider.shutdown()
            except Exception:  # a provider that fails to shut down must not hide the run's own error
                pass


def host_span_processor(tracer: Any) -> Tuple[str, Any, Optional[Resource]]:
    """Find what a host tracer feeds: ``("sdk", processor, resource)``, ``("none", None, None)`` or ``("unknown", None, None)``.

    An SDK tracer exposes its provider's active span processor as
    ``span_processor``. A proxy tracer (no global provider set yet) resolves to
    a no-op tracer, which feeds nothing. Any other tracer is looked through
    one wrapper level at a time; if no SDK tracer is found, the answer is
    ``"unknown"`` and the caller warns.
    """
    seen = 0
    current = tracer
    while current is not None and seen < 6:
        seen += 1
        processor = getattr(current, "span_processor", None)
        if processor is not None:
            return "sdk", processor, getattr(current, "resource", None)
        if type(current).__name__ in ("NoOpTracer", "NonRecordingTracer"):
            return "none", None, None
        nxt = None
        for attr in ("_tracer", "_real_tracer", "tracer", "_self_tracer", "wrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                nxt = candidate
                break
        current = nxt
    return "unknown", None, None
