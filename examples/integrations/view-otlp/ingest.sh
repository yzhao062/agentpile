#!/usr/bin/env bash
# Ingestion check for one platform: start it locally, send the fixture runs with view_otlp.py --send, query them back,
# and stop it. Run from the repository root after installing requirements.txt and requirements-ingest.txt.
#   bash examples/integrations/view-otlp/ingest.sh phoenix|jaeger|mlflow
# Jaeger runs as the jaegertracing/jaeger container, or as a binary when JAEGER_BIN points at one (with jaeger.yaml,
# in memory). MLFLOW_PORT moves MLflow off 5000, which macOS gives to AirPlay.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="$1"
JAEGER_VERSION=2.22.0
WORK="$(mktemp -d)"
PID=""
CONTAINER=""

stop() {  # the platform this script started, and nothing else; SIGKILL if it outlives SIGTERM by 10 s
  if [ -n "$PID" ]; then
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$PID" 2>/dev/null || true
  fi
  if [ -n "$CONTAINER" ]; then docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; fi  # the id docker run returned
  rm -rf "$WORK"  # this script's own temporary directory
}
trap stop EXIT

ready() {  # wait up to 120 s for a URL to answer; give up at once if the platform process or container has exited
  for _ in $(seq 1 120); do
    if curl -fs -o /dev/null "$1"; then return 0; fi
    if [ -n "$PID" ] && ! kill -0 "$PID" 2>/dev/null; then break; fi
    if [ -n "$CONTAINER" ] && [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" != true ]; then break; fi
    sleep 1
  done
  echo "$PLATFORM did not come up at $1; its log:" >&2
  if [ -n "$CONTAINER" ]; then docker logs "$CONTAINER" >&2 || true; else cat "$WORK/$PLATFORM.log" >&2 || true; fi
  return 1
}

case "$PLATFORM" in
  phoenix)
    PHOENIX_WORKING_DIR="$WORK" phoenix serve > "$WORK/phoenix.log" 2>&1 &
    PID=$!
    ready http://127.0.0.1:6006/healthz
    ;;
  jaeger)
    if [ -n "${JAEGER_BIN:-}" ]; then
      "$JAEGER_BIN" --config "$HERE/jaeger.yaml" > "$WORK/jaeger.log" 2>&1 &
      PID=$!
    else
      # No --rm and no fixed name: the log survives a failed start, and cleanup removes only this container.
      CONTAINER=$(docker run -d -p 4318:4318 -p 16686:16686 "jaegertracing/jaeger:$JAEGER_VERSION")
    fi
    ready http://127.0.0.1:16686/
    ;;
  mlflow)
    PORT="${MLFLOW_PORT:-5000}"
    mlflow server --host 127.0.0.1 --port "$PORT" --backend-store-uri "sqlite:///$WORK/mlflow.db" > "$WORK/mlflow.log" 2>&1 &
    PID=$!
    ready "http://127.0.0.1:$PORT/health"
    export MLFLOW_ENDPOINT="http://127.0.0.1:$PORT/v1/traces"
    ;;
  *)
    echo "usage: ingest.sh phoenix|jaeger|mlflow" >&2
    exit 2
    ;;
esac

python "$HERE/view_otlp.py" --send "$PLATFORM"
