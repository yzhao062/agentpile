"""Open runs in Inspect AI's log viewer.

ap.to_chat gives each run an OpenAI-style message list. This maps the messages to Inspect's chat message types (system,
user, assistant with tool calls, tool) and saves the runs with Inspect's own writer as one .eval log, one sample per
run, with the run id, dataset and outcome as sample metadata. The checks of the bundled fixtures read the log back
with read_eval_log, compare every sample with its chat row, and fetch the log from `inspect view`.

    pip install -e ".[hf]" -r examples/integrations/view-inspect/requirements.txt
    python examples/integrations/view-inspect/view_inspect.py                        # the bundled fixtures, with the checks
    python examples/integrations/view-inspect/view_inspect.py tau_bench --limit 20   # your runs: a dataset name or a trace file
    inspect view start --log-dir DIR                                                 # DIR is the folder the script prints
"""

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import agentpile as ap
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
DATASETS = ("tau_bench", "tau2_bench", "agentdojo")
LOG = "agentpile_runs.eval"


def inspect_messages(row):
    """One chat row (ap.to_chat with arguments="dict") as Inspect chat messages."""
    messages, called = [], {}
    for m in row["messages"]:
        text = m["content"] or ""
        if m["role"] == "system":
            messages.append(ChatMessageSystem(content=text))
        elif m["role"] == "user":
            messages.append(ChatMessageUser(content=text))
        elif m["role"] == "assistant":
            calls = [ToolCall(id=c["id"], function=c["function"]["name"], arguments=c["function"]["arguments"])
                     for c in m.get("tool_calls", [])]
            called.update({c.id: c.function for c in calls})
            messages.append(ChatMessageAssistant(content=text, tool_calls=calls or None))
        else:
            messages.append(ChatMessageTool(content=text, tool_call_id=m["tool_call_id"],
                                            function=called.get(m["tool_call_id"])))
    return messages


def write_log(runs, path):
    """The agentpile part: one sample per run that has a chat row, written as an Inspect .eval log. Returns the count."""
    report = {}
    chat = ap.to_chat(runs, arguments="dict", report=report)
    samples = []
    for entry in report["runs"]:
        if entry["row"] is None:  # a run with no events, or a config record
            continue
        run = runs[entry["input"]]
        messages = inspect_messages(chat[entry["row"]])
        first_user = next((m.content for m in messages if isinstance(m, ChatMessageUser)), "")
        samples.append(EvalSample(
            id=run.id, epoch=1, input=first_user, target="", messages=messages,
            metadata={"run_id": run.id, "dataset": run.source.get("dataset"), "outcome": run.outcome}))
    spec = EvalSpec(created=datetime.now(timezone.utc).isoformat(), task="agentpile_runs", model="none/recorded-runs",
                    config=EvalConfig(), dataset=EvalDataset(name="agentpile runs", samples=len(samples)))
    write_eval_log(EvalLog(eval=spec, status="success", samples=samples), str(path))
    return len(samples)


def check_viewer(log_dir, counts):
    """Start `inspect view` on a free local port, fetch its page, log listing and the log itself, stop it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "inspect_ai", "view", "start", "--log-dir", str(log_dir), "--host", "127.0.0.1",
           "--port", str(port)]
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}" + urllib.parse.quote(path, safe=":/"), timeout=10) as r:
            return r.read().decode("utf-8")

    try:
        for _ in range(120):
            assert server.poll() is None, "inspect view exited early"
            try:
                files = [f["name"] for f in json.loads(get("/api/logs"))["files"]]
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise AssertionError("inspect view did not answer within 60 s")
        page = get("/")
        assert "<html" in page.lower() and "inspect" in page.lower(), "unexpected viewer page"
        assert len(files) == 1 and files[0].endswith("/" + LOG), files
        served = json.loads(get("/api/logs/" + files[0]))  # the log as the viewer's browser client receives it
        assert {s["id"]: len(s["messages"]) for s in served["samples"]} == counts, "the viewer serves different samples"
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def check(runs, path, viewer=True):
    """Read the log back with Inspect and compare each sample with the chat row it came from."""
    report = {}
    chat = ap.to_chat(runs, arguments="dict", report=report)
    pairs = [(runs[e["input"]], chat[e["row"]]) for e in report["runs"] if e["row"] is not None]
    row_of = {run.id: (run, row) for run, row in pairs}
    assert len(row_of) == len(pairs), "two runs share an id"
    log = read_eval_log(str(path))
    assert log.status == "success" and log.eval.dataset.samples == len(pairs), log.status
    assert sorted(s.id for s in log.samples) == sorted(row_of), "sample ids changed"  # Inspect reads them sorted
    for back in log.samples:
        run, row = row_of[back.id]
        assert not any("name" in m for m in row["messages"]), "a multi-agent run needs a perspective"
        # per message: role, text, tool calls (id, name, arguments), id of the call it answers
        signature = [(m.role, m.text, [(c.id, c.function, c.arguments) for c in getattr(m, "tool_calls", None) or []],
                      getattr(m, "tool_call_id", None)) for m in back.messages]
        assert signature == [(m["role"], m["content"] or "", [(c["id"], c["function"]["name"], c["function"]["arguments"])
                                                              for c in m.get("tool_calls", [])], m.get("tool_call_id"))
                             for m in row["messages"]], back.id
        assert back.metadata == {"run_id": run.id, "dataset": run.source.get("dataset"), "outcome": run.outcome}, back.id
    if viewer:
        check_viewer(path.parent, {run.id: len(row["messages"]) for run, row in pairs})


def load_runs(sources, limit):
    """The first `limit` runs of each source; a dataset name takes limit= and downloads only that part, a file path does not."""
    return [run for source in sources
            for run in islice(ap.load(source, **({"limit": limit} if source in ap.datasets.names() else {})), limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="*", help="a dataset name or a trace file (default: the bundled fixtures, with the checks)")
    parser.add_argument("--limit", type=int, default=20, help="runs per source (default: 20)")
    parser.add_argument("--out", type=Path, help="directory for the .eval log (default: a new temporary one)")
    parser.add_argument("--no-viewer", action="store_true", help="skip starting `inspect view` in the checks")
    args = parser.parse_args()
    if args.source:
        runs = load_runs(args.source, args.limit)
    else:
        runs = [run for name in DATASETS for run in ap.load(name, root=FIXTURES / name)]
    log_dir = args.out or Path(tempfile.mkdtemp(prefix="agentpile-inspect-"))
    log_dir.mkdir(parents=True, exist_ok=True)
    count = write_log(runs, log_dir / LOG)
    print(f"wrote {count} samples from {len(runs)} runs to {log_dir / LOG}")
    print(f"open with: inspect view start --log-dir {log_dir}")
    if not args.source:
        check(runs, log_dir / LOG, viewer=not args.no_viewer)
        print("all checks passed")


if __name__ == "__main__":
    main()
