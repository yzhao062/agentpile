"""Read agentpile runs in Inspect AI's log viewer: three tool-use datasets become one Inspect log, one sample per run.

tau-bench, tau2-bench and AgentDojo store a conversation in three different shapes. ap.to_chat gives each run the
same OpenAI-style message list, this script maps those messages to Inspect's chat message types (system, user,
assistant with tool calls, tool), and Inspect's own writer saves them as an .eval log with the run id, dataset and
outcome as sample metadata. The script reads the log back with read_eval_log and compares every sample with its chat
row (message count, roles, text, tool calls). It then starts `inspect view` on a free local port, fetches the page,
the log listing and the log, and stops it. No network call beyond that local server, API key or model is involved.

    pip install -e ".[hf]" -r examples/integrations/view-inspect/requirements.txt
    python examples/integrations/view-inspect/view_inspect.py [LOG_DIR]    # default: a new temporary directory
    inspect view start --log-dir LOG_DIR                                   # then browse the log yourself

The script reads the small fixtures in tests/corpora/fixtures from the checkout. --no-serve skips the viewer check.
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
from pathlib import Path

import agentpile as ap
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall

DATASETS = ("tau_bench", "tau2_bench", "agentdojo")
FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"


def inspect_messages(row):
    """One chat row (ap.to_chat with arguments="dict") as Inspect chat messages."""
    messages, called = [], {}
    for m in row["messages"]:
        assert "name" not in m, "a multi-agent run needs a perspective; these datasets have one agent"
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


def build_samples():
    """One EvalSample per run that has a chat row, the chat rows, and a count line per dataset."""
    samples, rows, lines = [], [], []
    for name in DATASETS:
        runs = list(ap.load(name, root=FIXTURES / name))
        report = {}
        chat = ap.to_chat(runs, arguments="dict", report=report)
        for entry in report["runs"]:
            if entry["row"] is None:  # a placeholder run with no events, or a config record
                continue
            run, row = runs[entry["input"]], chat[entry["row"]]
            messages = inspect_messages(row)
            first_user = next((m.content for m in messages if isinstance(m, ChatMessageUser)), "")
            samples.append(EvalSample(
                id=run.id, epoch=1, input=first_user, target="", messages=messages,
                metadata={"run_id": run.id, "dataset": run.source["dataset"], "outcome": run.outcome}))
            rows.append(row)
        lines.append(f"{name}: {len(runs)} runs, {len(chat)} chat rows ({len(runs) - len(chat)} without events skipped)")
    return samples, rows, lines


def write_log(samples, log_dir):
    spec = EvalSpec(created=datetime.now(timezone.utc).isoformat(), task="agentpile_runs", model="none/recorded-runs",
                    config=EvalConfig(), dataset=EvalDataset(name="agentpile: " + ", ".join(DATASETS), samples=len(samples)))
    path = Path(log_dir) / "agentpile_runs.eval"
    write_eval_log(EvalLog(eval=spec, status="success", samples=samples), str(path))
    return path


def check_log(path, samples, rows):
    """Read the log back with Inspect and compare each sample with the chat row it came from."""
    log = read_eval_log(str(path))
    assert log.status == "success" and log.eval.dataset.samples == len(rows), log.status
    row_of = {s.id: (s, row) for s, row in zip(samples, rows)}
    assert len(row_of) == len(samples), "two runs share an id"
    assert sorted(s.id for s in log.samples) == sorted(row_of), "sample ids changed"  # Inspect reads them sorted
    n_messages = n_calls = 0
    for back in log.samples:
        sent, row = row_of[back.id]
        # per message: role, text, tool calls (id, name, arguments), id of the call it answers
        signature = [(m.role, m.text, [(c.id, c.function, c.arguments) for c in getattr(m, "tool_calls", None) or []],
                      getattr(m, "tool_call_id", None)) for m in back.messages]
        assert signature == [(m["role"], m["content"] or "", [(c["id"], c["function"]["name"], c["function"]["arguments"])
                                                              for c in m.get("tool_calls", [])], m.get("tool_call_id"))
                             for m in row["messages"]], back.id
        assert back.metadata == sent.metadata and back.metadata["run_id"] == back.id, back.id
        n_messages, n_calls = n_messages + len(signature), n_calls + sum(len(s[2]) for s in signature)
    return f"read back {len(log.samples)} samples, {n_messages} messages, {n_calls} tool calls: all match their chat rows"


def check_viewer(log_dir, samples, rows):
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
        assert len(files) == 1 and files[0].endswith("/agentpile_runs.eval"), files
        served = json.loads(get("/api/logs/" + files[0]))  # the log as the viewer's browser client receives it
        assert {s["id"]: len(s["messages"]) for s in served["samples"]} == \
            {s.id: len(row["messages"]) for s, row in zip(samples, rows)}, "the viewer serves different samples"
        return (f"inspect view on port {port}: page of {len(page)} characters, listing of {len(files)} log, "
                f"{len(served['samples'])} samples served with the same message counts")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("log_dir", nargs="?", help="directory for the .eval log (default: a new temporary directory)")
    parser.add_argument("--no-serve", action="store_true", help="skip the `inspect view` check")
    args = parser.parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else Path(tempfile.mkdtemp(prefix="agentpile-inspect-"))
    log_dir.mkdir(parents=True, exist_ok=True)

    samples, rows, lines = build_samples()
    assert samples, "no run gave a chat row"
    path = write_log(samples, log_dir)
    print("\n".join(lines))
    print(f"wrote {len(samples)} samples to {path}")
    print(check_log(path, samples, rows))
    if not args.no_serve:
        print(check_viewer(log_dir, samples, rows))
    print(f"all checks passed; browse with: inspect view start --log-dir {log_dir}")


if __name__ == "__main__":
    main()
