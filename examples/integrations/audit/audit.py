"""Audit agent runs with Invariant Guardrails rules, on the message lists that ap.to_chat builds.

Invariant's LocalPolicy checks rules on an OpenAI-style message list, locally: no model, no network, no key.
ap.to_chat(runs, arguments="dict", report=report) gives that list for a run of any of the 13 datasets, and the report
maps each message to the run and to the events of Run.events(), so a finding names the exact event.

One step is needed. to_chat puts each tool result right after the assistant turn that called it, so a result that the
source recorded later (a delegated call returning after its helper agent finished) moves earlier. Order rules need the
source chronology, so the script first sorts each row back into run.events() order with the report.

Rules: R1 and R1x are data-flow rules (a tool output that holds a recorded injection string, then a sensitive call or
an assistant turn); R2 is an ordering rule (a call issued after the result of another call). On the fixtures R1 stays
quiet, since no attack run there reaches a sensitive call. Checks: every event lands in one message; on TRAIL's
delegating run (aaa1) the sort restores source order; the rules give the same findings on the sorted rows as on message
lists built from run.events() alone.

    pip install -e ".[hf]" -r examples/integrations/audit/requirements.txt
    python examples/integrations/audit/audit.py        # reads tests/corpora/fixtures; offline, takes seconds
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("INVARIANT_MAX_ITERATIONS", "100000")  # two-variable rules on long runs exceed the default of 100
from invariant.analyzer import LocalPolicy  # LocalPolicy runs here; Policy is the remote one unless LOCAL_POLICY=1

import agentpile as ap

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
SENSITIVE = ["send_money", "schedule_transaction", "update_scheduled_transaction", "update_password", "send_email",
             "send_direct_message", "send_channel_message"]  # sinks that an injected instruction aims at
POLICY = LocalPolicy.from_string('''
raise "R1 injected output, then a sensitive call" if:
    (out: ToolOutput) -> (call: ToolCall)
    call.function.name in input.sensitive
    any([inj in out.content for inj in input.injections])

raise "R1x injected output, then an assistant turn" if:
    (out: ToolOutput) -> (msg: Message)
    msg.role == "assistant"
    any([inj in out.content for inj in input.injections])

raise "R2 call issued after the result of another call" if:
    (out: ToolOutput) -> (call: ToolCall)
    out.tool_call_id != call.id
''')


def check(ok, what):
    if not ok:
        sys.exit("CHECK FAILED: " + what)


def sort_row(row, items):
    """Messages and report items in run.events() order: by each message's first event position, instructions first."""
    order = sorted(range(len(items)), key=lambda j: items[j]["events"][0]["position"] if items[j].get("events") else -1)
    return [row["messages"][j] for j in order], [items[j] for j in order]


def source_trace(run):
    """A message list built from run.events() alone: one message per event, in source order, plus each event's address."""
    msgs, addrs, calls = [], [], set()
    text = lambda c: c if isinstance(c, str) else "" if c is None else json.dumps(c, default=str)
    for e in run.events():
        if e.kind == "call":
            call = {"id": e.call_id, "type": "function", "function": {
                "name": e.name, "arguments": e.content if isinstance(e.content, dict) else {"_raw": e.content}}}
            msgs.append({"role": "assistant", "content": None, "tool_calls": [call]})
            calls.add(e.call_id)
        elif e.kind == "result" and e.call_id in calls:
            msgs.append({"role": "tool", "tool_call_id": e.call_id, "content": text(e.content)})
        else:  # a message, or a result whose call the source never recorded (a user turn, as in to_chat)
            msgs.append({"role": {"agent": "assistant", "tool": "user"}.get(e.role, e.role), "content": text(e.content)})
        addrs.append(e.address)
    return msgs, addrs


def findings(msgs, where, run):
    """Apply the rules. A finding is (rule, spots); a spot is (step, call, result) of an event, from `where(i, call_id)`."""
    params = {"injections": list((run.dataset.get("injections") or {}).values()), "sensitive": SENSITIVE}
    found = []
    for err in POLICY.analyze(msgs, **params).errors:
        spots = []
        for rng in err.ranges:  # "messages.5" or "messages.4.tool_calls.0"
            i, k = re.match(r"messages\.(\d+)(?:\.tool_calls\.(\d+))?", rng.to_address()).groups()
            a = where(int(i), None if k is None else msgs[int(i)]["tool_calls"][int(k)]["id"])
            spots.append((a.get("step"), a.get("call"), a.get("result")))
        found.append((err.args[0].split()[0], tuple(spots)))
    return sorted(found, key=str)


def in_report(items):
    def where(i, call_id):  # the event of message i that carries call_id, else the first event of the message
        events = [e["address"] for e in items[i].get("events", [])] or [items[i]["instructions"]]
        return next((a for a in events if call_id and a.get("call") == call_id), events[0])
    return where


def index_of(msgs, kind, call_id):
    """Index of the assistant message that issues call_id (kind "call") or of the tool message that answers it."""
    return next(i for i, m in enumerate(msgs) if (kind == "call" and call_id in [c["id"] for c in m.get("tool_calls", [])])
                or (kind == "result" and m.get("tool_call_id") == call_id))


def main():
    done, moved, n_events = {}, 0, 0
    for name in ("agentdojo", "trail"):
        runs, report = list(ap.load(name, root=FIXTURES / name)), {}
        rows = ap.to_chat(runs, arguments="dict", report=report)
        for entry in report["runs"]:
            if entry["row"] is None:  # a placeholder fixture run: no events, so no row
                print(f"{entry['id']}: skipped ({entry['skipped']})")
                continue
            run, row, rid = runs[entry["input"]], rows[entry["row"]], entry["id"]
            msgs, items = sort_row(row, entry["messages"])
            src_msgs, addrs = source_trace(run)
            check(sorted(e["position"] for it in items for e in it.get("events", [])) == list(range(len(addrs))),
                  f"{rid}: each event is in exactly one message")
            d = done[rid] = dict(run=run, row=row, msgs=msgs, raw=findings(row["messages"], in_report(entry["messages"]), run),
                                 sorted=findings(msgs, in_report(items), run), source=findings(src_msgs, lambda i, c: addrs[i], run))
            check(d["sorted"] == d["source"], f"{rid}: findings on the sorted row equal findings on the source-order list")
            moved, n_events = moved + entry["counts"]["results_moved"], n_events + len(addrs)
    print(f"{len(done)} rows, {n_events} events, tool results moved by to_chat: {moved}\n\nfindings on the sorted rows:")
    step = lambda s: f"step {s[0]}" + (f" call {s[1]}" if s[1] else "") + (f" result {s[2]}" if s[2] is not None else "")
    for rid, d in done.items():
        for rule, spots in d["sorted"]:
            print(f"  {rule:3} {rid}: " + " -> ".join(map(step, spots)))
    totals = Counter(rule for d in done.values() for rule, _ in d["sorted"])
    print("  totals: " + ", ".join(f"{rule} {totals[rule]}" for rule in ("R1", "R1x", "R2")))

    rid = next(r for r in done if r.endswith("aaa1"))
    d, events = done[rid], list(done[rid]["run"].events())
    at = {(e.kind, e.call_id): i for i, e in enumerate(events)}
    actor = {e.call_id: e.actor for e in events if e.kind == "call"}
    check(actor["call_sub1"] == actor["call_sub2"] != actor["call_1"], "call_sub1 and call_sub2 are the helper's calls")
    check(at["call", "call_sub1"] < at["call", "call_sub2"] < at["result", "call_1"], "run.events(): helper calls first")
    place = lambda ms: [index_of(ms, "call", "call_sub1"), index_of(ms, "call", "call_sub2"), index_of(ms, "result", "call_1")]
    before, after = place(d["row"]["messages"]), place(d["msgs"])
    check(before[2] < before[0], "to_chat alone puts the result of call_1 before the helper's calls")
    check(after[0] < after[1] < after[2], "after the sort the result of call_1 comes after call_sub1 and call_sub2")
    check(d["raw"] != d["sorted"] == d["source"], "ordering findings: unsorted row differs, sorted row equals source order")
    print(f"\n{rid}: the result of call_1 moves from message {before[2]} to {after[2]}, after the helper's calls "
          f"at {after[0]} and {after[1]};\n  findings: {len(d['raw'])} unsorted, {len(d['sorted'])} sorted, "
          f"{len(d['source'])} from run.events()\nall checks passed")


if __name__ == "__main__":
    main()
