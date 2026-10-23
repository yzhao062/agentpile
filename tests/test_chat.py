"""``ap.to_chat`` and the ``chat-jsonl`` writer: one test per rule of the conversion.

Dataset fixtures give the real cases (a result TRAIL records after other turns, tau2-bench's simulated user, the call
id SWE-Gym reuses, a group chat in Who&When); small hand-built records give each rule a case of its own.
"""

import json
from pathlib import Path

import pytest

import agentpile as ap

FIX = Path(__file__).parent / "fixtures"
CORPORA = Path(__file__).parent / "corpora" / "fixtures"

QUICK_START = [
    {"role": "user", "content": "Is it raining in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "light rain, 14 C"},
    {"role": "assistant", "content": "Yes, light rain."},
]

# A part ATIF cannot hold, a step with no message, calls with unparsed, absent and null arguments, a call with no
# id, and a result with no content and one with null content.
STAND_INS = [
    {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "u"}}]},
    {"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "not json"}},
        {"type": "function", "function": {"name": "b"}},
        {"id": "c3", "type": "function", "function": {"name": "c", "arguments": None}},
    ]},
    {"role": "tool", "tool_call_id": "c1"},
    {"role": "tool", "tool_call_id": "c3", "content": None},
    {"role": "assistant", "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "d", "arguments": "{}"}}]},
]


def corpus(name):
    """A dataset fixture; a test skips when its reader needs an extra that is not installed (pyarrow for Parquet)."""
    try:
        return list(ap.load(name, root=str(CORPORA / name)))
    except ImportError as exc:
        pytest.skip(str(exc))


def trail(suffix):
    return next(run for run in corpus("trail") if run.id.endswith(suffix))


def quick_start():
    return next(ap.load(QUICK_START))


def config():
    return next(ap.load(FIX / "config_mcp.json"))


def step(sid, source, message=None, actor=None, calls=(), results=(), **ext):
    """An ATIF step; no ``message`` means the source had none."""
    out = {"step_id": sid, "source": source}
    x = dict(ext)
    if actor is not None:
        x["actor"] = actor
    if message is None:
        x.setdefault("message_state", "missing")
    else:
        out["message"] = message
    if calls:
        out["tool_calls"] = [{"tool_call_id": cid, "function_name": name, "arguments": args} for cid, name, args in calls]
    if results:
        out["observation"] = {"results": [{"source_call_id": cid, "content": content} for cid, content in results]}
    out["extra"] = {"agentpile": x}
    return out


def traj(steps, agents=(), tools=None, defs=None):
    agent = {"name": "a", "version": "1"}
    if defs is not None:
        agent["tool_definitions"] = defs
    return ap.Run({"schema_version": "ATIF-v1.7", "session_id": "t", "agent": agent, "steps": steps,
                   "extra": {"agentpile": {"agents": list(agents), "tools": tools or {}}}})


def every_fixture():
    runs = []
    for path in sorted(FIX.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".jsonl") and path.name not in ("versions.json", "not_a_trace.json") \
                and "grade" not in path.relative_to(FIX).parts:
            runs += list(ap.load(path))
    for folder in sorted(p for p in CORPORA.iterdir() if p.is_dir()):
        try:
            runs += list(ap.load(folder.name, root=str(folder)))
        except ImportError:  # a Parquet fixture without pyarrow: the other fixtures still run
            continue
    return runs


def shown(run, event, args):
    """Whether a chat row's arguments object is what the record holds for this call: read from the addressed ATIF
    call and its extension, not from the code under test."""
    step = next(s for s in run.steps if s["step_id"] == event.address["step"])
    call = next(c for c in step["tool_calls"] if c["tool_call_id"] == event.address["call"])
    cx = ((((step.get("extra") or {}).get("agentpile") or {}).get("calls") or {}).get(event.call_id)) or {}
    if cx.get("arguments_state") == "missing" or "arguments_stand_in" in cx:
        return args == {}
    if "arguments_raw" in cx and cx["arguments_raw"] is None:
        return args == {"_raw": None}
    if cx.get("arguments_state") == "unparsed":
        return args == {"_raw": cx.get("arguments_raw")}
    parsed = call.get("arguments")
    return args == (parsed if isinstance(parsed, dict) else {} if parsed is None else {"_raw": parsed})


def aligned(runs, rows, report, perspective=None):
    """An oracle for the report, independent of how to_chat builds it: each entry names its run, each message's
    events are the ones it was made from (kind, role, agent, text, call id, name and arguments), and every event of
    ``run.events()`` is used once."""
    assert [entry["input"] for entry in report["runs"]] == list(range(len(runs)))
    assert [entry["row"] for entry in report["runs"] if entry["row"] is not None] == list(range(len(rows)))
    for entry in report["runs"]:
        run = runs[entry["input"]]
        assert entry["id"] == run.id and entry["source"] == run.source
        if entry["row"] is None:
            assert entry["skipped"] and entry["messages"] == []
            continue
        row, events = rows[entry["row"]], list(run.events())
        simulators = {a.get("id") for a in run.agents if a.get("name") == "user_simulator"}
        named = any("name" in m for m in row["messages"])
        assert len(entry["messages"]) == len(row["messages"])
        used = []
        for msg, item in zip(row["messages"], entry["messages"]):
            if "instructions" in item:
                assert msg["role"] == "system" and item["instructions"]["trajectory"] == run.id
                continue
            for ev in item["events"]:
                assert events[ev["position"]].address == ev["address"]
                used.append(ev["position"])
            src = [events[ev["position"]] for ev in item["events"]]
            first = src[0]
            text = first.content if isinstance(first.content, str) else None
            content = msg["content"]
            if msg["role"] == "assistant":
                calls = [e for e in src if e.kind == "call"]
                assert [e.kind for e in src] in (["message"] + ["call"] * len(calls), ["call"] * len(calls))
                assert all(e.role == "agent" and e.actor not in simulators for e in src)
                assert all(e.actor == msg.get("name") for e in src) if named or "name" in msg else True
                assert perspective is None or all(e.actor in (perspective, None) for e in src)
                for e, call in zip(calls, msg.get("tool_calls", [])):
                    args = call["function"]["arguments"]
                    assert (e.call_id, e.name) == (call["id"], call["function"]["name"])
                    assert shown(run, e, json.loads(args) if isinstance(args, str) else args)
                assert len(calls) == len(msg.get("tool_calls", []))
                assert text is None or first.kind == "call" or content == text
                continue
            assert len(src) == 1 and (msg["role"] == "system") == (first.role == "system")
            if msg["role"] == "tool":
                assert first.kind == "result" and first.call_id == msg["tool_call_id"]
                assert text is None or content == text
                continue
            prefixed = first.role == "agent" and not (first.kind == "message" and first.actor in simulators)
            if first.kind == "result":  # an observation with no call, or another agent's result
                prefixed = content.startswith("[") and " tool result" in content
            if prefixed:
                assert first.role in ("agent", "tool") and content.startswith("[%s] " % (first.actor or "agent"))
                assert perspective is None or first.actor != perspective or first.kind == "result"
            if first.kind == "call":
                label = (first.actor or "agent") if first.role == "agent" else first.role
                prefix = "[%s] tool call%s " % (label, " " + first.name if first.name else "")
                assert content.startswith(prefix) and shown(run, first, json.loads(content[len(prefix):]))
            elif text is not None:  # as it is, or after its "[id] " or "[id] tool result <name>: " prefix
                assert content == text if not prefixed else content.endswith(" " + text)
        assert sorted(used) == list(range(len(events)))


def test_the_alignment_check_rejects_a_report_with_swapped_sources():
    """The oracle itself: equal text from different roles, agents or call arguments must not pass for each other."""
    def swapped(run, i, j, **kw):
        report = {}
        rows = ap.to_chat(run, report=report, **kw)
        aligned([run], rows, report, perspective=kw.get("perspective"))
        positions = [[ev["position"] for ev in item["events"]] for item in report["runs"][0]["messages"]]
        items = report["runs"][0]["messages"]
        items[i], items[j] = items[j], items[i]
        with pytest.raises(AssertionError):
            aligned([run], rows, report, perspective=kw.get("perspective"))
        return positions

    said = traj([step(1, "user", "OK"), step(2, "agent", "OK", actor="x")], [{"id": "x"}])
    assert swapped(said, 0, 1) == [[0], [1]]
    pair = traj([step(1, "agent", "OK", actor="x"), step(2, "agent", "OK", actor="y")], [{"id": "x"}, {"id": "y"}])
    assert swapped(pair, 0, 1) == [[0], [1]]
    assert swapped(pair, 0, 1, perspective="y") == [[0], [1]]
    reuse = traj([step(1, "agent", calls=[("a", "f", {"n": 1})]), step(2, "agent", calls=[("a", "f", {"n": 2})])])
    assert swapped(reuse, 0, 1) == [[0], [1]]
    assert swapped(reuse, 0, 1, arguments="dict") == [[0], [1]]
    # action strings the reader parsed and also kept as text (agent_reward_bench), as y's turns and projected for x
    actions = traj([step(1, "agent", "go", actor="x"), step(2, "agent", actor="y", calls=[("a", "click", {"arg0": "9"})]),
                    step(3, "agent", actor="y", calls=[("a", "click", {"arg0": "10"})])], [{"id": "x"}, {"id": "y"}])
    for st, text in zip(actions.record["steps"][1:], ("click('9')", "click('10')")):
        st["extra"]["agentpile"]["calls"] = {"a": {"arguments_raw": text}}
    for kw in ({}, {"arguments": "dict"}, {"perspective": "x"}):
        assert swapped(actions, 1, 2, **kw) == [[0], [1], [2]]

    def altered(run, change, **kw):
        report = {}
        rows = ap.to_chat(run, report=report, **kw)
        aligned([run], rows, report)
        change(rows[0]["messages"][1]["tool_calls"][0]["function"])
        with pytest.raises(AssertionError):
            aligned([run], rows, report)

    altered(next(ap.load(STAND_INS)), lambda fn: fn.update(arguments="{}"))  # was {"_raw": "not json"}
    altered(actions, lambda fn: fn.update(arguments={}), arguments="dict")  # was {"arg0": "9"}


def test_openai_messages_come_back_as_they_were():
    assert ap.to_chat(quick_start()) == [{"messages": QUICK_START}]
    assert "to_chat" in ap.__all__ and "chat-jsonl" in ap.writers


def test_list_jsonl_and_the_writer_agree(tmp_path):
    runs = every_fixture()
    listed, written = {}, {}
    rows = ap.to_chat(runs, report=listed)
    assert ap.to_chat(iter(runs), tmp_path / "all.jsonl", report=written) is None
    lines = (tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows and listed == written
    assert all(set(row) <= {"messages", "tools"} for row in rows)
    aligned(runs, rows, listed)
    one = quick_start()
    ap.writers.get("chat-jsonl").write(one, tmp_path / "one.jsonl")
    assert (tmp_path / "one.jsonl").read_text(encoding="utf-8") == json.dumps(ap.to_chat(one)[0], ensure_ascii=False) + "\n"


def test_configs_and_trajectories_without_events_give_no_row():
    empty = next(run for run in corpus("tau_bench") if not list(run.events()))
    with pytest.raises(ValueError, match="config record"):
        ap.to_chat(config())
    with pytest.raises(ValueError, match="no events"):
        ap.to_chat(empty)
    report = {"stale": 1}
    rows = ap.to_chat([config(), quick_start(), empty, quick_start()], report=report)
    assert list(report) == ["runs"] and len(rows) == 2
    assert [(e["input"], e["row"], e["skipped"]) for e in report["runs"]] == [
        (0, None, "config"), (1, 0, None), (2, None, "no events"), (3, 1, None)]


def test_a_run_without_an_assistant_turn_stays_a_transcript():
    report = {}
    assert ap.to_chat(traj([step(1, "user", "hi")]), report=report) == [{"messages": [{"role": "user", "content": "hi"}]}]
    assert report["runs"][0]["assistant_turns"] == 0
    report = {}
    rows = ap.to_chat(corpus("swe_agent_nebius"), report=report)
    quiet = [e for e in report["runs"] if e["row"] is not None and e["assistant_turns"] == 0]
    assert len(quiet) == 1 and not any(m["role"] == "assistant" for m in rows[quiet[0]["row"]]["messages"])


def test_system_steps_keep_their_place_and_roster_instructions_fill_in_for_none():
    solo = [{"id": "solo", "instructions": "Be brief."}]
    report = {}
    run = traj([step(1, "user", "hi"), step(2, "agent", "ok", actor="solo")], solo)
    assert ap.to_chat(run, report=report)[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert report["runs"][0]["messages"][0] == {"instructions": {"trajectory": "t", "agent": "solo"}}
    with_system = traj([step(1, "system", "S"), step(2, "user", "hi")], solo)
    assert [m["content"] for m in ap.to_chat(with_system)[0]["messages"]] == ["S", "hi"]
    two = [{"id": "x", "instructions": "X rules"}, {"id": "y", "instructions": "Y rules"}]
    pair = traj([step(1, "user", "hi"), step(2, "agent", "a", actor="x"), step(3, "agent", "b", actor="y")], two)
    assert ap.to_chat(pair)[0]["messages"][0] == {"role": "user", "content": "hi"}
    assert ap.to_chat(pair, perspective="y")[0]["messages"][0] == {"role": "system", "content": "Y rules"}
    # a roster that lists one of the two agents who speak: still several agents, so no instructions without a perspective
    partial = traj([step(1, "agent", "a", actor="x"), step(2, "agent", "b", actor="y")], two[:1])
    assert ap.to_chat(partial)[0]["messages"][0] == {"role": "assistant", "name": "x", "content": "a"}
    assert ap.to_chat(partial, perspective="x")[0]["messages"][0] == {"role": "system", "content": "X rules"}
    simulated = traj([step(1, "agent", "ok", actor="solo")], solo + [{"id": "user", "name": "user_simulator"}])
    assert ap.to_chat(simulated)[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    msgs = ap.to_chat(trail("aaa1"))[0]["messages"]
    assert [i for i, m in enumerate(msgs) if m["role"] == "system"] == [4, 10, 17]
    group = ap.to_chat(corpus("who_and_when")[0], perspective="Math_Expert")[0]["messages"]
    assert group[0]["role"] == "system" and group[0]["content"]


def test_tools_become_openai_function_tools_once_each():
    nested = {"type": "function", "function": {"name": "search", "description": "d", "parameters": {"type": "object"}}}
    tools = {
        "t1": {"name": "search", "definition": json.loads(json.dumps(nested))},
        "t2": {"name": "read", "definition": {"name": "read", "description": "r", "inputSchema": {"type": "object"}}},
        "t3": {"name": "edit", "definition": {"name": "edit", "input_schema": {"type": "object"}}},
        "t4": {"name": "search", "definition": {"name": "search", "parameters": {"type": "object", "required": ["q"]}}},
        "t5": {"name": "web", "definition": {"type": "web_search_preview"}},
        "t6": {"name": "bare"},
        "t7": {"name": "", "definition": {"name": "", "parameters": {"type": "object"}}},
    }
    report = {}
    row = ap.to_chat(traj([step(1, "user", "hi")], tools=tools, defs=[nested]), report=report)[0]
    assert row["tools"] == [
        nested,
        {"type": "function", "function": {"name": "read", "description": "r", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "required": ["q"]}}},
    ]
    assert report["runs"][0]["counts"]["tools_skipped"] == 2
    assert "tools" not in ap.to_chat(traj([step(1, "user", "hi")]))[0]
    # strict is kept: a flat and a nested form of one definition are one tool, and two strict settings are two
    strict = {"type": "function", "function": {"name": "s", "parameters": {"type": "object"}, "strict": True}}
    defs = [strict, {"name": "s", "parameters": {"type": "object"}, "strict": True},
            {"name": "s", "parameters": {"type": "object"}, "strict": False}]
    assert ap.to_chat(traj([step(1, "user", "hi")], defs=defs))[0]["tools"] == [
        strict, {"type": "function", "function": {"name": "s", "parameters": {"type": "object"}, "strict": False}}]
    flat = [t for row in ap.to_chat(corpus("trail")) for t in row.get("tools", [])]  # smolagents' flat shape
    assert flat and all(t["type"] == "function" and isinstance(t["function"]["name"], str) for t in flat)


def test_agents_speak_as_assistant_with_their_ids_as_names():
    # a roster of two names both agents, even when only one of them speaks
    two = [{"id": "x"}, {"id": "y"}]
    assert ap.to_chat(traj([step(1, "agent", "a", actor="x")], two))[0]["messages"] == [
        {"role": "assistant", "name": "x", "content": "a"}]
    # an OpenAI assistant named "user" is an agent like any other, not a simulated user
    named_user = next(ap.load([{"role": "assistant", "name": "user", "content": "I am the assistant."}]))
    assert ap.to_chat(named_user)[0]["messages"][0]["role"] == "assistant"
    assert ap.to_chat(named_user, perspective="user") == ap.to_chat(named_user)
    runs = corpus("who_and_when")
    for run, row in zip(runs, ap.to_chat(runs)):
        assert [m.get("name") for m in row["messages"] if m["role"] == "assistant"] == \
            [e.actor for e in run.events() if e.role == "agent"]
    assert not any("name" in m for row in ap.to_chat(corpus("tau_bench")) for m in row["messages"])


def test_a_perspective_turns_other_agents_into_user_text():
    run = traj([step(1, "user", "go"), step(2, "agent", "plan", actor="x"),
                step(3, "agent", "look", actor="y", calls=[("c1", "search", {"q": 1})], results=[("c1", "found")]),
                step(4, "agent", calls=[("c2", "", {})], actor="y"),
                step(5, "agent", "done", actor="x", calls=[("c3", "write", {})], results=[("c3", "ok")])],
               [{"id": "x"}, {"id": "y"}])
    report = {}
    assert ap.to_chat(run, perspective="x", report=report)[0]["messages"] == [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "plan"},
        {"role": "user", "content": "[y] look"},
        {"role": "user", "content": '[y] tool call search {"q": 1}'},
        {"role": "user", "content": "[y] tool result search: found"},
        {"role": "user", "content": "[y] tool call {}"},
        {"role": "assistant", "content": "done", "tool_calls": [
            {"id": "c3", "type": "function", "function": {"name": "write", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c3", "content": "ok"},
    ]
    assert report["runs"][0]["counts"]["projected"] == 4
    run, report = trail("aaa1"), {}
    msgs = ap.to_chat(run, perspective="a000000000000008", report=report)[0]["messages"]
    events = list(run.events())
    for msg, item in zip(msgs, report["runs"][0]["messages"]):
        sources = [events[ev["position"]] for ev in item["events"]]
        if any(e.actor == "a000000000000003" for e in sources):
            assert set(msg) == {"role", "content"} and msg["role"] == "user"
            assert msg["content"].startswith("[a000000000000003] ")
        elif any(e.actor == "a000000000000008" and e.kind != "result" for e in sources):
            assert msg["role"] == "assistant" and "name" not in msg
    contents = [m["content"] for m in msgs if m["role"] == "user"]
    assert any(c.startswith("[a000000000000003] tool call python_interpreter {") for c in contents)
    assert any(c.startswith("[a000000000000003] tool result python_interpreter: Call id: call_1") for c in contents)
    assert msgs[-1] == {"role": "user", "content": "[agent] FINAL ANSWER: A"}  # an agent step with no actor
    counts = report["runs"][0]["counts"]
    assert counts["projected"] == 8 and counts["results_moved"] == 0
    aligned([config(), run], ap.to_chat([config(), run], perspective="a000000000000008", report=report), report,
            perspective="a000000000000008")


def test_a_perspective_must_name_an_agent_other_than_the_simulator():
    run = corpus("tau2_bench")[0]
    with pytest.raises(ValueError, match="simulated user"):
        ap.to_chat(run, perspective="user")
    with pytest.raises(ValueError, match="nobody"):
        ap.to_chat(run, perspective="nobody")
    assert ap.to_chat(run, perspective="assistant") == ap.to_chat(run)


def test_the_simulated_user_is_user_context_in_both_modes():
    roster = [{"id": "assistant"}, {"id": "user", "name": "user_simulator"}]
    sim = traj([step(1, "agent", "Hi!", actor="assistant"), step(2, "agent", "No data.", actor="user"),
                step(3, "agent", actor="user", calls=[("u1", "toggle", {"on": False})], results=[("u1", "off")])], roster)
    for perspective in (None, "assistant"):
        assert ap.to_chat(sim, perspective=perspective)[0]["messages"] == [
            {"role": "assistant", "content": "Hi!"}, {"role": "user", "content": "No data."},
            {"role": "user", "content": '[user] tool call toggle {"on": false}'},
            {"role": "user", "content": "[user] tool result toggle: off"}]
    run = next(r for r in corpus("tau2_bench") if any(e.kind == "call" and e.actor == "user" for e in r.events()))
    events = list(run.events())
    for perspective in (None, "assistant"):
        report = {}
        msgs = ap.to_chat(run, perspective=perspective, report=report)[0]["messages"]
        for msg, item in zip(msgs, report["runs"][0]["messages"]):
            first = events[item["events"][0]["position"]]
            if first.actor == "user" and first.role != "user":
                assert msg["role"] == "user"
                if first.kind == "message":
                    assert msg["content"] == (first.content or "")
                else:
                    assert msg["content"].startswith("[user] tool ")


def test_a_result_follows_the_turn_of_its_call_and_a_move_is_reported():
    late = traj([step(1, "agent", calls=[("a", "f", {})], results=[("a", "late")], source_pos=0, result_pos=[2]),
                 step(2, "user", "meanwhile", source_pos=1)])
    report = {}
    assert [m["role"] for m in ap.to_chat(late, report=report)[0]["messages"]] == ["assistant", "tool", "user"]
    assert report["runs"][0]["counts"]["results_moved"] == 1
    assert [item["events"][0]["position"] for item in report["runs"][0]["messages"]] == [0, 2, 1]
    report = {}
    msgs = ap.to_chat(trail("aaa1"), report=report)[0]["messages"]
    i = next(k for k, m in enumerate(msgs) if any(c["id"] == "call_1" for c in m.get("tool_calls", [])))
    assert msgs[i + 1]["role"] == "tool" and msgs[i + 1]["tool_call_id"] == "call_1"
    assert report["runs"][0]["messages"][i + 1]["events"][0]["position"] == 15
    assert report["runs"][0]["counts"]["results_moved"] == 1
    order = sorted(range(len(msgs)), key=lambda k: report["runs"][0]["messages"][k]["events"][0]["position"])
    assert order.index(i + 1) > order.index(i + 2)  # sorting by position puts the result back after the helper's turns


def test_a_result_matches_the_latest_earlier_call_with_its_id():
    twice = traj([step(1, "agent", calls=[("x", "f", {"n": 1})], results=[("x", "first")]),
                  step(2, "agent", calls=[("x", "f", {"n": 2})]),
                  step(3, "agent", results=[("x", "second")])])
    report = {}
    assert [(m["role"], m["content"]) for m in ap.to_chat(twice, report=report)[0]["messages"]] == [
        ("assistant", None), ("tool", "first"), ("assistant", None), ("tool", "second")]
    assert report["runs"][0]["counts"]["results_moved"] == 0 and report["runs"][0]["counts"]["unanswered_calls"] == 0
    unanswered = traj([step(1, "agent", calls=[("x", "f", {"n": 1})]),
                       step(2, "agent", calls=[("x", "f", {"n": 2})], results=[("x", "r")])])
    report = {}
    assert [m["role"] for m in ap.to_chat(unanswered, report=report)[0]["messages"]] == ["assistant", "assistant", "tool"]
    assert report["runs"][0]["counts"]["unanswered_calls"] == 1
    report = {}
    run = corpus("swe_gym_openhands")[0]
    msgs = ap.to_chat(run, report=report)[0]["messages"]
    turns = [k for k, m in enumerate(msgs) if any(c["id"] == "call_A1" for c in m.get("tool_calls", []))]
    assert len(turns) == 2 and all(msgs[k + 1]["tool_call_id"] == "call_A1" for k in turns)
    assert [report["runs"][0]["messages"][k + 1]["events"][0]["position"] for k in turns] == [4, 14]
    aligned([run], [ap.to_chat(run)[0]], report)


def test_results_without_a_call_become_user_turns_and_unanswered_calls_stay():
    report = {}
    run = traj([step(1, "agent", "look", results=[(None, "obs")]), step(2, "agent", calls=[("q", "f", {})])])
    assert ap.to_chat(run, report=report)[0]["messages"] == [
        {"role": "assistant", "content": "look"}, {"role": "user", "content": "obs"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "q", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
    assert report["runs"][0]["counts"]["results_without_call"] == 1
    assert report["runs"][0]["counts"]["unanswered_calls"] == 1
    report = {}
    ap.to_chat(corpus("swe_agent_nebius"), report=report)
    assert sum(e["counts"]["results_without_call"] for e in report["runs"]) == 3


def test_arguments_parsed_raw_null_or_absent():
    report = {}
    msgs = ap.to_chat(next(ap.load(STAND_INS)), report=report)[0]["messages"]
    assert [c["function"]["arguments"] for m in msgs for c in m.get("tool_calls", [])] == [
        '{"_raw": "not json"}', "{}", '{"_raw": null}', "{}"]
    counts = report["runs"][0]["counts"]
    assert (counts["unparsed_arguments"], counts["missing_arguments"], counts["null_arguments"]) == (1, 1, 1)
    msgs = ap.to_chat(next(ap.load(STAND_INS)), arguments="dict")[0]["messages"]
    assert [c["function"]["arguments"] for m in msgs for c in m.get("tool_calls", [])] == [
        {"_raw": "not json"}, {}, {"_raw": None}, {}]
    rebuilt = traj([step(1, "agent", calls=[("a", "f", {})])])
    rebuilt.record["steps"][0]["extra"]["agentpile"]["calls"] = {"a": {"arguments_stand_in": "rule"}}
    report = {}
    assert ap.to_chat(rebuilt, report=report)[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert report["runs"][0]["counts"]["missing_arguments"] == 1
    # an action string that parsed keeps its parsed object; the reader kept the text as well
    action = traj([step(1, "agent", calls=[("a", "click", {"arg0": "9"})])])
    action.record["steps"][0]["extra"]["agentpile"]["calls"] = {"a": {"arguments_raw": "click('9')"}}
    assert ap.to_chat(action, arguments="dict")[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"arg0": "9"}
    with pytest.raises(ValueError, match="arguments"):
        ap.to_chat(quick_start(), arguments="text")
    shown = {c["id"]: c["function"]["arguments"] for row in ap.to_chat(corpus("agent_reward_bench"), arguments="dict")
             for m in row["messages"] for c in m.get("tool_calls", [])}
    assert {"arg0": "7", "arg1": "it's a 'mug'"} in shown.values()
    assert {"_raw": "fill('4', some_variable)"} in shown.values()


def test_content_text_parts_json_null_and_missing():
    report = {}
    row = ap.to_chat(next(ap.load(STAND_INS)), report=report)[0]
    msgs = row["messages"]
    assert msgs[0] == {"role": "user", "content": 'look\n{"type": "image_url", "image_url": {"url": "u"}}'}
    assert [m["content"] for m in msgs if m["role"] == "tool"] == ["", ""]
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == [None, None]
    counts = report["runs"][0]["counts"]
    assert (counts["content_parts"], counts["missing_content"], counts["null_content"]) == (1, 1, 1)
    assert "MISSING" not in json.dumps(row)
    odd = traj([step(1, "agent", "", message_raw={"k": 1}), step(2, "agent", None, message_state="null")])
    report = {}
    assert ap.to_chat(odd, report=report)[0]["messages"] == [
        {"role": "assistant", "content": '{"k": 1}'}, {"role": "assistant", "content": ""}]
    assert (report["runs"][0]["counts"]["content_json"], report["runs"][0]["counts"]["null_content"]) == (1, 1)


def test_subagents_are_runs_of_their_own():
    for name in ("openai_agents_0.24.0_oi_2.5.3_as_tool.json", "pydantic_ai_2.58.0_delegation.json"):
        run = next(ap.load(FIX / "otlp" / name))
        rows = ap.to_chat(run.subagents)
        assert len(rows) == len(run.subagents) == 1 and rows[0]["messages"][-1] == {"role": "assistant", "content": "42"}


def test_dict_rows_share_nothing_with_the_record():
    run = quick_start()
    row = ap.to_chat(run, arguments="dict")[0]
    row["messages"][1]["tool_calls"][0]["function"]["arguments"]["city"] = "Rome"
    assert run.steps[1]["tool_calls"][0]["arguments"] == {"city": "Paris"}


def test_a_failed_write_leaves_the_file_as_it_was(tmp_path):
    dest = tmp_path / "rows.jsonl"
    dest.write_text("old\n", encoding="utf-8")
    fits = traj([step(1, "agent", "a", actor="x")], [{"id": "x"}])
    other = traj([step(1, "agent", "b", actor="y")], [{"id": "y"}])
    with pytest.raises(ValueError, match="perspective"):
        ap.to_chat([fits, other], dest, perspective="x")
    assert dest.read_text(encoding="utf-8") == "old\n" and list(tmp_path.iterdir()) == [dest]


def test_bad_inputs_raise():
    with pytest.raises(TypeError, match="report"):
        ap.to_chat(quick_start(), report=[])
    with pytest.raises(TypeError):
        ap.to_chat([{"messages": []}])
