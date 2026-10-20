"""The resource rules and their extractors (``resource_chain``, ``last_writer``, ``prior_reads``, ``value_flow``)."""

import json
from pathlib import Path

import pytest

import agentpile as ap
from agentpile import deps as D
from agentpile.formats.messages import messages_to_trajectory


def run_of(messages, **kw):
    return ap.Run(messages_to_trajectory(messages, trajectory_id="t", **kw))


def call(cid, name, args=None, raw=None):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": raw if raw is not None else json.dumps(args or {})}}


def asst(*calls, content=""):
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def tool(cid, content):
    msg = {"role": "tool", "content": content}
    if cid is not None:
        msg["tool_call_id"] = cid
    return msg


def edit(cid, command, path):
    return call(cid, "str_replace_editor", {"command": command, "path": path})


USER = {"role": "user", "content": "go"}


def edges(run, rule):
    return [(d["src"]["call"], d["dst"]["call"], d["resource"]) for d in ap.derive_deps(run, rules=[rule])]


# -- entries and registry ---------------------------------------------------------------------

def test_entries_are_full_addresses_with_rule_version_and_extractor():
    run = run_of([USER, asst(edit("c1", "view", "/a.py")), tool("c1", "x"), asst(edit("c2", "view", "/a.py"))])
    (entry,) = ap.derive_deps(run, rules=["resource_chain_v1"])
    assert entry == {"src": {"trajectory": "t", "step": 2, "call": "c1"}, "dst": {"trajectory": "t", "step": 3, "call": "c2"},
                     "rule": "resource_chain", "rule_version": "1", "resource": "file:/a.py",
                     "extractor": "file_editor_paths_v1"}
    old = ap.derive_deps(run, rules=["tool_call_id_v1"])
    assert all(set(d) == {"src", "dst", "rule", "rule_version", "resource"} for d in old)  # the older rules are unchanged


def test_rules_resolve_by_name_and_latest_version():
    assert {"resource_chain_v1", "last_writer_v1", "prior_reads_v1", "value_flow_v1"} <= set(D.RULES)
    run = run_of([USER, asst(edit("c1", "view", "/a.py")), asst(edit("c2", "view", "/a.py"))])
    assert ap.derive_deps(run, rules=["resource_chain"]) == ap.derive_deps(run, rules=["resource_chain_v1"])
    assert ap.derive_deps(run, rules=[D.resource_chain()]) == ap.derive_deps(run, rules=["resource_chain_v1"])
    with pytest.raises(KeyError, match="unknown dependency rule"):
        ap.derive_deps(run, rules=["resource_chain_v2"])


def test_extractors_are_named_and_versioned():
    assert {e.id for e in D.EXTRACTORS.values()} == set(D.EXTRACTORS)
    assert set(D.EXTRACTORS) == {"file_editor_paths_v1", "shell_commands_v1", "web_urls_v1", "db_tool_names_v1",
                                 "world_location_v1"}
    assert D.file_editor_paths_v1.name == "file_editor_paths" and D.file_editor_paths_v1.version == "1"


def test_config_record_gets_no_resource_edges():
    config = next(ap.load(Path(__file__).parent / "fixtures" / "config_mcp.json"))
    assert ap.derive_deps(config, rules=["resource_chain_v1", "last_writer_v1", "prior_reads_v1", "value_flow_v1"]) == []


# -- resource_chain ---------------------------------------------------------------------------

def test_resource_chain_links_each_access_to_the_previous_one_on_the_same_resource():
    run = run_of([
        USER,
        asst(edit("c1", "view", "/a.py")), tool("c1", "x"),
        asst(edit("c2", "view", "/a.py"), edit("c3", "str_replace", "/a.py"), edit("c4", "view", "/b.py")),
        tool("c2", "x"), tool("c3", "x"), tool("c4", "x"),
        asst(call("c5", "execute_bash", {"command": "pytest"})),
        asst(edit("c6", "view", "/a.py")),
    ])
    assert edges(run, "resource_chain_v1") == [("c1", "c2", "file:/a.py"), ("c2", "c3", "file:/a.py"),
                                               ("c3", "c6", "file:/a.py")]  # mode does not gate: read after read links


def test_a_resource_the_data_does_not_give_means_no_edge():
    run = run_of([
        USER,
        asst(call("c1", "str_replace_editor", {"command": "view"})), asst(call("c2", "str_replace_editor", {"command": "view"})),
        asst(call("c3", "execute_bash", {"command": "cat a.py"})), asst(call("c4", "execute_bash", {"command": "cat a.py"})),
        asst(call("c5", "think", {"thought": "x"})), asst(call("c6", "think", {"thought": "x"})),
        asst(call("c7", "str_replace_editor", raw="not json")), asst(call("c8", "str_replace_editor", raw="not json")),
        asst(call("c9", "str_replace_editor", {"command": "view", "path": 7})),
        asst(call("c10", "str_replace_editor", {"command": "view", "path": 7})),
    ])
    assert edges(run, "resource_chain_v1") == []


def test_editor_path_falls_back_to_file_and_paths_are_not_normalized():
    run = run_of([
        USER,
        asst(call("c1", "str_replace_editor", {"command": "view", "path": "", "file": "a.py"})),
        asst(call("c2", "edit_file", {"command": "create", "file": "a.py"})),
        asst(call("c3", "str_replace_based_edit_tool", {"command": "view", "path": "/repo/a.py"})),
    ])
    assert edges(run, "resource_chain_v1") == [("c1", "c2", "file:a.py")]  # "/repo/a.py" is another resource


def test_file_editor_modes():
    run = run_of([USER, asst(*[edit(f"c{i}", cmd, "/a.py") for i, cmd in enumerate(
        ["view", "create", "str_replace", "insert", "append", "write", "undo_edit", "CREATE"])]),
        asst(call("f", "finish"), call("s", "submit"), call("b", "bash"), call("x", "other"))])
    modes = [(a.address["call"], a.mode) for a in D.file_editor_paths_v1(run)]
    assert modes == [("c0", "read"), ("c1", "write"), ("c2", "write"), ("c3", "write"), ("c4", "write"), ("c5", "write"),
                     ("c6", "read"), ("c7", "write"), ("f", "write"), ("s", "write"), ("b", "run"), ("x", None)]


# -- last_writer ------------------------------------------------------------------------------

def test_last_writer_links_to_the_last_write_only():
    run = run_of([USER, asst(edit("c1", "view", "/a.py"), edit("c2", "str_replace", "/a.py"), edit("c3", "view", "/a.py"),
                             edit("c4", "view", "/a.py"), edit("c5", "create", "/a.py"), edit("c6", "view", "/b.py"))])
    assert edges(run, "last_writer_v1") == [("c2", "c3", "file:/a.py"), ("c2", "c4", "file:/a.py"),
                                            ("c2", "c5", "file:/a.py")]  # reads before the first write, and other files, get none


def test_last_writer_on_a_text_world():
    texts = ["Thought: x. Action: look around", "Thought: go. Action: go to kitchen", "Thought: open. Action: open fridge",
             "Thought: jump. Action: teleport to outside", "Thought: take. Action: pick up thermometer", "no action here"]
    msgs = [USER]
    for t in texts:
        msgs += [{"role": "assistant", "content": t}, {"role": "user", "content": "ok"}]
    run = run_of(msgs)
    out = [(d["src"]["step"], d["dst"]["step"]) for d in ap.derive_deps(run, rules=[D.last_writer(D.world_location_v1)])]
    assert out == [(4, 6), (4, 8), (8, 10)]  # a move depends on the previous move; the turn without an action is no access
    assert {d["resource"] for d in ap.derive_deps(run, rules=[D.last_writer(D.world_location_v1)])} == {"world:location"}


# -- prior_reads ------------------------------------------------------------------------------

def test_prior_reads_gives_a_write_every_earlier_read():
    run = run_of([
        USER,
        asst(call("r1", "get_user_details", {"user_id": "u"})), tool("r1", "{}"),
        asst(call("n1", "calculate", {"expression": "1+1"})), tool("n1", "2"),
        asst(call("r2", "get_order_details", {"order_id": "#W1"}), call("w1", "cancel_pending_order", {"order_id": "#W1"})),
        asst(call("r3", "list_all_orders")),
        asst(call("w2", "modify_pending_order_address", {})),
        asst(call("x", "")),  # no name: neither a read nor a write
        asst(call("w3", "transfer_to_human_agents", {})),
    ])
    assert edges(run, "prior_reads_v1") == [("r1", "w1", None), ("r2", "w1", None), ("r1", "w2", None),
                                            ("r2", "w2", None), ("r3", "w2", None)]


def test_prior_reads_skips_accesses_that_have_a_resource():
    def with_resources(run):
        return [D.Access(a.address, "file:/a" if a.tool == "get_a" else None, a.mode, a.tool) for a in D.db_tool_names_v1(run)]

    ex = D.Extractor("custom", "1", with_resources)
    run = run_of([USER, asst(call("r1", "get_a"), call("r2", "get_b"), call("w", "book_x"))])
    out = [(d["src"]["call"], d["dst"]["call"], d["extractor"]) for d in ap.derive_deps(run, rules=[D.prior_reads(ex)])]
    assert out == [("r2", "w", "custom_v1")]  # get_a names a resource, so it is not an unscoped read


# -- value_flow -------------------------------------------------------------------------------

def vf(run):
    return [(d["src"]["call"], d["dst"]["call"], d["resource"]) for d in ap.derive_deps(run, rules=["value_flow_v1"])]


def test_value_flow_follows_an_id_from_a_result_into_a_later_call():
    run = run_of([
        USER,
        asst(call("a", "get_customer", {"phone": "none"})), tool("a", json.dumps({"customer": {"id": "C1001", "lines": ["L1001"]}})),
        asst(call("b", "get_line", {"filters": {"line": "L1001"}})), tool("b", "{}"),
        asst(call("c", "get_line", {"filters": ["x", {"deep": "L1001"}]})),
    ])
    assert vf(run) == [("a", "b", ["value:L1001"]), ("a", "c", ["value:L1001"])]


def test_value_flow_takes_the_latest_producer_and_one_dependency_per_call():
    run = run_of([
        USER,
        asst(call("p1", "get", {})), tool("p1", json.dumps({"a": "A100", "b": "B200"})),
        asst(call("p2", "get", {})), tool("p2", json.dumps({"a": "A100"})),
        asst(call("p3", "get", {})), tool("p3", json.dumps({"c": "C300"})),
        asst(call("q", "use", {"x": "A100", "y": "B200", "z": "C300", "w": "A100"})),
        asst(call("r", "use", {"x": "A100", "y": "B200"})),
    ])
    # q: A100 -> p2, B200 -> p1, C300 -> p3; the latest producer is p3. r: A100 -> p2 beats B200 -> p1.
    assert vf(run) == [("p3", "q", ["value:C300"]), ("p2", "r", ["value:A100"])]


def test_value_flow_resource_lists_every_id_that_points_at_the_chosen_producer():
    run = run_of([
        USER,
        asst(call("p", "get", {})), tool("p", json.dumps({"a": "Z9", "b": "A100", "c": "B200"})),
        asst(call("q", "use", {"x": "B200", "y": "A100", "z": "Z9"})),
    ])
    assert vf(run) == [("p", "q", ["value:A100", "value:B200"])]  # sorted; Z9 is too short to be an id


@pytest.mark.parametrize("text,is_id", [
    ("C1001", True), ("L1002", True), ("a_123", True), ("AB-1234", True), ("abcd1234", True), ("555-123-2002", True),
    ("2024-05-20", True), ("12", False), ("123", True), ("hello", False), ("John Smith 12345", False),
    ("abcde1234", False), ("A1", False), ("", False),
])
def test_the_id_pattern_is_frozen(text, is_id):
    assert bool(D.ID_LIKE.match(text)) is is_id


def test_value_flow_scans_values_not_keys_and_parses_results_as_json():
    run = run_of([
        USER,
        asst(call("a", "get", {})), tool("a", json.dumps({"K1001": "plain"})),  # an id used as a key
        asst(call("b", "use", {"x": "K1001"})),
        asst(call("c", "get", {})), tool("c", "not json: C5005"),  # unparseable result: no ids
        asst(call("d", "use", {"x": "C5005"})),
        asst(call("e", "get", {})), tool("e", json.dumps("S7007")),  # a bare JSON string is scanned as a string
        asst(call("f", "use", {"x": "S7007"})),
        asst(call("g", "use", raw="\"S7007\"")),  # arguments that are not an object give no ids
    ])
    assert vf(run) == [("e", "f", ["value:S7007"])]


def test_a_call_sees_only_results_that_arrived_before_it():
    run = run_of([
        USER,
        asst(call("a", "get", {"id": "N1000"}), call("b", "get", {"id": "N2000"})),
        tool("a", json.dumps({"id": "N2000"})), tool("b", json.dumps({"id": "N1000"})),
    ])
    assert vf(run) == []  # both results come after both calls
    run = run_of([USER, asst(call("a", "get", {})), tool("a", json.dumps({"id": "N1000"})),
                  asst(call("b", "get", {"id": "N1000"}))])
    assert vf(run) == [("a", "b", ["value:N1000"])]


def positional(messages):
    return run_of(messages, pair_by_position=True)


def test_results_are_paired_to_their_own_call_not_the_last_one():
    run = positional([
        USER,
        asst(call("a", "get_customer"), call("b", "get_bills")),
        tool(None, json.dumps({"customer_id": "C1001", "line_ids": ["L1001"]})), tool(None, json.dumps({"bill_id": "B1001"})),
        asst(call("c", "get_line", {"id": "L1001"})),
    ])
    assert vf(run) == [("a", "c", ["value:L1001"])]  # the reader paired the first result with the first call
    by_id = run_of([USER, asst(call("a", "g"), call("b", "g")),
                    tool("b", json.dumps({"bill_id": "B1001"})), tool("a", json.dumps({"customer_id": "C1001"})),
                    asst(call("c", "use", {"id": "C1001"}))])
    assert vf(by_id) == [("a", "c", ["value:C1001"])]  # out-of-order results, paired by id


def test_calls_of_a_promoted_user_message_are_neither_consumers_nor_producers():
    def source_of(msg):
        return "agent" if msg.get("role") == "user" and msg.get("tool_calls") else None

    user_call = {"role": "user", "content": "", "tool_calls": [call("u", "check_device", {"id": "D2002"})]}
    run = run_of([
        USER,
        asst(call("a", "get_customer")), tool("a", json.dumps({"customer_id": "C1001", "device": "D2002"})),
        user_call, tool("u", json.dumps({"device_id": "D2002", "new": "N9009"})),
        asst(call("c", "get_details", {"id": "D2002"})),
        asst(call("d", "get_details", {"id": "N9009"})),
    ], source_of=source_of)
    assert vf(run) == [("a", "c", ["value:D2002"])]  # D2002 still comes from the assistant's own call; N9009 has no producer


def test_unpaired_results_and_unanswered_calls():
    run = positional([
        USER,
        asst(call("a", "get")), tool(None, json.dumps({"id": "A1000"})), tool(None, json.dumps({"id": "A2000"})),  # 2nd answers nothing
        asst(call("b", "use", {"id": "A2000"})),
        asst(call("c", "use", {"id": "A1000"})), asst(call("d", "use", {"id": "A1000"})),  # c has no result
    ])
    assert vf(run) == [("a", "c", ["value:A1000"]), ("a", "d", ["value:A1000"])]
    assert [d["dst"]["call"] for d in ap.derive_deps(run, rules=["value_flow_v1"])] == ["c", "d"]


# -- other extractors ----------------------------------------------------------------------------

def swe_agent_run(*turns):
    msgs = [{"role": {"ai": "assistant"}.get(r, r), "content": t} for r, t in turns]
    return ap.Run(messages_to_trajectory(msgs, trajectory_id="t"))


def block(cmd):
    return f"thought\n```\n{cmd}\n```"


def test_shell_commands_extractor():
    run = swe_agent_run(
        ("system", "s"), ("user", "task\n(Open file: n/a)\n(Current directory: /repo)"),
        ("ai", block("ls -a")), ("user", "files"),
        ("ai", block("open a.py")), ("user", "[File: a.py]\n(Open file: /repo/a.py)\n(Current directory: /repo)"),
        ("ai", block("edit 1:2\nx\nend_of_edit")), ("user", "ok"),
        ("ai", block("search_file \"q\"")), ("user", "ok"),
        ("ai", block("search_file \"q\" b.py")), ("user", "ok"),
        ("ai", block("grep -rn q .")), ("user", "ok"),
        ("ai", block("grep q b.py")), ("user", "ok"),
        ("ai", "no code block at all"), ("user", "ok"),
        ("ai", block("python run.py")), ("user", "ok"),
        ("ai", block("submit")),
    )
    got = [(a.tool, a.resource, a.mode) for a in D.shell_commands_v1(run)]
    assert got == [("ls", None, "read"), ("open", "file:a.py", "read"), ("edit", "file:/repo/a.py", "write"),
                   ("search_file", "file:/repo/a.py", "read"), ("search_file", "file:b.py", "read"), ("grep", None, "read"),
                   ("grep", "file:b.py", "read"), ("python", None, "run"), ("submit", None, "write")]
    assert [d["resource"] for d in ap.derive_deps(run, rules=[D.resource_chain(D.shell_commands_v1)])] == [
        "file:/repo/a.py", "file:b.py"]  # a.py and /repo/a.py are not the same resource: no path is normalized


def swe_agent_reader_run(*turns):
    """The SWE-agent reader's conversion of (role, text) turns: later user turns become results, not steps."""
    raw = {"instance_id": "demo__repo-1", "model_name": "m", "target": False,
           "trajectory": [{"role": r, "text": None if r == "system" else t, "system_prompt": t if r == "system" else None}
                          for r, t in turns]}
    where = {"file": "data/train-00000-of-00012.parquet", "config": "default", "split": "train"}
    return ap.datasets.get("swe_agent_nebius").convert("train-00000-of-00012:0", raw, where)


def footer(path):
    return f"output\n(Open file: {path})\n(Current directory: /repo)\nbash-$"


def test_shell_commands_reads_the_open_file_from_the_results_of_the_reader():
    run = swe_agent_reader_run(
        ("system", "s"), ("user", "task\n" + footer("n/a")),
        ("ai", block("open src/a.py")), ("user", footer("/repo/src/a.py")),
        ("ai", block("scroll_down")), ("user", footer("/repo/src/a.py")),
        ("ai", block("edit 1:1\nx\nend_of_edit")), ("user", footer("n/a")),
        ("ai", block("edit 1:1\nx\nend_of_edit")), ("user", "no footer here"),
        ("ai", block("goto 3")), ("user", "ok"),
    )
    assert [s["source"] for s in run.steps] == ["system", "user"] + ["agent"] * 5  # the footers are not steps
    got = [(a.tool, a.resource, a.mode) for a in D.shell_commands_v1(run)]
    assert got == [("open", "file:src/a.py", "read"), ("scroll_down", "file:/repo/src/a.py", "read"),
                   ("edit", "file:/repo/src/a.py", "write"), ("edit", None, "write"), ("goto", None, "read")]
    # the footer replaces the path the open command was given; "n/a" clears it and a missing footer leaves it
    out = [(d["src"]["step"], d["dst"]["step"], d["resource"])
           for d in ap.derive_deps(run, rules=[D.resource_chain(D.shell_commands_v1)])]
    assert out == [(4, 5, "file:/repo/src/a.py")]  # scroll_down to edit; the open command's own path chains to nothing


def web_run(*steps, last=None):
    """The AgentRewardBench reader's conversion of hand-written records.

    A record is ``(action, url)``; an action of ``None`` is a record whose agent reply holds no action. ``last`` is
    the url of the observation after the last action, a record with no action and no reply.
    """
    recs = []
    for i, (action, url) in enumerate(steps):
        rec = {"num": i, "action": action, "url": url, "reasoning": "r", "axtree": "tree"}
        if action is None:
            rec["reasoning"], rec["chat_messages"] = None, [{"role": "assistant", "content": "unsure"}]
        recs.append(rec)
    if last is not None:
        recs.append({"num": len(recs), "url": last, "axtree": "tree"})
    where = {"file": "cleaned/webarena/A/run/webarena.1.json", "config": "webarena", "split": "full", "task_id": "webarena.1"}
    return ap.datasets.get("agent_reward_bench").convert(
        "webarena/A/webarena.1", {"head": {"goal": "g", "agent": "A"}, "steps": recs, "annotations": []}, where)


def test_web_urls_extractor():
    run = web_run(("goto('http://s/home')", "http://s/home"), ("click('4')", "http://s/home"), ("noop()", ""),
                  ("scroll(0, 9)", "http://s/home?x=1"), (None, "http://s/home"), ("fill('2', 'a')", "http://s/home"),
                  last="http://s/home")
    # record 0 is the user step's observation, records 1 to 4 are results, record 5 follows a record with no call (unpaired)
    assert run.dataset["unpaired_observations"][0]["record"] == 5
    got = [(a.tool, a.resource, a.mode) for a in D.web_urls_v1(run)]
    assert got == [("goto", "url:http://s/home", "write"), ("click", "url:http://s/home", "write"), ("noop", None, "read"),
                   ("scroll", "url:http://s/home?x=1", "read"), ("noop", "url:http://s/home", "read"),
                   ("fill", "url:http://s/home", "write")]  # the last record is no step and so no access
    out = [(d["src"]["step"], d["dst"]["step"]) for d in ap.derive_deps(run, rules=[D.resource_chain(D.web_urls_v1)])]
    assert out == [(2, 3), (3, 6), (6, 7)]  # step 1 is the goal; a query string makes another resource; no URL means no edge


def test_web_urls_takes_the_action_of_the_call_and_the_url_of_the_record():
    run = web_run(("click('a', button='left')  # note", "http://s/1"), ("press('b', 'Enter')", "http://s/1"),
                  ("send_msg_to_user('x (y)')", "http://s/2"), last="http://s/3")
    assert [(a.address["call"], a.tool, a.mode, a.resource) for a in D.web_urls_v1(run)] == [
        ("ap-2-0", "click", "write", "url:http://s/1"), ("ap-3-0", "press", "write", "url:http://s/1"),
        ("ap-4-0", "send_msg_to_user", "read", "url:http://s/2")]
