import json
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats.messages import messages_to_trajectory

FIX = Path(__file__).parent / "fixtures"


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


CONVERSATION = [
    {"role": "system", "content": "You are a support agent."},
    {"role": "user", "content": "Cancel order 42."},
    {"role": "assistant", "content": None,
     "tool_calls": [_call("a", "get_order", '{"id": 42}'), _call("b", "get_user", '{"id": 7}')]},
    {"role": "tool", "tool_call_id": "b", "name": "get_user", "content": "user 7"},
    {"role": "tool", "tool_call_id": "a", "name": "get_order", "content": "order 42"},
    {"role": "assistant", "content": "Cancelling.", "tool_calls": [_call("c", "cancel", "{'id': 42")]},
    {"role": "user", "content": "Thanks."},
    {"role": "tool", "tool_call_id": "c", "content": None},
    {"role": "assistant", "content": "Done.", "turn_latency_ms": 812},
]


def _run(messages=CONVERSATION):
    return ap.Run(messages_to_trajectory(messages, trajectory_id="t1", ext={"source": {"format": "test"}}))


def test_valid_atif_with_calls_paired_by_id():
    run = _run()
    assert ap.validate(run) == []
    assert [s["source"] for s in run.steps] == ["system", "user", "agent", "agent", "user", "agent"]
    results = run.steps[2]["observation"]["results"]
    assert [r["source_call_id"] for r in results] == ["b", "a"]
    assert run.steps[2]["extra"]["agentpile"]["result_refs"] == [3, 4]


def test_late_result_attaches_to_its_call_and_keeps_source_order():
    run = _run()
    step = run.steps[3]
    assert step["observation"]["results"][0]["source_call_id"] == "c"
    assert step["extra"]["agentpile"]["result_refs"] == [7]  # after the user turn at index 6


def test_stand_ins_are_reported_as_source_values():
    events = list(_run().events())
    first_agent = [e for e in events if e.kind == "message" and e.address["step"] == 3][0]
    assert first_agent.content is None
    bad_args = [e for e in events if e.call_id == "c" and e.kind == "call"][0]
    assert bad_args.content == "{'id': 42"
    null_result = [e for e in events if e.call_id == "c" and e.kind == "result"][0]
    assert null_result.content is None
    named = [e for e in events if e.call_id == "b" and e.kind == "result"][0]
    assert named.name == "get_user" and named.content == "user 7"


def test_unknown_keys_are_kept():
    last = _run().steps[-1]
    assert last["extra"]["agentpile"]["source_fields"] == {"turn_latency_ms": 812}


def test_missing_ids_are_generated_and_never_paired():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "content": "a.py"},
        {"role": "tool", "tool_call_id": "zzz", "content": "orphan"},
    ]
    run = _run(msgs)
    assert ap.validate(run) == []
    call = run.steps[1]["tool_calls"][0]
    assert call["tool_call_id"] == "ap-2-0"
    assert run.steps[1]["extra"]["agentpile"]["calls"]["ap-2-0"]["id_generated"] is True
    results = run.steps[1]["observation"]["results"]
    assert [r["source_call_id"] for r in results] == [None, None]
    assert run.steps[1]["extra"]["agentpile"]["result_fields"][1]["tool_call_id"] == "zzz"
    assert ap.derive_deps(run, rules=["tool_call_id_v1"]) == []


def test_agentdojo_call_shape():
    msgs = [{"role": "assistant", "content": "x",
             "tool_calls": [{"id": "1", "function": "send_email", "args": {"to": "a@b.c"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "sent"}]
    run = _run(msgs)
    assert run.steps[0]["tool_calls"] == [{"tool_call_id": "1", "function_name": "send_email",
                                           "arguments": {"to": "a@b.c"}}]
    assert ap.validate(run) == []


def test_empty_conversation_is_marked_not_invented():
    run = _run([])
    assert len(run.steps) == 1 and run.steps[0]["extra"]["agentpile"]["empty_source"] is True
    assert list(run.events()) == []


def test_file_reader_bare_list_and_jsonl(tmp_path):
    bare = tmp_path / "chat.json"
    bare.write_text(json.dumps(CONVERSATION))
    run = next(ap.load(bare))
    assert run.source["format"] == "openai-messages" and len(run.steps) == 6
    jl = tmp_path / "chats.jsonl"
    jl.write_text("\n".join(json.dumps({"id": i, "messages": CONVERSATION, "label": i % 2}) for i in range(3)))
    runs = list(ap.load(jl))
    assert [r.id for r in runs] == ["0", "1", "2"]
    assert runs[1].dataset == {"label": 1}


def test_messages_reader_does_not_claim_other_formats():
    assert next(ap.load(FIX / "atif_multiagent.json")).source["format"] == "synthetic"
    assert next(ap.load(FIX / "config_mcp.json")).kind == "config"


def test_pair_by_position_for_id_less_results():
    msgs = [
        {"role": "user", "content": "check two things"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x1", "name": "get_a", "arguments": {"q": 1}},
            {"id": "x2", "name": "get_b", "arguments": {"q": 2}}]},
        {"role": "tool", "content": "A"},
        {"role": "tool", "content": "B"},
        {"role": "user", "content": "and?"},
        {"role": "tool", "content": "late"},
    ]
    plain = _run(msgs)
    assert [r["source_call_id"] for r in plain.steps[1]["observation"]["results"]] == [None, None, None]
    run = ap.Run(messages_to_trajectory(msgs, trajectory_id="p", pair_by_position=True))
    results = run.steps[1]["observation"]["results"]
    assert [r["source_call_id"] for r in results] == ["x1", "x2", None]  # a user turn closes the window
    fields = run.steps[1]["extra"]["agentpile"]["result_fields"]
    assert fields[0] == {"paired_by": "position"} and fields[2] is None
    assert ap.validate(run) == []


def test_source_override_for_a_tool_calling_user_simulator():
    msgs = [
        {"role": "assistant", "content": "Please toggle airplane mode."},
        {"role": "user", "content": None, "tool_calls": [{"id": "u1", "name": "toggle_airplane_mode", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "u1", "content": "airplane mode off"},
        {"role": "user", "content": "Done."},
    ]

    def source_of(m):
        return "agent" if m.get("role") == "user" and m.get("tool_calls") else None

    def actor(m):
        return "user" if m.get("role") == "user" else "assistant"

    run = ap.Run(messages_to_trajectory(msgs, trajectory_id="s", source_of=source_of, actor=actor))
    assert [s["source"] for s in run.steps] == ["agent", "agent", "user"]
    sx = run.steps[1]["extra"]["agentpile"]
    assert sx["role"] == "user" and sx["actor"] == "user"
    assert run.steps[1]["observation"]["results"][0]["source_call_id"] == "u1"
    assert ap.validate(run) == []


# -- recorded source order -------------------------------------------------------------------------

def test_events_follow_the_recorded_message_order():
    events = list(_run().events())
    # Before this check the result of call c (message 7) came out beside its call, ahead of the user turn at 6.
    assert [e.source_ref for e in events if e.kind != "call"] == list(range(9))
    late = [e for e in events if e.kind == "result" and e.source_ref == 7][0]
    assert late.call_id == "c" and late.address == {"trajectory": "t1", "step": 4, "result": 0}
    assert [e.address["step"] for e in events if e.source_ref in (5, 6, 7)] == [4, 5, 4]  # step 4 holds messages 5 and 7


# -- absent, null and empty arguments ---------------------------------------------------------------

def _args_conversation():
    return [
        {"role": "user", "content": "call the tools"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "absent", "type": "function", "function": {"name": "f"}},
            {"id": "null", "type": "function", "function": {"name": "f", "arguments": None}},
            {"id": "empty", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            {"id": "object", "type": "function", "function": {"name": "f", "arguments": {}}},
            {"id": "text", "type": "function", "function": {"name": "f", "arguments": "not json"}},
            {"id": "dojo-absent", "function": "g"},
            {"id": "dojo-null", "function": "g", "args": None},
            {"id": "flat-absent", "name": "h"},
            {"id": "flat-null", "name": "h", "arguments": None},
            {"id": "list", "type": "function", "function": {"name": "f", "arguments": "[1]"}},
        ]},
    ]


def _arguments(run):
    return {e.call_id: e.content for e in run.events() if e.kind == "call"}


def test_absent_null_and_empty_arguments_stay_three_different_things():
    run = _run(_args_conversation())
    args = _arguments(run)
    assert args["absent"] is ap.MISSING and args["null"] is None
    assert args["empty"] == {} and args["object"] == {}
    assert args["text"] == "not json"
    assert args["dojo-absent"] is ap.MISSING and args["dojo-null"] is None
    assert args["flat-absent"] is ap.MISSING and args["flat-null"] is None
    calls = run.steps[1]["extra"]["agentpile"]["calls"]
    assert calls["absent"] == {"arguments_state": "missing"}  # a JSON-safe marker, never the MISSING object
    assert calls["null"] == {"arguments_raw": None}
    assert "empty" not in calls and "object" not in calls  # an explicit empty object needs no marker
    # arguments that are no JSON object (text, a JSON list) stay as given, and the call says ATIF holds a stand-in
    assert calls["text"] == {"arguments_raw": "not json", "arguments_state": "unparsed"}
    assert calls["list"] == {"arguments_raw": "[1]", "arguments_state": "unparsed"} and args["list"] == "[1]"
    assert "arguments_state" not in calls["null"]  # a null is not a failed parse
    assert all(c["arguments"] == {} for c in run.steps[1]["tool_calls"][:4])  # ATIF holds {} in every case
    assert ap.validate(run) == []


def test_argument_states_survive_serialization_and_reloading(tmp_path):
    run = _run(_args_conversation())
    text = run.to_json()
    assert "MISSING" not in text
    again = ap.Run(json.loads(text))
    assert again == run and _arguments(again) == _arguments(run)
    path = tmp_path / "args.json"
    path.write_text(json.dumps({"id": "args", "messages": _args_conversation()}))
    loaded = next(ap.load(path))
    args = _arguments(loaded)
    assert args["absent"] is ap.MISSING and args["null"] is None and args["empty"] == {}
    out = tmp_path / "out.json"
    ap.writers.get("json").write(loaded, out)
    assert _arguments(next(ap.load(out))) == args


def test_argument_state_marker_is_checked_by_the_schema_and_harbor():
    run = _run(_args_conversation())
    assert ap.validate(run) == []
    trajectories = pytest.importorskip("harbor.models.trajectories")
    trajectories.Trajectory.model_validate(run.record)


# -- tool definitions in the public view -----------------------------------------------------------

def _fn(name, desc="d"):
    return {"type": "function", "function": {"name": name, "description": desc,
                                              "parameters": {"type": "object", "properties": {}}}}


def test_recorded_tool_definitions_reach_the_public_tool_view():
    defs = [_fn("search"), _fn("book")]
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [_call("c1", "book", "{}"), _call("c2", "other", "{}")]}]
    run = ap.Run(messages_to_trajectory(msgs, trajectory_id="t", tool_definitions=defs, ext={"source": {"format": "test"}}))
    assert ap.validate(run) == []
    assert {t: (e["name"], e["atif_index"], e["definition"]) for t, e in run.tools.items()} == {
        "search": ("search", 0, defs[0]), "book": ("book", 1, defs[1])}
    for entry in run.tools.values():
        assert entry["definition"] == run.record["agent"]["tool_definitions"][entry["atif_index"]]
        assert entry["definition"] is not run.record["agent"]["tool_definitions"][entry["atif_index"]]
        assert (entry["agent_or_server"], entry["permission"], entry["permission_from"]) == (None, "unknown", "unknown")
    calls = run.steps[1]["extra"]["agentpile"]["calls"]
    assert calls == {"c1": {"tool_id": "book"}}  # a name no definition has links nothing
    assert [e.extra["tool_id"] for e in run.events() if e.kind == "call"] == ["book", None]


def test_a_name_two_definitions_share_links_no_call():
    defs = [_fn("dup", "first"), _fn("dup", "second"), _fn("solo")]
    msgs = [{"role": "assistant", "content": "", "tool_calls": [_call("c1", "dup", "{}"), _call("c2", "solo", "{}")]}]
    run = ap.Run(messages_to_trajectory(msgs, trajectory_id="t", tool_definitions=defs))
    assert ap.validate(run) == []
    assert {t: e["atif_index"] for t, e in run.tools.items()} == {"dup": 0, "dup#1": 1, "solo": 2}
    assert run.steps[0]["extra"]["agentpile"]["calls"] == {"c2": {"tool_id": "solo"}}


def test_tools_a_caller_supplied_are_kept_not_duplicated():
    defs = [_fn("a"), _fn("b")]
    mine = {"x.a": {"name": "a", "agent_or_server": "x", "type": "function", "definition": defs[0],
                    "definition_format": "openai", "atif_index": 0, "permission": "read", "permission_from": "declared"},
            "x.b": {"name": "b", "agent_or_server": "x", "type": "function", "definition": defs[1],
                    "definition_format": "openai", "atif_index": 1, "permission": "write", "permission_from": "declared"}}
    msgs = [{"role": "assistant", "content": "", "tool_calls": [_call("c1", "a", "{}")]}]
    run = ap.Run(messages_to_trajectory(msgs, trajectory_id="t", tool_definitions=defs, ext={"tools": mine}))
    assert run.tools == mine and ap.validate(run) == []
    assert "calls" not in run.steps[0].get("extra", {}).get("agentpile", {})  # the caller owns the links


def test_message_file_tools_member_is_read_and_a_non_list_is_kept(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps({"id": "t", "tools": [_fn("only")], "messages": [{"role": "user", "content": "x"}]}))
    run = next(ap.load(path))
    assert list(run.tools) == ["only"] and run.record["agent"]["tool_definitions"] == [_fn("only")]
    assert "tools" not in run.dataset and ap.validate(run) == []
    path.write_text(json.dumps({"id": "t", "tools": None, "messages": [{"role": "user", "content": "x"}]}))
    run = next(ap.load(path))
    assert run.tools == {} and "tool_definitions" not in run.record["agent"] and run.dataset == {"tools": None}


def test_tool_view_and_atif_definitions_agree_for_harbor():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    msgs = [{"role": "user", "content": "go"}]
    record = messages_to_trajectory(msgs, trajectory_id="t", tool_definitions=[_fn("a")])
    trajectories.Trajectory.model_validate(record)


# -- content parts ---------------------------------------------------------------------------------

MULTIMODAL = FIX / "tools" / "multimodal_messages.json"
STAND_IN = "[image_url part: original in agentpile extension]"


def test_multimodal_fixture_is_valid_atif_for_both_validators():
    run = next(ap.load(MULTIMODAL))
    assert ap.validate(run) == []
    trajectories = pytest.importorskip("harbor.models.trajectories")
    trajectories.Trajectory.model_validate(run.record)


def test_text_parts_map_and_unsupported_parts_get_a_marked_stand_in():
    run = next(ap.load(MULTIMODAL))
    user = run.steps[1]
    assert user["message"] == [{"type": "text", "text": "What is in this picture?"}, {"type": "text", "text": STAND_IN}]
    original = json.loads(MULTIMODAL.read_text())["messages"][1]["content"]
    assert user["extra"]["agentpile"]["message_raw"] == original  # image_url and its detail stay whole
    result = run.steps[2]["observation"]["results"][0]["content"]
    assert result == [{"type": "text", "text": "A cat on a sofa."}, {"type": "text", "text": STAND_IN}]
    original_result = json.loads(MULTIMODAL.read_text())["messages"][3]["content"]
    assert run.steps[2]["extra"]["agentpile"]["result_fields"][0]["content_raw"] == original_result
    # a text-only list is already ATIF, so it is kept as it is and needs no copy of the source
    assert run.steps[2]["message"] == [{"type": "text", "text": "Let me look at it."}]
    assert "message_raw" not in run.steps[2]["extra"]["agentpile"]
    # a text part with a vendor member maps to its text; the member stays in the source copy
    last = run.steps[3]
    assert last["message"] == [{"type": "text", "text": "It is a cat."}]
    assert last["extra"]["agentpile"]["message_raw"][0]["cache_control"] == {"type": "ephemeral"}


def test_events_give_the_original_parts():
    run = next(ap.load(MULTIMODAL))
    raw = json.loads(MULTIMODAL.read_text())["messages"]
    events = list(run.events())
    assert [e.content for e in events if e.kind == "message"][1:] == [
        raw[1]["content"], raw[2]["content"], raw[4]["content"]]
    assert [e.content for e in events if e.kind == "result"] == [raw[3]["content"]]
    assert ap.Run(json.loads(run.to_json())) == run


def test_an_atif_image_part_and_other_shapes_of_part():
    image = {"type": "image", "source": {"media_type": "image/png", "path": "shots/a.png"}}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "see"}, image]},
            {"role": "user", "content": ["a bare string", {"type": "input_audio", "input_audio": {"data": "x"}}, 5, {}]},
            {"role": "user", "content": []}]
    run = _run(msgs)
    assert run.steps[0]["message"] == [{"type": "text", "text": "see"}, image]  # already ATIF: kept, no copy
    assert "message_raw" not in run.steps[0]["extra"]["agentpile"]
    mixed = run.steps[1]
    assert mixed["message"] == [{"type": "text", "text": "a bare string"},
                                {"type": "text", "text": "[input_audio part: original in agentpile extension]"},
                                {"type": "text", "text": "[unknown part: original in agentpile extension]"},
                                {"type": "text", "text": "[unknown part: original in agentpile extension]"}]
    assert mixed["extra"]["agentpile"]["message_raw"] == msgs[1]["content"]
    assert run.steps[2]["message"] == [] and ap.validate(run) == []
    trajectories = pytest.importorskip("harbor.models.trajectories")
    trajectories.Trajectory.model_validate(run.record)


def test_a_gzipped_message_list_reads_like_the_plain_file(tmp_path):
    import gzip

    packed = tmp_path / "chat.json.gz"
    with gzip.open(packed, "wt", encoding="utf-8") as fh:
        json.dump(CONVERSATION, fh)
    plain = tmp_path / "chat.json"
    plain.write_text(json.dumps(CONVERSATION))
    run = next(ap.load(packed))
    assert run.source["format"] == "openai-messages" and run.steps == next(ap.load(plain)).steps
