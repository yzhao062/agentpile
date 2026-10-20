import copy
import json
import pickle
from pathlib import Path

import agentpile as ap

FIX = Path(__file__).parent / "fixtures"


def _run():
    return next(ap.load(FIX / "atif_multiagent.json"))


def test_events_follow_source_order():
    # The fixture stores the results of step 3 as [c2, c1] but records their source positions as [3, 2]:
    # c1's result is message 2 of the source and c2's is message 3, so the event view gives c1 first.
    # (An earlier version of this test pinned the ATIF order c2, c1, which is not the source order.)
    kinds = [(e.kind, e.address.get("step"), e.call_id) for e in _run().events()]
    assert kinds == [
        ("message", 1, None),
        ("message", 2, None),
        ("message", 3, None),
        ("call", 3, "c1"),
        ("call", 3, "c2"),
        ("result", 3, "c1"),
        ("result", 3, "c2"),
        ("message", 4, None),
        ("call", 4, "c3"),
        ("result", 4, "c3"),
        ("message", 5, None),
    ]


def test_event_order_leaves_addresses_and_call_association_alone():
    results = [e for e in _run().events() if e.kind == "result" and e.address["step"] == 3]
    assert [(e.call_id, e.address["result"]) for e in results] == [("c1", 1), ("c2", 0)]  # indexes into the ATIF results
    assert [e.content for e in results] == ["[{\"offer_id\": 7, \"depart\": \"09:00\"}]", "No events."]


def test_events_keep_the_step_order_where_positions_are_not_integers():
    for ref in ({"span": "s1"}, "goal", 1.5, True):
        record = _run().to_dict()
        record["steps"][2]["extra"]["agentpile"]["result_refs"] = [ref, 2]  # a position the order cannot use
        results = [e for e in ap.Run(record).events() if e.kind == "result" and e.address["step"] == 3]
        assert [e.call_id for e in results] == ["c2", "c1"], ref  # the ATIF order


def test_events_give_source_values_where_atif_needed_stand_ins():
    events = list(_run().events())
    step4_message = [e for e in events if e.kind == "message" and e.address["step"] == 4][0]
    assert step4_message.content is None  # null in the source, "" in ATIF
    call = [e for e in events if e.call_id == "c3" and e.kind == "call"][0]
    assert call.content == "{'offer_id': 7"  # not JSON in the source, {} in ATIF
    assert call.error["type"] == "tool_error"
    result = [e for e in events if e.call_id == "c3" and e.kind == "result"][0]
    assert result.error == call.error


def test_events_carry_actor_address_and_source_ref():
    events = list(_run().events())
    c1 = [e for e in events if e.kind == "call" and e.call_id == "c1"][0]
    assert c1.actor == "planner" and c1.name == "search_flights"
    assert c1.address == {"trajectory": "fx-1", "step": 3, "call": "c1"}
    assert c1.extra["tool_id"] == "t.search"
    results = [e for e in events if e.kind == "result" and e.address["step"] == 3]
    assert [e.source_ref for e in results] == [2, 3]  # source order, not the order of the ATIF results [3, 2]


def test_missing_message_yields_no_message_event():
    record = _run().to_dict()
    record["steps"][4]["extra"]["agentpile"]["message_state"] = "missing"
    events = list(ap.Run(record).events())
    assert not [e for e in events if e.kind == "message" and e.address["step"] == 5]


def test_shared_accessors():
    run = _run()
    assert [a["id"] for a in run.agents] == ["planner", "booker"]
    assert run.tools["t.book"]["permission"] == "write"
    assert run.labels[0]["target"] == {"kind": "step", "address": {"trajectory": "fx-1", "step": 4}}
    assert run.outcome["success"] is False
    assert run.task["id"] == "travel-1"
    assert run.dataset["fixture_note"].startswith("synthetic")


def test_config_record_has_no_steps_or_events():
    run = next(ap.load(FIX / "config_mcp.json"))
    assert run.steps == [] and list(run.events()) == [] and run.subagents == []
    assert run.tools["fs.delete_file"]["definition"]["annotations"]["destructiveHint"] is True
    assert run.id == "fx-config-1"


def test_json_round_trip(tmp_path):
    run = _run()
    assert ap.Run(json.loads(run.to_json())) == run
    out = tmp_path / "run.json"
    ap.writers.get("json").write(run, out)
    assert next(ap.load(out)) == run


def test_to_dict_is_a_copy():
    run = _run()
    d = run.to_dict()
    d["steps"].clear()
    assert len(run.steps) == 5


def test_missing_is_a_singleton():
    assert copy.deepcopy(ap.MISSING) is ap.MISSING
    assert pickle.loads(pickle.dumps(ap.MISSING)) is ap.MISSING
    assert not ap.MISSING


def test_native_atif_tool_definitions_reach_the_public_tool_view(tmp_path):
    defs = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "search", "description": "a second definition, same name"}},
            {"type": "function", "function": {"name": "book"}}]
    record = {"schema_version": "ATIF-v1.7", "session_id": "native-1", "trajectory_id": "native-1",
              "agent": {"name": "third-party", "version": "1", "tool_definitions": defs},
              "steps": [{"step_id": 1, "source": "user", "message": "hi"}]}
    run = ap.Run(record)
    assert ap.validate(record) == []
    assert [(t["atif_index"], t["definition"]) for t in run.tools.values()] == list(enumerate(defs))
    assert len(set(run.tools)) == 3  # the repeated name still gets its own id
    path = tmp_path / "native.json"
    path.write_text(json.dumps(record))
    back = next(ap.load(path))
    assert back.record == record and back.tools == run.tools  # the view adds entries; the record is unchanged
