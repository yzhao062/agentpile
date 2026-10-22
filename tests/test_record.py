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


def _span_addressed():
    """The fixture as a reader with span addresses writes it: references are dicts, integer sort positions sit beside them."""
    record = _run().to_dict()
    for step in record["steps"]:
        sx = step.get("extra", {}).get("agentpile", {})
        if sx.get("source_ref") is not None:
            sx["source_pos"], sx["source_ref"] = sx["source_ref"], {"span": f"s{sx['source_ref']}"}
        if sx.get("result_refs") is not None:
            sx["result_pos"] = sx["result_refs"]
            sx["result_refs"] = [{"span": f"s{r}", "index": 0} for r in sx["result_refs"]]
    return record


def _shape(run):
    return [(e.kind, e.address.get("step"), e.call_id) for e in run.events()]


def test_sort_positions_order_events_whose_references_are_not_integers():
    # The source order of the integer-addressed fixture, though no reference is an integer.
    record = _span_addressed()
    assert isinstance(record["steps"][2]["extra"]["agentpile"]["source_ref"], dict)
    assert _shape(ap.Run(record)) == _shape(_run())
    results = [e for e in ap.Run(record).events() if e.kind == "result" and e.address["step"] == 3]
    assert [(e.call_id, e.address["result"]) for e in results] == [("c1", 1), ("c2", 0)]
    assert [e.source_ref for e in results] == [{"span": "s2", "index": 0}, {"span": "s3", "index": 0}]  # the address, as stored


def test_a_sort_position_wins_over_an_integer_reference():
    record = _run().to_dict()
    record["steps"][4]["extra"]["agentpile"]["source_pos"] = -5  # the last step's reference is 6
    first = next(ap.Run(record).events())
    assert (first.kind, first.address["step"]) == ("message", 5)
    assert next(_run().events()).address["step"] == 1


def test_events_keep_the_step_order_where_a_reference_has_no_usable_sort_position():
    plain = _run().to_dict()
    for step in plain["steps"]:
        for key in ("source_ref", "result_refs"):
            step.get("extra", {}).get("agentpile", {}).pop(key, None)
    atif = _shape(ap.Run(plain))
    assert [c for k, s, c in atif if k == "result" and s == 3] == ["c2", "c1"]
    cases = {"fewer sort positions than results": ("result_pos", [3]),
             "a sort position that is not an integer": ("result_pos", ["x", 2]),
             "a bool": ("result_pos", [3, True]),
             "a step sort position that is not an integer": ("source_pos", 1.5)}
    for why, (key, value) in cases.items():
        record = _span_addressed()
        record["steps"][2]["extra"]["agentpile"][key] = value
        assert _shape(ap.Run(record)) == atif, why
    record = _span_addressed()
    del record["steps"][2]["extra"]["agentpile"]["result_pos"]  # dict references and no sort position
    assert _shape(ap.Run(record)) == atif


def test_a_null_sort_position_leaves_the_result_beside_its_step():
    record = _span_addressed()
    sx = record["steps"][3]["extra"]["agentpile"]  # step 4, one result
    sx["result_refs"], sx["result_pos"] = [None], [None]
    assert [(k, s) for k, s, _ in _shape(ap.Run(record)) if s == 4] == [("message", 4), ("call", 4), ("result", 4)]


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
