from pathlib import Path

import pytest

import agentpile as ap

FIX = Path(__file__).parent / "fixtures"


def _run():
    return next(ap.load(FIX / "atif_multiagent.json"))


def test_tool_call_id_rule_links_results_to_the_next_agent_step():
    deps = ap.derive_deps(_run(), rules=["tool_call_id_v1"])
    edges = [(d["src"]["step"], d["src"]["call"], d["dst"]["step"]) for d in deps]
    assert edges == [(3, "c1", 4), (3, "c2", 4), (4, "c3", 5)]
    assert {(d["rule"], d["rule_version"]) for d in deps} == {("tool_call_id", "1")}
    assert all(d["src"]["trajectory"] == "fx-1" for d in deps)


def test_tool_call_id_rule_does_not_guess_unlinked_results():
    record = _run().to_dict()
    for res in record["steps"][2]["observation"]["results"]:
        res["source_call_id"] = None
    deps = ap.derive_deps(ap.Run(record), rules=["tool_call_id_v1"])
    assert [(d["src"]["call"], d["dst"]["step"]) for d in deps] == [("c3", 5)]


def test_full_context_accepts_the_unversioned_name():
    deps = ap.derive_deps(_run(), rules=["full_context"])
    assert len(deps) == 10  # 5 steps: 4 + 3 + 2 + 1
    assert {d["rule"] for d in deps} == {"full_context"}


def test_custom_rule_and_unknown_rule():
    def nothing(run):
        return []

    assert ap.derive_deps(_run(), rules=[nothing]) == []
    with pytest.raises(KeyError, match="unknown dependency rule"):
        ap.derive_deps(_run(), rules=["telepathy_v1"])


def test_config_record_has_no_deps():
    assert ap.derive_deps(next(ap.load(FIX / "config_mcp.json")), rules=["full_context"]) == []


# -- recorded source order ------------------------------------------------------------------------------

def _delayed(final_message=None):
    from agentpile.formats.messages import messages_to_trajectory

    call = {"id": "c", "type": "function", "function": {"name": "slow", "arguments": "{}"}}
    last = {"role": "assistant", "content": "ok"} if final_message is None else final_message
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "assistant", "content": "still waiting"},
        {"role": "tool", "tool_call_id": "c", "content": "done"},
        last,
    ]
    return ap.Run(messages_to_trajectory(msgs, trajectory_id="d"))


def test_tool_call_id_rule_never_uses_a_result_before_it_occurred():
    run = _delayed()
    assert [s["source"] for s in run.steps] == ["user", "agent", "agent", "agent"]
    assert [e.source_ref for e in run.events() if e.kind != "call"] == [0, 1, 2, 3, 4]
    deps = ap.derive_deps(run, rules=["tool_call_id_v1"])
    # step 3 ("still waiting") was recorded before the result at message 3, so only step 4 can depend on it
    assert [(d["src"]["step"], d["src"]["call"], d["dst"]["step"]) for d in deps] == [(2, "c", 4)]


def test_an_agent_step_with_no_message_still_consumes_a_result_that_came_before_it():
    run = _delayed({"role": "assistant"})  # no content member at all
    assert run.steps[3]["extra"]["agentpile"]["message_state"] == "missing"
    assert not [e for e in run.events() if e.kind == "message" and e.address["step"] == 4]
    deps = ap.derive_deps(run, rules=["tool_call_id_v1"])
    assert [(d["src"]["step"], d["dst"]["step"]) for d in deps] == [(2, 4)]


def test_a_user_turn_after_the_result_does_not_consume_it():
    run = _delayed({"role": "user", "content": "and?"})
    assert ap.derive_deps(run, rules=["tool_call_id_v1"]) == []


def test_results_in_source_order_make_a_pending_set_in_that_order():
    deps = ap.derive_deps(_run(), rules=["tool_call_id_v1"])
    assert [(d["src"]["call"], d["dst"]["step"]) for d in deps] == [("c1", 4), ("c2", 4), ("c3", 5)]
