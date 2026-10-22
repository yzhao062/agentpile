import json
from pathlib import Path

import pytest

import agentpile as ap

FIX = Path(__file__).parent / "fixtures"


def _record():
    return json.loads((FIX / "atif_multiagent.json").read_text())


def _config():
    return json.loads((FIX / "config_mcp.json").read_text())


def test_fixtures_are_valid():
    assert ap.validate(_record()) == []
    assert ap.validate(_config()) == []


def _break_step_id(r):
    r["steps"][2]["step_id"] = 7


def _break_source(r):
    r["steps"][1]["source"] = "tool"


def _break_message(r):
    r["steps"][3]["message"] = None


def _break_result_link(r):
    r["steps"][2]["observation"]["results"][0]["source_call_id"] = "c3"


def _break_permission(r):
    r["extra"]["agentpile"]["tools"]["t.book"]["permission"] = "sudo"


def _break_atif_index(r):
    r["extra"]["agentpile"]["tools"]["t.book"]["atif_index"] = 0


def _break_call_ext(r):
    r["steps"][2]["extra"]["agentpile"]["calls"]["c9"] = {"tool_id": "t.search"}


def _break_label_target(r):
    r["extra"]["agentpile"]["labels"][0]["target"]["kind"] = "token"


def _break_no_steps(r):
    r["steps"] = []


def _break_unknown_key(r):
    r["steps"][0]["speaker"] = "x"


def _break_agent_only(r):
    r["steps"][1]["reasoning_content"] = "hm"


def _break_agent_version(r):
    del r["agent"]["version"]


@pytest.mark.parametrize("breaker, needle", [
    (_break_step_id, "step_id: expected 3"),
    (_break_source, "source: 'tool'"),
    (_break_message, "message: ATIF requires"),
    (_break_result_link, "matches no call on this step"),
    (_break_permission, "'sudo' is not one of"),
    (_break_atif_index, "atif_index values"),
    (_break_call_ext, "'c9' matches no call"),
    (_break_label_target, "'token' is not one of"),
    (_break_no_steps, "at least one step"),
    (_break_unknown_key, "unknown ATIF field 'speaker'"),
    (_break_agent_only, "agent steps only"),
    (_break_agent_version, "agent.version: required"),
])
def test_validate_reports_each_break(breaker, needle):
    r = _record()
    breaker(r)
    errors = ap.validate(r)
    assert any(needle in e for e in errors), errors


def test_config_requires_schema_and_source():
    c = _config()
    del c["source"]
    assert any("missing required field 'source'" in e for e in ap.validate(ap.Run(c, "config")))


def test_subagent_trajectories_are_checked():
    r = _record()
    sub = _record()
    sub["steps"][0]["step_id"] = 2
    sub["trajectory_id"] = "sub-1"
    r["subagent_trajectories"] = [sub]
    assert any(e.startswith("$.subagent_trajectories[0].steps[0].step_id") for e in ap.validate(r))


# -- the public tool definition must equal the ATIF one it indexes ------------------------------------

def _tool(r, tid="t.book"):
    return r["extra"]["agentpile"]["tools"][tid]


def _null_definition(r):
    _tool(r)["definition"] = None


def _unrelated_definition(r):
    _tool(r)["definition"] = {"name": "something_else"}


def _mutated_definition(r):
    _tool(r)["definition"]["function"]["description"] = "Book two flights."


def _swapped_definitions(r):
    _tool(r, "t.search")["definition"], _tool(r, "t.calendar")["definition"] = (
        _tool(r, "t.calendar")["definition"], _tool(r, "t.search")["definition"])


def _missing_tool_entry(r):
    del r["extra"]["agentpile"]["tools"]["t.book"]


@pytest.mark.parametrize("breaker, needle", [
    (_null_definition, "tools['t.book'].definition: differs from agent.tool_definitions[2]"),
    (_unrelated_definition, "tools['t.book'].definition: differs from agent.tool_definitions[2]"),
    (_mutated_definition, "tools['t.book'].definition: differs from agent.tool_definitions[2]"),
    (_swapped_definitions, "tools['t.search'].definition: differs from agent.tool_definitions[0]"),
    (_missing_tool_entry, "atif_index values [0, 1] do not cover"),
])
def test_validate_checks_public_definitions_against_atif(breaker, needle):
    r = _record()
    breaker(r)
    errors = ap.validate(r)
    assert any(needle in e for e in errors), errors


def test_a_tool_without_an_atif_index_may_hold_any_definition():
    r = _record()
    r["extra"]["agentpile"]["tools"]["extra.tool"] = {"name": "x", "definition": {"anything": 1}, "atif_index": None}
    assert ap.validate(r) == []


# -- ATIF content parts ------------------------------------------------------------------------------

IMAGE = {"type": "image", "source": {"media_type": "image/png", "path": "a.png"}}


def _with_message(content):
    def breaker(r):
        r["steps"][1]["message"] = content
    return breaker


def _with_result(content):
    def breaker(r):
        r["steps"][2]["observation"]["results"][0]["content"] = content
    return breaker


@pytest.mark.parametrize("breaker, needle", [
    (_with_message([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]),
     "steps[1].message[1].type: 'image_url' is not an ATIF content part type"),
    (_with_message([{"type": "text"}]), "message[0].text: a text part needs a string"),
    (_with_message([{"type": "text", "text": "a", "cache_control": {}}]), "unknown ATIF content part field 'cache_control'"),
    (_with_message(["bare"]), "message[0]: ATIF content part must be an object"),
    (_with_message([{"type": "image"}]), "message[0].source: an image part needs a source object"),
    (_with_message([{"type": "image", "source": {"media_type": "image/bmp", "path": "a"}}]), "media_type: 'image/bmp'"),
    (_with_message([{"type": "image", "source": {"media_type": "image/png"}}]), "source.path: required string"),
    (_with_message([{"type": "image", "source": IMAGE["source"], "text": "x"}]), "text: not allowed on an image part"),
    (_with_result([{"type": "text", "text": "a"}, {"type": "audio"}]), "results[0].content[1].type: 'audio'"),
    (_with_result({"not": "a string"}), "results[0].content: ATIF requires a string, content parts or null"),
])
def test_validate_checks_content_parts(breaker, needle):
    r = _record()
    breaker(r)
    errors = ap.validate(r)
    assert any(needle in e for e in errors), errors


def test_validate_accepts_text_and_image_parts_and_null_result_content():
    for breaker in (_with_message([{"type": "text", "text": "a"}, IMAGE]), _with_message([]), _with_result(None),
                    _with_result([IMAGE, {"type": "text", "text": ""}])):
        r = _record()
        breaker(r)
        assert ap.validate(r) == []


def test_validate_and_harbor_agree_on_content_parts():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    cases = [_with_message([{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u"}}]),
             _with_message([{"type": "text", "text": "a", "cache_control": {}}]),
             _with_message([{"type": "image", "source": {"media_type": "image/bmp", "path": "a"}}]),
             _with_result([{"type": "audio"}]),
             _with_message([{"type": "text", "text": "a"}, IMAGE]), _with_message([]), _with_result(None)]
    for breaker in cases:
        r = _record()
        breaker(r)
        ours = ap.validate(r) == []
        try:
            trajectories.Trajectory.model_validate(r)
            theirs = True
        except Exception:
            theirs = False
        assert ours == theirs, (r["steps"][1]["message"], r["steps"][2]["observation"]["results"][0]["content"])


# -- the argument-state marker --------------------------------------------------------------------------

def test_validate_checks_the_arguments_state_marker():
    r = _record()
    calls = r["steps"][2]["extra"]["agentpile"]["calls"]
    calls["c1"]["arguments_state"] = "missing"
    assert ap.validate(r) == []
    calls["c1"]["arguments_raw"] = None  # absent and null together say two things
    assert any("cannot both be set" in e for e in ap.validate(r))
    del calls["c1"]["arguments_raw"]
    calls["c1"]["arguments_state"] = "null"
    assert any("arguments_state: 'null' is not 'missing'" in e for e in ap.validate(r))


def test_validate_checks_the_unparsed_marker():
    r = _record()
    step = r["steps"][3]  # call c3 carries the source text of arguments that were no JSON object
    cx = step["extra"]["agentpile"]["calls"]["c3"]
    assert cx["arguments_state"] == "unparsed" and ap.validate(r) == []
    cx["arguments_raw"] = None  # a null is no failed parse
    assert any("needs the source value in arguments_raw" in e for e in ap.validate(r))
    del cx["arguments_raw"]
    assert any("needs the source value in arguments_raw" in e for e in ap.validate(r))
    cx["arguments_raw"] = "{bad"
    step["tool_calls"][0]["arguments"] = {"offer_id": 7}  # the marker says ATIF holds the stand-in {}
    assert any("says ATIF holds {}" in e for e in ap.validate(r))
    step["tool_calls"][0]["arguments"] = {}
    assert ap.validate(r) == []
