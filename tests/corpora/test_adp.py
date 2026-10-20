"""ADP loader on a synthetic copy of the pinned file layout (no dataset content)."""

import collections
import json
import os
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.corpora import adp as adp_mod

ROOT = Path(__file__).parent / "fixtures" / "adp"


def _runs(**kw):
    return list(ap.load("adp", root=str(ROOT), **kw))


def _by_id():
    return {r.id: r for r in _runs()}


def _ext(step):
    return step["extra"]["agentpile"]


def test_default_load_reads_every_supported_config_in_manifest_order():
    runs = _runs()
    manifest = ap.datasets.get("adp").manifest
    configs = [c["config"] for c in manifest["supported_configs"]]
    assert len(configs) == 14 and "mind2web" not in configs
    order = [r.source["config"] for r in runs]
    assert [c for c in configs if c in order] == list(dict.fromkeys(order)) and set(order) == set(configs)
    assert all(ap.validate(r) == [] for r in runs)
    first = runs[0]
    assert first.id == "agenttuning_alfworld/synth_alf_0"
    assert first.source["file"] == "agenttuning_alfworld/full_std.jsonl" and first.source["split"] == "std"
    assert first.source["record_key"] == first.id and first.source["row"] == 0
    assert first.source["dataset"] == "neulab/agent-data-collection"
    assert first.source["revision"] == ap.datasets.get("adp").revision


def test_config_selects_one_and_rejects_unsupported_or_unknown_configs():
    assert [r.id for r in _runs(config="agenttuning_kg")] == [
        "agenttuning_kg/synth_kg_0", "agenttuning_kg/synth_kg_empty", "agenttuning_kg/synth_kg_edge"]
    assert [r.source["row"] for r in _runs(config="agenttuning_kg")] == [0, 1, 2]
    with pytest.raises(ValueError, match=r"unsupported: count check not possible"):
        _runs(config="mind2web")
    with pytest.raises(ValueError, match="config must be one of"):
        _runs(config="no_such_config")
    with pytest.raises(TypeError, match="takes config= only"):
        _runs(split="sft")


def test_linked_result_is_attached_to_its_action_step_and_flagged():
    step = _by_id()["agenttuning_alfworld/synth_alf_0"].steps[3]
    assert step["source"] == "agent" and step["message"] == "Check the counter first."
    call = step["tool_calls"][0]
    assert call == {"tool_call_id": "call_000001", "function_name": "go", "arguments": {"location": "\"counter 1\""}}
    assert step["observation"]["results"] == [{"source_call_id": "call_000001", "content": "You see a mug 1."}]
    sx = _ext(step)
    assert sx["source_ref"] == 3 and sx["result_refs"] == [4] and sx["source_fields"] == {"class_": "api_action"}
    assert sx["calls"]["call_000001"] == {"id_generated": True, "source_ref": 3}
    assert sx["result_fields"] == [{"class_": "text_observation", "source": "user", "paired_by": "adp_sequence"}]
    # one generated id per linked pair, counted in row order
    ids = [t["tool_call_id"] for s in _by_id()["agenttuning_alfworld/synth_alf_0"].steps for t in s.get("tool_calls", [])]
    assert ids == ["call_000001", "call_000002"]


def test_blank_description_stays_blank_and_null_or_missing_is_marked():
    steps = _by_id()["agenttuning_alfworld/synth_alf_0"].steps
    assert steps[4]["message"] == "" and "message_state" not in _ext(steps[4])
    assert _ext(_by_id()["agenttuning_alfworld/synth_alf_1"].steps[3])["message_state"] == "null"
    edge = _by_id()["agenttuning_kg/synth_kg_edge"].steps
    assert _ext(edge[4])["message_state"] == "missing"  # the api_action with no description key
    # the source value of an absent or null message reaches the event view
    assert [e.content for e in _by_id()["agenttuning_alfworld/synth_alf_1"].events() if e.kind == "message"][3] is None


def test_code_actions_follow_adp_function_names_and_keep_the_language():
    def calls(run_id):
        return [(t["function_name"], t["arguments"], _ext(s)["calls"][t["tool_call_id"]]["source_fields"]["language"])
                for s in _by_id()[run_id].steps for t in s.get("tool_calls", []) if s["extra"]["agentpile"]["source_fields"]["class_"] == "code_action"]

    assert calls("agenttuning_os/synth_os_0") == [
        ("execute_bash", {"command": "ls /synthetic | wc -l"}, "bash"),
        ("execute_bash", {"command": "echo done"}, "shell"),
        ("execute_bash", {"command": "true"}, "sh")]
    assert [(n, a, lang) for n, a, lang in calls("codeactinstruct/synth/tabular/q-1/model-x")] == [
        ("execute_ipython_cell", {"code": "print(len([1, 2, 3]))"}, "python"),
        ("execute_ipython_cell", {"code": "print(3)"}, "python3"),
        ("execute_ipython_cell", {"code": "print('ok')"}, "py")]
    assert calls("code_feedback/1")[1] == ("execute_code", {"language": "ruby", "content": "def double(x)\n  2 * x\nend"}, "ruby")
    assert calls("agenttuning_db/synth_db_0")[0][0] == "execute_code" and calls("agenttuning_db/synth_db_0")[0][2] == "mysql"


def test_observation_without_a_call_is_its_own_step():
    kg = _by_id()["agenttuning_kg/synth_kg_0"]
    ack = kg.steps[1]  # the item says source "agent"
    assert ack["source"] == "agent" and ack["message"] == "Understood." and _ext(ack)["role"] == "agent"
    assert "tool_calls" not in ack and "observation" not in ack
    cf = _by_id()["code_feedback/1"].steps
    assert cf[3]["source"] == "user" and _ext(cf[3])["role"] == "user"
    assert cf[6]["source"] == "user" and _ext(cf[6])["role"] == "environment"  # ATIF has no environment source
    assert _ext(cf[6])["source_fields"] == {"class_": "text_observation"}
    alf = _by_id()["agenttuning_alfworld/synth_alf_1"].steps
    assert [s["source"] for s in alf] == ["user", "agent", "user", "agent", "user", "agent"]  # two observations in a row
    assert alf[3]["observation"]["results"][0]["content"] == "Nothing new."
    assert alf[4]["message"] == "A second note from the environment."


def test_result_source_is_kept_as_the_item_gave_it():
    cf = _by_id()["code_feedback/1"].steps
    assert _ext(cf[1])["result_fields"][0]["source"] == "environment"
    assert _ext(cf[4])["result_fields"][0]["source"] == "user"  # ADP's own output relabels this one, agentpile does not


def test_action_with_no_observation_gets_a_step_based_id():
    steps = _by_id()["synatra/0"].steps
    assert [s["source"] for s in steps] == ["user", "user", "agent"]
    last = steps[2]
    assert last["tool_calls"][0]["tool_call_id"] == "call_000003" and "observation" not in last
    assert _ext(last)["calls"]["call_000003"] == {"id_generated": True, "source_ref": 2}
    assert "result_fields" not in _ext(last)
    edge = _by_id()["agenttuning_kg/synth_kg_edge"].steps
    assert edge[1]["tool_calls"][0]["tool_call_id"] == "call_000002u"  # call_000002 is taken by a linked pair
    ids = [t["tool_call_id"] for s in edge for t in s.get("tool_calls", [])]
    assert len(ids) == len(set(ids))


def test_web_observation_content_order_and_remaining_fields():
    go = _by_id()["go-browse-wa/0"].steps
    assert go[1]["source"] == "user" and go[1]["message"].startswith("RootWebArea 'Synthetic Map'")
    web = _ext(go[1])["source_fields"]
    assert web["class_"] == "web_observation" and web["content_from"] == "axtree"
    assert web["web"] == {"html": None, "url": None, "viewport_size": [1280, 1440],
                          "image_observation": {"class_": "image_observation", "content": "datasets/synthetic/screenshots/00000-00.png",
                                                "annotations": None, "source": "environment"}}
    assert _ext(go[2])["result_fields"][0]["web"]["viewport_size"] == [1280, 1440]
    assert go[2]["observation"]["results"][0]["content"].startswith("RootWebArea 'Synthetic Map route'")
    nn = _by_id()["nnetnav-wa/synthetic_openended_1"].steps
    fields = [_ext(s)["result_fields"][0] for s in nn if "observation" in s]
    assert [f["content_from"] for f in fields] == ["html", "url", None]  # html wins over axtree, then url, then nothing
    assert fields[0]["web"]["axtree"] == "RootWebArea 'Orders'" and "html" not in fields[0]["web"]
    assert [s["observation"]["results"][0]["content"] for s in nn if "observation" in s][2] == ""
    oh = _by_id()["openhands/1"].steps
    assert [s["source"] for s in oh] == ["user", "agent", "agent", "user", "agent", "agent"]  # web after web is a step


def test_details_outcome_task_tools_and_agent_block():
    runs = _by_id()
    swe = runs["swe-gym_openhands_sampled_trajectories/synth__repo-7_0"]
    assert swe.outcome == {"success": True, "score": None, "raw": {"resolved": "True"}}
    assert [d["function"]["name"] for d in swe.record["agent"]["tool_definitions"]] == ["execute_bash", "finish", "str_replace_editor"]
    assert {t: (v["name"], v["atif_index"], v["permission"], v["permission_from"]) for t, v in swe.tools.items()} == {
        "execute_bash": ("execute_bash", 0, "unknown", "unknown"), "finish": ("finish", 1, "unknown", "unknown"),
        "str_replace_editor": ("str_replace_editor", 2, "unknown", "unknown")}
    tool_ids = [cx.get("tool_id") for s in swe.steps for cx in _ext(s).get("calls", {}).values()]
    assert tool_ids == ["str_replace_editor", "execute_bash", "str_replace_editor"]
    assert isinstance(swe.dataset["details"]["tools"], str) and "test_result" in swe.dataset["details"]  # raw details stay whole
    assert swe.dataset["id"] == "synth__repo-7_0"
    nebius = runs["nebius_SWE-agent-trajectories/synth__repo-1_0"]
    assert nebius.record["agent"] == {"name": "unknown", "version": "unknown", "model_name": "synthetic-model-70b"}
    assert nebius.outcome == {"success": None, "score": None, "raw": {"exit_status": "submitted (exit_format)"}}
    assert runs["openhands/1"].outcome["raw"] == {"feedback": "positive", "polarity": ""}
    assert runs["openhands/1"].outcome["success"] is None and runs["openhands/1"].dataset["details"]["timestamp"] == "2000-01-01 00:00:00"
    assert runs["synatra/0"].task == {"id": None, "text": "Press the synthetic button"}
    assert runs["agenttuning_os/synth_os_0"].task == {"id": None, "text": None}
    assert all(r.record["agent"]["name"] == "unknown" for r in runs.values())


def test_adp_has_no_labels_and_no_failure_outcome():
    runs = _runs()
    assert all(r.labels == [] for r in runs)
    assert all(r.outcome["success"] in (None, True) for r in runs)  # the std files hold no failure value


def test_empty_and_odd_rows_keep_what_the_source_gave():
    runs = _by_id()
    empty = runs["agenttuning_kg/synth_kg_empty"]
    assert [s["source"] for s in empty.steps] == ["system"] and _ext(empty.steps[0]) == {"message_state": "missing", "empty_source": True}
    edge = runs["agenttuning_kg/synth_kg_edge"].steps
    raw_args = edge[3]
    assert raw_args["tool_calls"][0]["arguments"] == {} and _ext(raw_args)["calls"]["call_000001"]["arguments_raw"] == ["not", "an", "object"]
    assert raw_args["observation"]["results"][0]["content"] == ""
    assert _ext(raw_args)["result_fields"][0]["content_state"] == "null"
    assert [e.content for e in runs["agenttuning_kg/synth_kg_edge"].events() if e.kind == "call"][1] == ["not", "an", "object"]
    odd_name = edge[4]
    assert odd_name["tool_calls"][0]["function_name"] == "" and _ext(odd_name)["calls"]["call_000002"]["function_name_raw"] == 7
    assert _ext(odd_name)["result_fields"][0]["name"] == "named-source"
    assert _ext(edge[5])["result_fields"][0]["content_state"] == "missing" and _ext(edge[5])["message_state"] == "missing"
    assert edge[6]["message"] == '{"unexpected": "object"}' and _ext(edge[6])["source_fields"]["content"] == {"unexpected": "object"}
    odd_result = edge[7]  # a non-string result keeps the source value for the event view
    assert odd_result["tool_calls"][0]["tool_call_id"] == "call_000004"
    assert odd_result["observation"]["results"][0]["content"] == '{"nested": 1}'
    assert _ext(odd_result)["result_fields"][0]["content_raw"] == {"nested": 1}
    assert [e.content for e in runs["agenttuning_kg/synth_kg_edge"].events() if e.kind == "result"][-1] == {"nested": 1}


def _convert(raw):
    return adp_mod.dataset.convert("c/x", raw, {"file": "c/full_std.jsonl", "config": "c", "split": "std", "row": 0})


def test_tools_string_is_used_only_when_it_is_a_list_of_named_definitions():
    item = {"class_": "text_observation", "content": "t", "name": "Alice", "source": "user"}
    for bad in ("not json", "[1, 2]", "[]", '[{"function": {"description": "no name"}}]', 7):
        run = _convert({"id": "x", "content": [item], "details": {"tools": bad}})
        assert "tool_definitions" not in run.record["agent"] and run.tools == {} and run.dataset["details"]["tools"] == bad
    twin = json.dumps([{"function": {"name": "a"}, "type": "function"}, {"name": "a"}])
    run = _convert({"id": "x", "content": [item], "details": {"tools": twin}})
    assert {t: v["atif_index"] for t, v in run.tools.items()} == {"a": 0, "a#1": 1} and ap.validate(run) == []
    assert _ext(run.steps[0])["name"] == "Alice"  # a named observation keeps its name


def test_only_a_true_or_false_resolved_flag_sets_the_outcome():
    item = {"class_": "text_observation", "content": "t", "name": None, "source": "user"}
    got = {repr(v): _convert({"id": "x", "content": [item], "details": {"resolved": v}}).outcome
           for v in ("True", "False", True, "maybe", ["True"])}
    assert [o["success"] for o in got.values()] == [True, False, True, None, None]
    assert got["['True']"]["raw"] == {"resolved": ["True"]}  # the source value is echoed whatever it is


def test_unknown_item_class_is_refused_not_guessed():
    raw = {"id": "x", "content": [{"class_": "image_observation", "content": "a.png"}], "details": {}}
    with pytest.raises(ValueError, match="unsupported class_ 'image_observation'"):
        _convert(raw)


def test_blank_lines_are_skipped_and_bad_lines_are_named(tmp_path):
    good = json.dumps({"id": "a", "content": [{"class_": "text_observation", "content": "t", "name": None, "source": "user"}], "details": {}})
    path = tmp_path / "agenttuning_mind2web" / "full_std.jsonl"
    path.parent.mkdir()
    path.write_text(good + "\n\n   \n" + good.replace('"a"', '"b"') + "\n", encoding="utf-8")
    runs = list(ap.load("adp", root=str(tmp_path), config="agenttuning_mind2web"))
    assert [r.id for r in runs] == ["agenttuning_mind2web/a", "agenttuning_mind2web/b"] and [r.source["row"] for r in runs] == [0, 1]
    path.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not JSON"):
        list(ap.load("adp", root=str(tmp_path), config="agenttuning_mind2web"))
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not a JSON object"):
        list(ap.load("adp", root=str(tmp_path), config="agenttuning_mind2web"))


def test_manifest_lists_supported_and_unsupported_configs():
    m = ap.datasets.get("adp").manifest
    supported = {c["config"]: c for c in m["supported_configs"]}
    assert sum(c["rows"] for c in supported.values()) == m["counts"]["runs_after_load"]["value"] == 200914
    assert m["allow_patterns"] == [c["file"] for c in m["supported_configs"]]
    assert "count_note" in m["counts"]["runs_after_load"]
    unsupported = {u["config"] for u in m["unsupported"] if "config" in u}
    assert {"mind2web", "nnetnav-live", "orca_agentinstruct", "swe-smith"} <= unsupported and not unsupported & set(supported)
    assert all(u["reason"] for u in m["unsupported"])
    for c in supported.values():  # the fixture mirrors the pinned layout
        assert (ROOT / c["file"]).is_file()


def test_harbor_accepts_every_fixture_run():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    data = os.environ.get("AGENTPILE_DATA")
    root = str(Path(data) / "adp") if data else None
    manifest = ap.datasets.get("adp").manifest
    per_config = collections.Counter()
    calls = results = steps = 0
    n = 0
    for run in ap.load("adp", root=root, verify=True):
        n += 1
        per_config[run.source["config"]] += 1
        assert ap.validate(run) == [] and run.labels == []
        steps += len(run.steps)
        for s in run.steps:
            calls += len(s.get("tool_calls") or [])
            results += len((s.get("observation") or {}).get("results") or [])
    assert n == 200914 == manifest["counts"]["runs_after_load"]["value"]
    assert per_config == {c["config"]: c["rows"] for c in manifest["supported_configs"]}
    assert (steps, calls, results) == (1068940, 549187, 449266)


def test_extension_tools_hold_the_declared_definitions():
    swe = _by_id()["swe-gym_openhands_sampled_trajectories/synth__repo-7_0"]
    defs = swe.record["agent"]["tool_definitions"]
    assert len(swe.tools) == len(defs) == 3
    for entry in swe.tools.values():
        assert entry["definition"] == defs[entry["atif_index"]] and entry["definition"] is not defs[entry["atif_index"]]
    assert ap.validate(swe) == []
    broken = swe.to_dict()
    next(iter(broken["extra"]["agentpile"]["tools"].values()))["definition"] = None
    assert any("differs from agent.tool_definitions[0]" in e for e in ap.validate(broken))  # the old reader's output


def test_a_call_links_to_a_tool_only_when_one_declared_tool_has_its_name():
    def action(name):
        return [{"class_": "api_action", "function": name, "kwargs": {}, "description": "d"},
                {"class_": "text_observation", "content": "o", "name": None, "source": "environment"}]

    tools = json.dumps([{"function": {"name": "a"}, "type": "function"}, {"name": "a"}, {"function": {"name": "b"}}])
    run = _convert({"id": "x", "content": action("a") + action("b") + action("c"), "details": {"tools": tools}})
    assert {t: v["atif_index"] for t, v in run.tools.items()} == {"a": 0, "a#1": 1, "b": 2}
    linked = {t["function_name"]: _ext(s)["calls"][t["tool_call_id"]].get("tool_id")
              for s in run.steps for t in s.get("tool_calls", [])}
    assert linked == {"a": None, "b": "b", "c": None}  # a is declared twice, c is not declared
    assert ap.validate(run) == []


def test_api_action_arguments_absent_null_and_empty_stay_different():
    base = {"class_": "api_action", "function": "f", "description": "d"}
    obs = {"class_": "text_observation", "content": "o", "name": None, "source": "environment"}
    items = [dict(base), obs, dict(base, kwargs=None), obs, dict(base, kwargs={}), obs, dict(base, kwargs=["x"]), obs]
    run = _convert({"id": "x", "content": items, "details": {}})
    args = [e.content for e in run.events() if e.kind == "call"]
    assert args[0] is ap.MISSING and args[1] is None and args[2] == {} and args[3] == ["x"]
    calls = [_ext(s)["calls"][s["tool_calls"][0]["tool_call_id"]] for s in run.steps if "tool_calls" in s]
    assert calls[0]["arguments_state"] == "missing" and "arguments_raw" not in calls[0]
    assert calls[1]["arguments_raw"] is None and "arguments_state" not in calls[1]
    assert "arguments_raw" not in calls[2] and "arguments_state" not in calls[2]
    assert calls[3]["arguments_raw"] == ["x"]
    assert all(s["tool_calls"][0]["arguments"] == {} for s in run.steps if "tool_calls" in s)
    assert ap.validate(run) == []
    assert [e.content for e in ap.Run(json.loads(run.to_json())).events() if e.kind == "call"] == args
