"""AgentRewardBench loader on a synthetic copy of the pinned file layout (no dataset content)."""

import json
import os
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.corpora._base import CountMismatch
from agentpile.corpora.agent_reward_bench import parse_action, read_run_file

ROOT = Path(__file__).parent / "fixtures" / "agent_reward_bench"
ONE = "webarena/GenericAgent-synth-llama/webarena.1"
TWO = "webarena/GenericAgent-synth-llama/webarena.2"
RETRY = "visualwebarena/GenericAgent-synth-gpt/visualwebarena.resized.5"
LIST_GOAL = "workarena/GenericAgent-synth-claude/workarena.servicenow.order-synth-laptop"
EMPTY = "assistantbench/GenericAgent-synth-llama/assistantbench.improved.validation.0"


def _runs(**kw):
    return {r.id: r for r in ap.load("agent_reward_bench", root=str(ROOT), **kw)}


def _ext(step):
    return step["extra"]["agentpile"]


def _agent_steps(run):
    return [s for s in run.steps if s["source"] == "agent"]


def test_files_become_runs_in_path_order():
    runs = _runs()
    assert list(runs) == [EMPTY, RETRY, ONE, TWO, LIST_GOAL]
    assert all(ap.validate(r) == [] for r in runs.values())
    assert list(_runs(benchmark="webarena")) == [ONE, TWO]
    assert list(_runs(benchmark=["workarena", "assistantbench"])) == [EMPTY, LIST_GOAL]
    with pytest.raises(ValueError):
        _runs(benchmark="miniwob")
    with pytest.raises(TypeError):
        _runs(split="dev")


def test_system_user_and_agent_steps():
    run = _runs()[ONE]
    assert [s["source"] for s in run.steps] == [
        "system", "user", "user", "agent", "system", "user", "agent", "system", "user", "agent"]
    assert run.steps[0]["message"].startswith("You are a synthetic agent")
    assert run.steps[1]["message"] == "Find the price of the synthetic mug"
    assert run.task == {"id": "webarena.1", "text": "Find the price of the synthetic mug"}
    assert run.record["agent"] == {"name": "GenericAgent-synth-llama", "version": "unknown", "model_name": "synth/llama"}
    assert [s["message"] for s in _agent_steps(run)] == [
        "I need to open the mugs page.", "<action>\nfill('7', \"it's a 'mug'\")\n</action>", "Scroll down to see more."]
    assert {_ext(s)["actor"] for s in _agent_steps(run)} == {"GenericAgent-synth-llama"}
    assert run.tools == {} and run.agents[0]["model"] == "synth/llama"
    assert run.source["file"].endswith("webarena.1.json") and run.source["config"] == "webarena"
    assert run.source["record_key"] == ONE and run.source["revision"] == ap.datasets.get("agent_reward_bench").revision


def test_initial_observation_sits_on_the_user_step():
    run = _runs()[ONE]
    user = run.steps[1]
    fields = _ext(user)["source_fields"]
    assert fields["observation"]["url"] == "https://shop.example/" and fields["observation"]["num"] == 0
    assert fields["observation"]["axtree"].startswith("RootWebArea 'Shop 0'")
    assert "observation" not in user and "tool_calls" not in user and "prompt_text" not in fields
    # the first prompt is a step of its own: its text part in ATIF, the whole content (image part included) kept
    prompt = run.steps[2]
    assert prompt["source"] == "user" and _ext(prompt)["source_ref"] == {"record": 0, "chat_message": 1}
    assert prompt["message"][0]["text"].startswith("# Instructions") and "# Action space:" in prompt["message"][0]["text"]
    assert prompt["message"][1] == {"type": "text", "text": "[image_url part: original in agentpile extension]"}
    assert _ext(prompt)["message_raw"][1] == {"type": "image_url", "image_url": {"url": "REMOVED"}}


def test_every_model_call_keeps_its_own_inputs(tmp_path):
    """A later prompt and a changed system message survive loading, the event view and serialization."""
    run = _runs()[ONE]
    raw = json.loads(next(ROOT.glob("cleaned/webarena/*/*/webarena.1.json")).read_text(encoding="utf-8"))
    chat1 = raw["steps"][1]["chat_messages"]
    second = _agent_steps(run)[1]
    held = {i["chat_message"]: run.steps[i["step"] - 1] for i in _ext(second)["inputs"]}
    assert held[0]["source"] == "system" and held[0]["message"] == chat1[0]["content"]
    assert "Second-call system rule" in held[0]["message"]
    assert held[1]["message"][0] == {"type": "text", "text": chat1[1]["content"][0]["text"]}
    assert "UNIQUE-SECOND-CALL-PROMPT-7f3a" in held[1]["message"][0]["text"]
    assert _ext(held[1])["message_raw"] == chat1[1]["content"]
    # the third call goes back to the first system message: a new system step, since it differs from the latest
    third = _agent_steps(run)[2]
    system3 = run.steps[_ext(third)["inputs"][0]["step"] - 1]
    assert system3["step_id"] == 8 and system3["message"] == raw["steps"][2]["chat_messages"][0]["content"]
    messages = [json.dumps(e.content) for e in run.events() if e.kind == "message"]
    assert any("Second-call system rule" in m for m in messages)
    assert any("UNIQUE-SECOND-CALL-PROMPT-7f3a" in m for m in messages)
    path = tmp_path / "run.json"
    path.write_text(run.to_json(), encoding="utf-8")
    back = next(ap.load(str(path)))
    assert back.record == run.record and ap.validate(back) == []
    otlp = tmp_path / "run.otlp.json"
    otlp.write_text(ap.writers.get("otlp-json").dumps(run), encoding="utf-8")
    assert next(ap.load(str(otlp))).record == run.record
    # every input event gives the whole source content of its chat message, image parts included
    for view in (run, back):
        content = {e.address["step"]: e.content for e in view.events() if e.kind == "message"}
        for agent_step in _agent_steps(view):
            chat = raw["steps"][_ext(agent_step)["source_ref"]]["chat_messages"]
            for i in _ext(agent_step)["inputs"]:
                assert content[i["step"]] == chat[i["chat_message"]]["content"]


def test_a_repeated_system_message_reuses_its_step():
    run = _runs()[TWO]
    assert [s["source"] for s in run.steps].count("system") == 1
    assert [i for s in _agent_steps(run) for i in _ext(s)["inputs"] if i["chat_message"] == 0] == [
        {"chat_message": 0, "step": 1}] * 6
    assert [_ext(run.steps[i["step"] - 1])["source_ref"] for s in _agent_steps(run)
            for i in _ext(s)["inputs"] if i["chat_message"] == 1] == [{"record": j, "chat_message": 1} for j in range(6)]


def test_actions_become_calls_with_generated_ids():
    run = _runs()[ONE]
    calls = [s["tool_calls"][0] for s in _agent_steps(run)]
    assert [(c["tool_call_id"], c["function_name"], c["arguments"]) for c in calls] == [
        ("ap-4-0", "click", {"arg0": "12"}),
        ("ap-7-0", "fill", {"arg0": "7", "arg1": "it's a 'mug'"}),
        ("ap-10-0", "scroll", {"arg0": 0, "arg1": 500})]
    cx = _ext(_agent_steps(run)[1])["calls"]["ap-7-0"]
    assert cx["id_generated"] is True and cx["arguments_raw"] == "fill('7', \"it's a 'mug'\")"


def test_result_is_the_next_records_observation():
    run = _runs()[ONE]
    first, second, third = _agent_steps(run)
    assert first["observation"]["results"] == [{
        "source_call_id": "ap-4-0", "content": "RootWebArea 'Shop 1', focused\n\t[12] link 'Mugs ☕ café', clickable, visible\n"
                                              "\t[7] searchbox 'Search', visible"}]
    assert _ext(first)["result_refs"] == [1]
    rf = _ext(first)["result_fields"][0]
    assert rf["paired_by"] == "next_record" and rf["num"] == 1 and rf["url"] == "https://shop.example/"
    assert rf["screenshot_path"].endswith("step_1.png") and "axtree" not in rf
    # the final record only supplies the last observation
    assert third["observation"]["results"][0]["content"].startswith("RootWebArea 'Shop 3'")
    assert _ext(third)["result_fields"][0]["url"] == "https://shop.example/mugs"
    assert len(run.steps) == 10
    # last_action_error of record i + 1 is the error of action i
    assert _ext(second)["calls"]["ap-7-0"]["error"] == "TimeoutError: fill did not finish"
    assert "error" not in _ext(first)["calls"]["ap-4-0"] and "error" not in _ext(third)["calls"]["ap-10-0"]


def test_events_follow_source_order():
    events = list(_runs()[ONE].events())
    turn = ["message", "call", "result"]
    assert [e.kind for e in events] == ["message"] * 3 + turn + ["message"] * 2 + turn + ["message"] * 2 + turn
    call = events[4]
    assert call.name == "click" and call.content == "click('12')" and call.call_id == "ap-4-0"
    assert events[5].call_id == "ap-4-0" and events[5].address == {"trajectory": ONE, "step": 4, "result": 0}
    assert events[10].error == "TimeoutError: fill did not finish"


def test_call_id_dependencies_follow_the_next_record_pairing():
    deps = ap.derive_deps(_runs()[ONE])
    assert [(d["src"]["step"], d["src"]["call"], d["dst"]["step"]) for d in deps] == [(4, "ap-4-0", 7), (7, "ap-7-0", 10)]


def test_odd_actions_keep_everything():
    run = _runs()[TWO]
    calls = {s["tool_calls"][0]["tool_call_id"]: s for s in _agent_steps(run)}
    assert [s["tool_calls"][0]["function_name"] for s in _agent_steps(run)] == [
        "click", "fill", "noop", "go_back", "", "keyboard_press"]
    # a call followed by an HTML comment parses, and the comment is kept
    cx = _ext(calls["ap-4-0"])["calls"]["ap-4-0"]
    assert calls["ap-4-0"]["tool_calls"][0]["arguments"] == {"arg0": "3"}
    assert cx["source_fields"] == {"trailing": "<!-- the model added a comment -->"}
    assert cx["arguments_raw"] == "click('3')  <!-- the model added a comment -->"
    # a non-literal argument leaves arguments empty, flagged
    cx = _ext(calls["ap-6-0"])["calls"]["ap-6-0"]
    assert calls["ap-6-0"]["tool_calls"][0]["arguments"] == {} and cx["source_fields"] == {"arguments_parsed": False}
    assert cx["arguments_raw"] == "fill('4', some_variable)"
    assert calls["ap-8-0"]["tool_calls"][0]["arguments"] == {"wait_ms": 2000}
    assert calls["ap-10-0"]["tool_calls"][0]["arguments"] == {} and "source_fields" not in _ext(calls["ap-10-0"])["calls"]["ap-10-0"]
    # text with no call keeps the whole string as the raw function name
    cx = _ext(calls["ap-12-0"])["calls"]["ap-12-0"]
    assert cx["function_name_raw"] == "this is not a call" and cx["source_fields"] == {"arguments_parsed": False}
    assert _ext(calls["ap-14-0"])["calls"]["ap-14-0"]["source_fields"] == {"arguments_parsed": False}
    # the run crashed before a final record, so the last call has no result
    assert "observation" not in run.steps[-1] and "result_refs" not in _ext(run.steps[-1])
    assert run.dataset["summary_info"]["err_msg"] == "KeyboardInterrupt: stop"


def test_retry_turns_and_a_reply_without_an_action():
    run = _runs()[RETRY]
    first, last = _agent_steps(run)
    assert first["message"] == "Try again." and first["tool_calls"][0]["arguments"] == {"arg0": "9"}
    tail = _ext(first)["source_fields"]["chat_messages_tail"]
    assert [m["role"] for m in tail] == ["assistant", "user", "assistant"] and tail[0]["content"] == "no action here"
    # the last record has a reply but no action: an agent step without a call, not dropped
    assert last["source"] == "agent" and "tool_calls" not in last and last["message"] == "I cannot decide."
    fields = _ext(last)["source_fields"]
    assert fields["action"] is None and fields["message_from"] == "last_assistant_reply"
    assert len(fields["chat_messages_tail"]) == 4
    # the actionless record's observation is still the result of the call before it
    assert first["observation"]["results"][0]["content"].startswith("RootWebArea 'Shop 1'")


def test_labels_per_annotation_row():
    run = _runs()[ONE]
    assert [(l["name"], l["value"], l["label_source"]) for l in run.labels] == [
        ("trajectory_success", "Successful", "expert:A"), ("trajectory_side_effect", "No", "expert:A"),
        ("trajectory_optimality", "4. Completely Optimal", "expert:A"), ("trajectory_looping", "No", "expert:A")]
    assert run.labels[0]["target"] == {"kind": "run", "address": {"trajectory": ONE}}
    assert run.labels[0]["source_ref"] == {"file": "data/annotations.csv", "row": 0,
                                           "field": "trajectory_success", "annotator_name": "A"}
    two = _runs()[TWO]  # two annotators: " H" keeps its raw spelling in source_ref only
    assert [l["label_source"] for l in two.labels] == ["expert:B"] * 4 + ["expert:H"] * 4
    assert two.labels[4]["source_ref"]["annotator_name"] == " H" and two.labels[4]["source_ref"]["row"] == 2
    assert [row["row"] for row in two.dataset["annotation_rows"]] == [1, 2]


def test_outcome_follows_the_first_annotation_row():
    runs = _runs()
    assert runs[ONE].outcome["success"] is True and runs[ONE].outcome["score"] == 1.0
    assert runs[TWO].outcome["success"] is None  # the first row says Unsure
    assert runs[TWO].outcome["raw"]["trajectory_success"] == [
        {"annotator_name": "B", "value": "Unsure"}, {"annotator_name": " H", "value": "Successful"}]
    retry = runs[RETRY].outcome  # first row Successful, second Unsuccessful: both stay in raw
    assert retry["success"] is True and retry["score"] == 0.0 and retry["raw"]["cum_reward"] == 0.0
    assert [v["value"] for v in retry["raw"]["trajectory_success"]] == ["Successful", "Unsuccessful"]
    assert runs[EMPTY].outcome["success"] is False
    unlabeled = runs[LIST_GOAL]  # a run file without an annotation row
    assert unlabeled.labels == [] and unlabeled.outcome["success"] is None
    assert unlabeled.outcome["raw"]["trajectory_success"] == []


def test_header_fields_kept_and_browser_dumps_dropped():
    run = _runs()[ONE]
    d = run.dataset
    assert d["benchmark"] == "webarena" and d["seed"] == 0 and d["summary_info"]["cum_reward"] == 1.0
    assert d["flags"]["action"]["action_set"]["multiaction"] is False and "goal" not in d and "agent" not in d
    assert d["annotation_rows"][0]["exp_name"] == "GenericAgent-synth-llama_on_webarena"
    text = run.to_json()
    for dump in ("nodeId", "bbox", "visibility", "link 'Mugs'"):
        assert dump not in text
    assert "steps[].axtree_obj" in run.source["lost_fields"]
    assert not any("chat_messages" in f for f in run.source["lost_fields"])  # every prompt is kept


def test_degenerate_runs():
    empty = _runs()[EMPTY]  # no goal and no records: one marked stand-in step, as in the shared parser
    assert len(empty.steps) == 1 and _ext(empty.steps[0]) == {"message_state": "missing", "empty_source": True}
    assert empty.task == {"id": "assistantbench.improved.validation.0", "text": None}
    run = _runs()[LIST_GOAL]  # a goal that is not a string, a run with no chat, a record with no reasoning key
    assert [s["source"] for s in run.steps] == ["user", "agent", "agent"]
    assert json.loads(run.steps[0]["message"]) == [{"type": "text", "text": "Order the synthetic laptop"}]
    assert _ext(run.steps[0])["source_fields"]["goal"] == [{"type": "text", "text": "Order the synthetic laptop"}]
    assert "prompt_text" not in _ext(run.steps[0])["source_fields"] and run.task["text"] is None
    assert run.steps[2]["message"] == "" and _ext(run.steps[2])["message_state"] == "missing"
    assert run.steps[2]["tool_calls"][0]["arguments"] == {"arg0": "2", "arg1": ["a", "b"]}
    assert _ext(run.steps[1]).get("source_fields", {}).get("chat_messages_tail") is None


def test_observations_without_a_call_to_answer_are_kept():
    record = {"head": {"goal": "g", "agent": "a"}, "annotations": [],
              "steps": [{"num": 0, "reasoning": None, "action": None, "axtree": "first", "url": "u0"},
                        {"num": 1, "reasoning": "r", "action": "click('1')", "axtree": "second", "url": "u1"},
                        {"num": 2, "reasoning": None, "action": None, "axtree": None, "url": "u2"}]}
    run = ap.datasets.get("agent_reward_bench").convert("k", record, {"file": "f", "task_id": "t"})
    assert ap.validate(run) == []
    assert [s["source"] for s in run.steps] == ["user", "agent"]  # no chat, so no system step
    assert _ext(run.steps[0])["source_fields"]["observation"]["axtree"] == "first"  # record 0: nothing to answer
    unpaired = run.dataset["unpaired_observations"]  # record 1 follows a record without a call
    assert [(u["record"], u["axtree"], u["url"]) for u in unpaired] == [(1, "second", "u1")]
    assert _ext(run.steps[1])["result_fields"][0]["content_state"] == "null"
    assert run.steps[1]["observation"]["results"][0]["content"] == ""


@pytest.mark.parametrize("action,expected", [
    ("click('63')", ("click", {"arg0": "63"}, True, "")),
    ("scroll(0, -500)", ("scroll", {"arg0": 0, "arg1": -500}, True, "")),
    ("noop()", ("noop", {}, True, "")),
    ("click('1', button='right', modifiers=['Shift'])", ("click", {"arg0": "1", "button": "right", "modifiers": ["Shift"]}, True, "")),
    ("send_msg_to_user(\"it's done (really)\")", ("send_msg_to_user", {"arg0": "it's done (really)"}, True, "")),
    ('send_msg_to_user("""a ) b""")', ("send_msg_to_user", {"arg0": "a ) b"}, True, "")),
    ("fill('1', 'a\\'b')", ("fill", {"arg0": "1", "arg1": "a'b"}, True, "")),
    ("click('708')  <!-- Assuming 'x' ) -->", ("click", {"arg0": "708"}, True, "<!-- Assuming 'x' ) -->")),
    ("mouse_click(1, 2)\nmouse_click(3, 4)", ("mouse_click", {"arg0": 1, "arg1": 2}, True, "mouse_click(3, 4)")),
    ("fill('a', x)", ("fill", {}, False, "")),
    ("click(*ids)", ("click", {}, False, "")),
    ("click(a=1, a=2)", ("click", {}, False, "")),
    ("click(b'x')", ("click", {}, False, "")),
    ("click('1'", ("click", {}, False, "")),
    ("click('1'])", ("click", {}, False, "")),
    ("plain words", None),
    ("", None),
])
def test_parse_action(action, expected):
    assert parse_action(action) == expected


def test_streaming_reader_matches_json_load_at_any_chunk_size():
    for path in sorted(ROOT.glob("cleaned/*/*/*/*.json")):
        whole = json.loads(path.read_text(encoding="utf-8"))
        want_head = {k: v for k, v in whole.items() if k != "steps"}
        for chunk in (1, 7, 100, 1 << 22):
            head, steps = read_run_file(path, drop=(), chunk=chunk)
            assert head == want_head and steps == whole["steps"]
            assert list(head) == list(want_head)
        _, trimmed = read_run_file(path)
        assert all("axtree_obj" not in s and "axtree_pruned" not in s for s in trimmed)


def test_streaming_reader_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"goal": "g", "steps": [{"num": 0}, {"num": 1', encoding="utf-8")
    with pytest.raises(ValueError):
        read_run_file(path)


def test_missing_annotation_file_is_an_error(tmp_path):
    (tmp_path / "cleaned").mkdir()
    with pytest.raises(FileNotFoundError):
        list(ap.load("agent_reward_bench", root=str(tmp_path)))


def test_manifest_pins_the_run_files_and_the_annotation_csv():
    ds = ap.datasets.get("agent_reward_bench")
    assert ds.repo_id == "McGill-NLP/agent-reward-bench" and ds.revision == "b6d17e646009d6cb63d5dd7be78807b680693f61"
    assert ds.allow_patterns == ["data/annotations.csv", "cleaned/*.json"]  # no screenshots/ and no judgments/
    assert ds.expected_runs == 1302
    with pytest.raises(CountMismatch):
        list(ap.load("agent_reward_bench", root=str(ROOT), verify=True))


def test_harbor_accepts_every_run():
    trajectories = pytest.importorskip("harbor.models.trajectories")
    for run in _runs().values():
        trajectories.Trajectory.model_validate(run.record)


@pytest.mark.network
def test_full_pinned_dataset():
    # 1,302 run files, 30 GB: without AGENTPILE_DATA/agent_reward_bench this downloads them into the cache
    data = os.environ.get("AGENTPILE_DATA")
    root = Path(data) / "agent_reward_bench" if data else None
    if root is not None and not root.is_dir():
        pytest.skip(f"no local copy at {root}")
    n, labelled = 0, 0
    for run in ap.load("agent_reward_bench", root=str(root) if root else None, verify=True):
        assert ap.validate(run) == [], run.id
        n += 1
        labelled += bool(run.labels)
    assert n == 1302 and labelled == 1302
