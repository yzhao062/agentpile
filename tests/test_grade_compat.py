"""Gate 2: agentpile reproduces GRADE's trace converters through load -> derive_deps -> projection.

The expected steps are GRADE's own output (commit 3839a57, and the frozen tau2 rules ``tau2-frozen-v1``), made
independently of agentpile. The offline tests run the hand-checked samples, the ScienceWorld fixture and two
synthetic records whose expected steps GRADE's converters produced. The network tests (``-m network``) compare every
run of every expected file, step by step, using the pinned data under ``AGENTPILE_DATA`` and the expected files under
``AGENTPILE_GRADE_EXPECTED``. A difference that is not listed in ``fixtures/grade/exceptions.json`` fails.

    AGENTPILE_DATA=<dir with one folder per dataset> AGENTPILE_GRADE_EXPECTED=<dir with expected/*.jsonl> \\
        pytest -m network tests/test_grade_compat.py -s

A difference that the exception list names is a behaviour of GRADE that agentpile does not copy, or a defect of an
expected file: the causes are in the ``causes`` block of the list and in GATE2-NOTES.md of the expectations work.
"""

import json
import os
from pathlib import Path

import pytest

import agentpile as ap
from agentpile import deps as D
from agentpile.formats.messages import messages_to_trajectory
from compat import grade as G

FIX = Path(__file__).parent / "fixtures" / "grade"
EXCEPTIONS = FIX / "exceptions.json"


def _sample(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _openhands_run(raw):
    """``convert`` is what ``load`` calls on each Parquet row; calling it directly keeps the offline test free of pyarrow."""
    where = {"file": "trajectories.parquet", "config": "default", "split": "train", "row": 0, "tool_definitions": []}
    return ap.datasets.get("swe_rebench_openhands").convert(raw["trajectory_id"], raw, where)


def _scienceworld_run(fixture_id, raw):
    """No ScienceWorld reader exists: each conversation turn becomes one message, the action text stays in it."""
    roles = {"gpt": "assistant", "human": "user"}
    msgs = []
    for turn in raw["conversations"]:
        role = (turn.get("role") or turn.get("from") or "").lower()
        msgs.append({"role": roles.get(role, role), "content": turn.get("content") or turn.get("value") or ""})
    ext = {"source": {"format": "scienceworld-fixture", "record_key": fixture_id, "reader": "test",
                      "file": f"fixtures/{fixture_id}.json", "lost_fields": []}}
    return ap.Run(messages_to_trajectory(msgs, trajectory_id=fixture_id, ext=ext))


def _swe_agent_run(raw):
    """The SWE-agent reader's own conversion: later ``user`` turns become results of the agent step before them."""
    where = {"file": "data/train-00000-of-00012.parquet", "config": "default", "split": "train"}
    return ap.datasets.get("swe_agent_nebius").convert("train-00000-of-00012:0", raw, where)


def _swe_agent_flat_run(raw):
    """The same turns as flat messages (every user turn a user step), the other layout the extractor accepts."""
    roles = {"ai": "assistant"}
    msgs = [{"role": roles.get(t["role"], t["role"]), "content": t.get("text") or ""} for t in raw["trajectory"]]
    return ap.Run(messages_to_trajectory(msgs, trajectory_id=raw["instance_id"]))


def _arb_records(raw):
    """Run-file records of a sample: its own ``records``, or one record for each (action, url) of its ``steps``.

    A step with an empty action becomes a record whose agent reply holds no action, as GRADE reads an empty action
    for it. The last step of such a sample has a call, so its run has no observation after the last action.
    """
    if "records" in raw:
        return raw["records"]
    recs = []
    for i, st in enumerate(raw["steps"]):
        rec = {"num": i, "action": st["action"] or None, "url": st["url"], "reasoning": "r", "axtree": "page text"}
        if not st["action"]:
            rec["reasoning"], rec["chat_messages"] = None, [{"role": "assistant", "content": "no action"}]
        recs.append(rec)
    return recs


def _arb_run(raw):
    """The AgentRewardBench reader's own conversion of run-file records."""
    key = f"{raw['benchmark']}/{raw['model_name']}/{raw['task_id']}"
    where = {"file": f"cleaned/{raw['benchmark']}/{raw['model_name']}/run/{raw['task_id']}.json",
             "config": raw["benchmark"], "split": "full", "task_id": raw["task_id"]}
    head = raw.get("head") or {"goal": "Find shoes.", "agent": raw["model_name"], "model": "demo-model"}
    return ap.datasets.get("agent_reward_bench").convert(
        key, {"head": head, "steps": _arb_records(raw), "annotations": []}, where)


def _steps(corpus, run):
    return G.project_run(corpus, run).steps


# -- offline: samples, fixtures, synthetic records --------------------------------------------------------

@pytest.mark.parametrize("name", ["a_openhands_event_order", "b_openhands_multi_call_turn"])
def test_openhands_samples(name):
    sample = _sample(name)
    assert _steps("swe_rebench_openhands", _openhands_run(sample["raw"])) == sample["expected_steps_hand"]


@pytest.mark.parametrize("name", ["a_tau_bench_event_order", "b_tau_bench_multi_call_turn"])
def test_tau_bench_samples(name, tmp_path):
    sample = _sample(name)
    _jsonl(tmp_path / "gpt-4.1.jsonl", [sample["raw"]])
    (run,) = ap.load("tau_bench", root=str(tmp_path))
    assert _steps("tau_bench", run) == sample["expected_steps_hand"]


def test_tau2_samples_follow_the_frozen_rules(tmp_path):
    sample = _sample("c_tau2_parallel_pairing")
    _jsonl(tmp_path / "sample-model.jsonl", [r["raw"] for r in sample["records"]])
    runs = list(ap.load("tau2_bench", root=str(tmp_path)))
    assert len(runs) == 3
    for rec, run in zip(sample["records"], runs):
        assert _steps("tau2_bench", run) == rec["expected_frozen_hand"], rec["id"]


def test_tau2_sample_c1_pairs_by_position_and_c3_by_id(tmp_path):
    sample = _sample("c_tau2_parallel_pairing")
    _jsonl(tmp_path / "sample-model.jsonl", [r["raw"] for r in sample["records"]])
    c1, c2, c3 = ap.load("tau2_bench", root=str(tmp_path))
    flags = lambda run: [f.get("paired_by") for s in run.steps  # noqa: E731
                         for f in s["extra"]["agentpile"].get("result_fields", [])]
    assert "position" in flags(c1) and "position" in flags(c2) and "position" not in flags(c3)
    edges = [(d["src"]["call"], d["dst"]["call"], d["resource"]) for d in ap.derive_deps(c1, rules=["value_flow_v1"])]
    assert edges == [("call_0_a", "call_2_c", ["value:L1001"]), ("call_2_c", "call_3_d", ["value:D1001"])]
    edges = [(d["src"]["call"], d["dst"]["call"]) for d in ap.derive_deps(c3, rules=["value_flow_v1"])]
    assert edges == [("call_0_a", "call_2_c")]  # by id: the first call's result arrived second


@pytest.mark.parametrize("n", [0, 1])
def test_scienceworld_fixture(n):
    sample = _sample("scienceworld_fixture")
    fixture = sample["fixtures"][n]
    expected = G.load_expected(FIX / "scienceworld_fixture.expected.jsonl")[n]
    assert expected["steps"] == fixture["expected_steps_hand"]  # the stored GRADE output equals the hand value
    run = _scienceworld_run(fixture["fixture_id"], fixture["raw"])
    assert _steps("scienceworld", run) == expected["steps"]
    src = expected["source"]
    assert G.grade_run_key("scienceworld", src["file"], src["row"], src["ids"].values()) == expected["run_key"]


@pytest.mark.parametrize("build", [_swe_agent_run, _swe_agent_flat_run])
def test_swe_agent_extractor_and_projection_on_a_synthetic_record(build):
    sample = _sample("swe_agent_synthetic")
    assert _steps("swe_agent_nebius", build(sample["raw"])) == sample["expected_steps"]


def test_swe_agent_footers_come_from_the_results_of_the_reader():
    """The reader turns every user turn after the first agent turn into a result, so the open file is read from there."""
    run = _swe_agent_run(_sample("swe_agent_synthetic")["raw"])
    assert [s["source"] for s in run.steps if s["source"] != "agent"] == ["system", "user"]
    assert any((s.get("observation") or {}).get("results") for s in run.steps)
    resources = [d["resource"] for d in ap.derive_deps(run, rules=[D.resource_chain(D.shell_commands_v1)])]
    assert "file:/repo/src/a.py" in resources  # the path of a footer, not the path the open command was given


def test_web_extractor_and_projection_on_a_synthetic_record():
    """The page of each step is its record's url, which the reader keeps in the result of the call before it."""
    sample = _sample("agent_reward_bench_synthetic")
    assert _steps("agent_reward_bench", _arb_run(sample["raw"])) == sample["expected_steps"]


def test_the_page_after_the_last_action_is_a_step_in_grade_only():
    """The last record is the page after the last action and holds no action. GRADE turns it into a decision step and
    a noop step, which agentpile does not copy (see the tail difference)."""
    sample = _sample("agent_reward_bench_records")
    run = _arb_run(sample["raw"])
    expected = sample["expected_steps"]
    assert _steps("agent_reward_bench", run) == expected[:-2]
    diffs = G.diff_steps("agent_reward_bench", "k", expected, _steps("agent_reward_bench", run))
    assert [(d["field"], d["expected"], d["got"]) for d in diffs] == [
        ("n_steps", len(expected), len(expected) - 2),
        ("tail", [{"idx": 16, "agent": "agent", "kind": "decision", "deps": []},
                  {"idx": 17, "agent": "env", "kind": "tool_call", "deps": [13]}], [])]
    assert sample["raw"]["records"][-1].get("action") is None
    assert [s["source"] for s in run.steps] == ["user"] + ["agent"] * 8  # no step for the last record


# -- offline: the projection computes no dependency ---------------------------------------------------

def _openhands_sample_run():
    return _openhands_run(_sample("b_openhands_multi_call_turn")["raw"])


def test_projection_has_no_dependencies_of_its_own():
    run = _openhands_sample_run()
    bare = G.project("swe_rebench_openhands", run, [])
    assert [s["deps"] for s in bare.steps] == [[]] * len(bare.steps)
    addr = lambda call: {"trajectory": run.id, "step": 3, "call": call}  # noqa: E731
    extra = [D._dep(addr("c1"), addr("c3"), "made_up", "1")]
    out = G.project("swe_rebench_openhands", run, extra)
    assert [i for i, s in enumerate(out.steps) if s["deps"]] == [3] and out.steps[3]["deps"] == [1]
    ghost = [D._dep({"trajectory": run.id, "step": 99, "call": "x"}, addr("c3"), "made_up", "1")]
    out = G.project("swe_rebench_openhands", run, ghost)
    assert out.dropped_edges == 1 and out.edges == 0


def test_a_wrong_rule_is_caught_by_the_comparison():
    sample = _sample("b_openhands_multi_call_turn")
    run = _openhands_run(sample["raw"])
    got = G.project("swe_rebench_openhands", run, ap.derive_deps(run, rules=[D.last_writer(D.file_editor_paths_v1)]))
    diffs = G.diff_steps("swe_rebench_openhands", "k", sample["expected_steps_hand"], got.steps)
    assert diffs and {d["field"] for d in diffs} == {"deps"}


def test_run_key_matches_the_expected_files_format():
    key = G.grade_run_key("who_and_when", "Who&When/Algorithm-Generated/1.json", 0,
                          ["4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2"])
    assert key == "who_and_when:Who%26When%2FAlgorithm-Generated%2F1.json:0:4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2"
    key = G.grade_run_key("tau2_bench", "m.jsonl", 3, ["telecom", "telecom_[a]b|c[PERSONA:None]"])
    assert key == "tau2_bench:m.jsonl:3:telecom:telecom_%5Ba%5Db%7Cc%5BPERSONA%3ANone%5D"


# -- offline: the exception list mechanism --------------------------------------------------------------

def _expected_line(sample, key="k"):
    return {"corpus": "swe_rebench_openhands", "run_key": key, "steps": sample["expected_steps_hand"],
            "source": {"file": "trajectories.parquet", "row": 0, "ids": {"trajectory_id": "sample-b",
                       "instance_id": "demo__repo-2"}, "label_failed": 1}}


def test_a_difference_outside_the_list_fails_and_a_listed_one_passes():
    sample = _sample("b_openhands_multi_call_turn")
    run = _openhands_run(sample["raw"])
    key = G.run_key("swe_rebench_openhands", run)
    line = _expected_line(sample, key)
    clean = G.compare_corpus("swe_rebench_openhands", [run], [line])
    assert clean.clean and clean.runs_compared == 1 and clean.steps_compared == 8 and clean.edges_expected == 2
    bent = json.loads(json.dumps(line))
    bent["steps"][5]["deps"] = [1]  # GRADE said step 5 depends on step 2; claim step 1
    report = G.compare_corpus("swe_rebench_openhands", [run], [bent])
    assert not report.clean and len(report.unlisted) == 1
    entry = dict(report.unlisted[0], id="E0", reason="test")
    assert entry["expected"] == [1] and entry["got"] == [2] and entry["step"] == 5
    listed = G.compare_corpus("swe_rebench_openhands", [run], [bent], [entry])
    assert listed.clean and len(listed.listed) == 1 and listed.listed[0]["exception"] == "E0"
    stale = G.compare_corpus("swe_rebench_openhands", [run], [line], [entry])
    assert not stale.clean and stale.stale == [entry]  # a listed difference that does not occur is an error
    missing = G.compare_corpus("swe_rebench_openhands", [], [line])
    assert missing.missing_runs == [key]


def test_a_wrong_join_is_reported():
    sample = _sample("b_openhands_multi_call_turn")
    run = _openhands_run(sample["raw"])
    line = _expected_line(sample, G.run_key("swe_rebench_openhands", run))
    line["source"]["label_failed"] = 0  # the run is not resolved: label_failed must be 1
    assert G.compare_corpus("swe_rebench_openhands", [run], [line]).join_errors


# -- network: every run of every expected file ---------------------------------------------------------------

def _expected(name):
    where = os.environ.get("AGENTPILE_GRADE_EXPECTED")
    if not where:  # a selected comparison that cannot run is a failure, not a silent skip
        pytest.fail("AGENTPILE_GRADE_EXPECTED is not set: point it at the decompressed GRADE expected outputs, "
                    "or deselect these tests with -m 'network and not grade'")
    return G.load_expected(Path(where) / name)


def _root(corpus):
    data = os.environ.get("AGENTPILE_DATA")
    return str(Path(data) / corpus) if data else None


def _compare(corpus, file, runs, steps, edges, label=None, exceptions=None):
    expected = _expected(file)
    assert (len(expected), sum(len(e["steps"]) for e in expected),
            sum(len(s["deps"]) for e in expected for s in e["steps"])) == (runs, steps, edges), "unexpected expected file"
    listed = G.load_exceptions(EXCEPTIONS) if exceptions is None else exceptions
    report = G.compare_corpus(corpus, G.load_runs(corpus, _root(corpus), expected), expected, listed, label=label)
    print("\n" + report.summary())
    assert report.runs_compared == runs and not report.missing_runs
    assert report.steps_compared == steps and report.edges_expected == edges
    assert not report.join_errors, report.join_errors[:3]
    assert not report.unlisted, report.unlisted[:3]
    assert not report.stale, report.stale[:3]
    return report


@pytest.mark.network
@pytest.mark.grade
def test_who_and_when_full_context():
    report = _compare("who_and_when", "who_and_when.jsonl", 184, 4092, 126699)
    assert report.edges_projected == 126699 and report.dropped_edges == 0 and not report.differences


@pytest.mark.network
@pytest.mark.grade
def test_tau_bench_prior_reads():
    report = _compare("tau_bench", "tau_bench.jsonl", 660, 20932, 5584)
    assert report.edges_projected == 5584 and report.dropped_edges == 0 and not report.differences


@pytest.mark.network
@pytest.mark.grade
def test_tau2_bench_value_flow_against_the_frozen_baseline():
    report = _compare("tau2_bench", "tau2_bench.jsonl", 278, 6653, 1250)
    assert report.edges_projected == 1250 and report.dropped_edges == 0 and not report.differences


@pytest.mark.network
@pytest.mark.grade
def test_tau2_bench_against_grade_head_differs_only_by_the_listed_pairing_exception():
    report = _compare("tau2_bench", "tau2_bench.grade-fe6e54d.jsonl", 278, 6653, 1250, label="tau2_bench@fe6e54d")
    assert [(d["step"], d["expected"], d["got"], d["exception"]) for d in report.listed] == [(40, [37], [35], "E3")]


@pytest.mark.network
@pytest.mark.grade
def test_swe_gym_resource_chain():
    pytest.importorskip("pyarrow")  # the dataset is read from Parquet
    report = _compare("swe_gym_openhands", "swe_gym_openhands.jsonl", 376, 13467, 3229)
    assert report.edges_projected == 3229 and report.dropped_edges == 0 and not report.differences


@pytest.mark.network
@pytest.mark.grade
def test_swe_rebench_resource_chain():
    pytest.importorskip("pyarrow")  # the dataset is read from Parquet
    report = _compare("swe_rebench_openhands", "swe_rebench_openhands.jsonl", 600, 77140, 7827)
    assert report.edges_projected == 7827 and report.dropped_edges == 0 and not report.differences


@pytest.mark.network
@pytest.mark.grade
def test_swe_agent_resource_chain():
    report = _compare("swe_agent_nebius", "swe_agent_nebius.jsonl", 600, 30846, 5380)
    assert report.edges_projected == 5380 and report.dropped_edges == 0 and not report.differences


@pytest.mark.network
@pytest.mark.grade
def test_agent_reward_bench_resource_chain():
    """Every difference is listed: the final observation GRADE turns into a step (E17).

    The expected file was regenerated for two runs on 2026-10-20 (E18, a mirror splice), with GRADE 3839a57's
    own to_steps on the pinned bytes, which added two edges (7,968 to 7,970)."""
    report = _compare("agent_reward_bench", "agent_reward_bench.jsonl", 600, 20840, 7970)
    by_cause = {}
    for d in report.listed:
        by_cause.setdefault(d["exception"], []).append(d)
    assert {k: len(v) for k, v in by_cause.items()} == {"E17": 2 * 597}
    assert {d["field"] for d in by_cause["E17"]} == {"n_steps", "tail"}
    tail_edges = sum(len(s["deps"]) for d in by_cause["E17"] if d["field"] == "tail" for s in d["expected"])
    assert report.dropped_edges == 0 and report.edges_projected == 7970 - tail_edges


@pytest.mark.network
@pytest.mark.grade
def test_the_comparison_is_sensitive_to_the_rule():
    pytest.importorskip("pyarrow")  # SWE-Gym is read from Parquet
    """A control: the wrong rule on real data produces differences, so zero differences above means something."""
    expected = _expected("swe_gym_openhands.jsonl")[:20]
    runs = G.load_runs("swe_gym_openhands", _root("swe_gym_openhands"), expected)
    wrong = G.CORPUS_RULES["swe_gym_openhands"]
    try:
        G.CORPUS_RULES["swe_gym_openhands"] = [D.last_writer(D.file_editor_paths_v1)]
        report = G.compare_corpus("swe_gym_openhands", runs, expected)
    finally:
        G.CORPUS_RULES["swe_gym_openhands"] = wrong
    assert report.differences and {d["field"] for d in report.differences} == {"deps"}
