"""ap.show: the plain text view of one run (deterministic, so the tests check it) and a rich smoke test."""

import ast
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats.messages import messages_to_trajectory

HERE = Path(__file__).parent
FIX = HERE / "fixtures"
CORPORA = HERE / "corpora" / "fixtures"

QUICK_START = [
    {"role": "user", "content": "Is it raining in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "light rain, 14 C"},
    {"role": "assistant", "content": "Yes, light rain."},
]


def view(run, **kw):
    """The plain view of a run, 100 columns wide unless told otherwise."""
    buf = io.StringIO()
    kw.setdefault("width", 100)
    ap.show(run, file=buf, plain=True, **kw)
    return buf.getvalue()


def corpus(name, **kw):
    return list(ap.load(name, root=str(CORPORA / name), **kw))


def row(out, fragment):
    """The first line holding `fragment` and the lines under it up to the next event row, with spaces collapsed."""
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if fragment in line)
    end = start + 1
    while end < len(lines) and lines[end].strip() and not re.match(r" {2} *\d+ {2}\S", lines[end]):
        end += 1
    return " ".join(" ".join(lines[start:end]).split())


def conversation(*messages):
    return next(ap.load(list(messages)))


# -- the Quick Start conversation ------------------------------------------------------------

def test_quick_start_plain_output_with_the_default_rule():
    assert view(next(ap.load(QUICK_START)), deps=True) == """\
agentpile trajectory: conversation-0
  record  trajectory (ATIF-v1.7)
  from    format openai-messages
  counts  steps 3, messages 3, calls 1, results 1

events
  step  role   kind     content
  ----  -----  -------  -------
     1  user   message  Is it raining in Paris?
     2  agent  message  [stand-in: source was null]
     2  agent  call     get_weather {"city": "Paris"}  id=call_1
     2  tool   result   light rain, 14 C  answers call_1
     3  agent  message  Yes, light rain.
                        <- depends on step 2 call_1 (tool_call_id)
"""


def test_dependencies_appear_only_when_asked_for():
    run = next(ap.load(QUICK_START))
    assert "depends on" not in view(run)
    assert "depends on" not in view(run, deps=None) and "depends on" not in view(run, deps=False)
    assert view(run, deps=True) == view(run, deps=["tool_call_id_v1"]) == view(run, deps="tool_call_id")


def test_the_quick_start_conversation_is_the_one_in_the_readme():
    text = (HERE.parent / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assign = next((m for m in (re.search(r"^messages = (\[.*?^\])", b, re.DOTALL | re.MULTILINE) for b in blocks)
                   if m), None)
    if assign is None:
        pytest.skip("no python block of the README assigns messages")
    assert ast.literal_eval(assign.group(1)) == QUICK_START


# -- the corpus fixtures ------------------------------------------------------------------------

def test_who_and_when_shows_step_and_agent_labels_beside_their_events():
    out = view(corpus("who_and_when")[0])
    assert out == """\
agentpile trajectory: Algorithm-Generated/1
  record   trajectory (ATIF-v1.7)
  from     Kevin355/Who_and_When @ 59b9fcb
           file Who&When/Algorithm-Generated/1.json
  task     What is 2+2 in base 3?
  agents   Math_Expert
           Coder_Expert
           Computer_terminal
  outcome  success=false, raw={"is_correct": false}
  counts   steps 5, messages 5, calls 0, results 0

events
  step  role   actor              kind     content
  ----  -----  -----------------  -------  -------
     1  agent  Math_Expert        message  Plan: convert 4 to base 3.
                                           <- agent label mistake_agent = true (who_and_when)
     2  agent  Coder_Expert       message  ```python print(4) ```
     3  agent  Computer_terminal  message  exitcode: 0 (execution succeeded) Code output: 4
     4  agent  Math_Expert        message  (empty)
                                           <- agent label mistake_agent = true (who_and_when)
     5  agent  Math_Expert        message  The answer is 4.
                                           <- step label mistake_step = true (who_and_when)
                                           <- step label mistake_reason = "Did not convert to base
                                              3." (who_and_when)
                                           <- agent label mistake_agent = true (who_and_when)
"""


def test_who_and_when_hand_crafted_run_labels_and_roles():
    out = view(corpus("who_and_when")[2])
    assert "  labels   mistake_type = \"Tool failure\" (who_and_when)" in out
    step4 = row(out, "I clicked an unrelated page.")
    assert "step label mistake_step = true" in step4 and "agent label mistake_agent = true" in step4
    assert "agent label" not in row(out, "Please search the web.")  # Orchestrator is not the labeled agent
    assert "user - message Find the synthetic shop." in row(out, "message  Find the synthetic shop.")


def test_trail_span_labels_sit_on_the_step_or_call_their_span_maps_to():
    pytest.importorskip("pyarrow")
    run = corpus("trail")[0]
    out = view(run, max_chars=500, width=300)
    spans = run.spans
    assert spans["a000000000000007"]["address"]["step"] == 7  # an LLM span that became step 7
    assert spans["a000000000000012"]["address"]["call"] == "call_sub1"  # a TOOL span tied to a call
    assert "span a000000000000007" in row(out, "Thought: delegate to the helper.")
    call_row = row(out, "web_search {\"query\": \"synthetic fact\"}")
    assert "span a000000000000012" in call_row and "[paired by unique_tool_span]" in call_row
    assert call_row.count("span a000000000000011") == 2  # two labels on one LLM span of step 12, listed beside it
    assert "span a000000000000012" not in row(out, "Observation: The synthetic fact is A.")


def test_trail_span_with_no_step_goes_to_the_closing_section():
    pytest.importorskip("pyarrow")
    run = corpus("trail")[1]
    assert run.spans["b000000000000003"]["address"] is None  # a CHAIN span is not a step
    out = view(run, max_chars=500, width=300)
    head, closing = out.split("labels with nothing to sit beside\n")
    assert "span b000000000000003" not in head.split("events\n")[1]
    assert "span b000000000000003 (CHAIN Step 1) maps to no step or call" in " ".join(closing.split())


def test_trail_run_level_labels_and_inferred_pairings_are_marked():
    pytest.importorskip("pyarrow")
    runs = corpus("trail")
    out = view(runs[0])
    assert "  labels  error = {\"category\": \"Resource Abuse\"" in out  # the unresolved location stays on the run
    assert "overall = 3.5 (trail)" in out
    hist = row(view(runs[3]), "id=call_hist")  # a call read from a history message, placed on the latest agent step
    assert "[call from history message]" in hist and "[paired by latest_agent_step]" in hist


def test_catchbench_pre_is_a_config_with_tool_labels_beside_the_tools():
    out = view(corpus("catchbench_pre")[0])
    assert out == """\
agentpile config: crewai-x1
  record  config (agentpile-config/0)
  from    yzhao062/catchbench @ 30a516c
          file data/pre/crewai.json
  steps   none: a configuration with no execution, so there are no events
  counts  agents 1, tools 2
  labels  minimal_reference = ["WebSearch"] (catchbench)

agents
  agent  model  tools  instructions
  -----  -----  -----  ------------
  agent  -          2  -

tools
  tool        permission  from      details
  ----------  ----------  --------  -------
  WebSearch   network     declared  owner agent, type network
  FileWriter  write       declared  owner agent, type file
                                    <- tool label excess_capability = true (llm_judge)
"""


def test_config_record_lists_agents_tools_and_where_each_permission_came_from():
    out = view(next(ap.load(FIX / "config_mcp.json")))
    assert out == """\
agentpile config: fx-config-1
  record       config (agentpile-config/0)
  from         format mcp-tools-list
  instruction  Help with files.
  steps        none: a configuration with no execution, so there are no events
  counts       agents 1, tools 2

agents
  agent      model  tools  instructions
  ---------  -----  -----  ------------
  assistant  -          2  Help with files.
                           <- agent label excess_permission = true (fixture)

tools
  tool                          permission  from        details
  ----------------------------  ----------  ----------  -------
  read_file (fs.read_file)      read        annotation  owner fs, type filesystem, mcp definition
  delete_file (fs.delete_file)  write       annotation  owner fs, type filesystem, mcp definition
"""


def test_a_config_record_gets_no_dependencies():
    assert view(next(ap.load(FIX / "config_mcp.json")), deps=True) == view(next(ap.load(FIX / "config_mcp.json")))


def test_tau2_results_paired_by_position_say_so():
    out = view(corpus("tau2_bench")[0], width=300)
    assert "answers a-c1  [paired by position]" in out


# -- every reader's output shows without error --------------------------------------------------

NAMES = ["who_and_when", "mast", "trail", "tau_bench", "tau2_bench", "agent_reward_bench", "agentdojo",
         "swe_rebench_openhands", "swe_agent_nebius", "swe_gym_openhands", "swe_smith", "adp", "catchbench_pre",
         "trace_commons"]
NEEDS_PYARROW = ("trail", "swe_rebench_openhands", "swe_agent_nebius", "swe_gym_openhands", "swe_smith")


def test_the_corpus_list_covers_every_registered_dataset():
    assert sorted(NAMES) == sorted(ap.datasets.names())


@pytest.mark.parametrize("name", NAMES)
def test_first_run_of_each_corpus_fixture_shows(name):
    if name in NEEDS_PYARROW:
        pytest.importorskip("pyarrow")
    run = next(ap.load(name, root=str(CORPORA / name)))
    out = view(run, deps=True)
    assert out.startswith("agentpile %s: %s\n" % (run.kind, run.id))
    assert out == view(run, deps=True)  # deterministic
    assert out.endswith("\n") and "\t" not in out
    assert all(line == line.rstrip() for line in out.splitlines())


def test_layout_text_is_ascii_and_only_source_text_is_not():
    for run in corpus("who_and_when") + corpus("catchbench_pre") + [next(ap.load(QUICK_START))]:
        assert view(run, deps=True).isascii()
    arb = "".join(view(r) for r in corpus("agent_reward_bench"))
    assert {c for c in arb if not c.isascii()} <= set("é☕")  # characters of the fixture's own page text


# -- marks the record carries -------------------------------------------------------------------

def test_stand_ins_and_unpaired_results_are_marked_from_the_record():
    run = conversation(
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "no_args"}},
            {"id": "b", "type": "function", "function": {"name": "null_args", "arguments": None}},
            {"id": "c", "type": "function", "function": {"name": "bad_args", "arguments": "{'id': 42"}},
            {"type": "function", "function": {"name": "no_id", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": None},
        {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "ok"},
                                                          {"type": "image_url", "image_url": {"url": "x"}}]},
        {"role": "tool", "tool_call_id": "zzz", "content": "orphan"},
        {"role": "assistant", "content": [{"type": "text", "text": "Done."}, {"type": "image_url", "image_url": {}}]})
    out = view(run, width=300)
    assert "message  [stand-in: source was null]" in out
    assert "no_args  id=a  [stand-in: source had no arguments]" in out
    assert "null_args  id=b  [stand-in: source arguments were null]" in out
    assert "bad_args {'id': 42  id=c  [stand-in: arguments were not a JSON object, ATIF holds {}]" in out
    assert "no_id {}  id=ap-3-3  [stand-in: call id generated]" in out
    assert "result   [stand-in: source content was null]  answers a" in out
    assert "ok [image_url part]  answers b  [stand-in: content substituted in ATIF, source value kept]" in out
    assert "result   orphan" in out and "answers zzz" not in out  # the id matched no call, so no pairing is claimed
    assert "Done. [image_url part]  [stand-in: content substituted in ATIF, source value kept]" in out


def test_a_generated_call_id_names_the_source_id():
    run = conversation({"role": "assistant", "content": "x", "tool_calls": [
        {"id": "dup", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "dup", "type": "function", "function": {"name": "g", "arguments": "{}"}}]})
    assert "[stand-in: call id generated, source id was \"dup\"]" in view(run, width=300)


def test_an_error_shows_on_the_result_that_answers_the_call_and_not_twice():
    out = view(next(ap.load(FIX / "atif_multiagent.json")), width=300)
    assert "error: {\"type\": \"tool_error\", \"message\": \"malformed arguments\"}" in row(out, "answers c3")
    assert "error:" not in row(out, "book_flight")
    assert "book_flight {'offer_id': 7  id=c3  [stand-in: arguments were not a JSON object" in out


def test_events_follow_the_source_order_not_the_atif_order():
    out = view(next(ap.load(FIX / "atif_multiagent.json")))
    assert out.index("answers c1") < out.index("answers c2")  # the fixture stores c2's result first


def test_an_empty_source_says_so():
    out = view(ap.Run(messages_to_trajectory([], trajectory_id="empty")))
    assert "steps 0, messages 0, calls 0, results 0" in out
    assert "no events: the source held none" in out


# -- text handling ------------------------------------------------------------------------------

def test_content_is_cut_at_max_chars_and_says_how_much_is_left():
    run = conversation({"role": "user", "content": "0123456789" * 6})
    assert "message  0123456789... (+50 chars)" in view(run, max_chars=10)
    assert "0123456789" * 6 in view(run, max_chars=None)
    assert view(run) == view(run, max_chars=300)  # the default
    long = conversation({"role": "user", "content": "x" * 400})
    assert "x" * 300 + "... (+100 chars)" in view(long, width=1000)


def test_task_arguments_and_label_values_are_cut_too():
    run = corpus("who_and_when")[0]
    out = view(run, max_chars=8)
    assert "task     What is... (+14 chars)" in out
    assert "mistake_reason = \"Did not... (+20 chars)" in out


def test_whitespace_collapses_and_control_characters_are_spelled_out():
    run = conversation({"role": "user", "content": "line one\n\n  line\ttwo \x1b[31mred\x1b[0m \x00"})
    out = view(run)
    assert "line one line two \\x1b[31mred\\x1b[0m \\x00" in out
    assert "\x1b" not in out and "\x00" not in out


def test_long_lines_wrap_at_the_width_with_a_hanging_indent():
    out = view(conversation({"role": "user", "content": " ".join(["word"] * 40)}), width=50)
    assert max(len(line) for line in out.splitlines()) <= 50
    lines = out.splitlines()
    first = next(i for i, line in enumerate(lines) if "message  word" in line)
    assert lines[first + 1].startswith(" " * lines[first].index("word")) and "word" in lines[first + 1]


def test_the_default_width_is_fixed_when_the_stream_is_not_a_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "40")
    run = conversation({"role": "user", "content": " ".join(["word"] * 40)})
    buf = io.StringIO()
    ap.show(run, file=buf, plain=True)
    assert max(len(line) for line in buf.getvalue().splitlines()) <= 100
    assert any(len(line) > 60 for line in buf.getvalue().splitlines())


# -- dependencies --------------------------------------------------------------------------------

def test_full_context_lists_its_sources_as_ranges():
    out = view(corpus("who_and_when")[0], deps=["full_context_v1"])
    assert "<- depends on step 1 (full_context)" in row(out, "```python print(4)")
    assert "<- depends on steps 1-4 (full_context)" in row(out, "The answer is 4.")


def test_a_rule_with_a_resource_names_it_and_a_call_destination_gets_the_note():
    def rule(run):
        return [{"src": {"trajectory": run.id, "step": 1}, "dst": {"trajectory": run.id, "step": 2, "call": "call_1"},
                 "rule": "demo", "rule_version": "1", "resource": ["file:/a.py", "file:/b.py"]}]

    out = view(next(ap.load(QUICK_START)), deps=rule)
    assert "<- depends on step 1 (demo, file:/a.py, file:/b.py)" in row(out, "get_weather")
    assert "depends on" not in row(out, "Yes, light rain.")


def test_a_dependency_with_no_destination_event_goes_to_the_closing_section():
    def rule(run):
        return [{"src": {"trajectory": run.id, "step": 1}, "dst": {"trajectory": run.id, "step": 9},
                 "rule": "demo", "rule_version": "1", "resource": None}]

    out = view(next(ap.load(QUICK_START)), deps=rule)
    assert out.endswith("dependencies with nothing to sit beside\n"
                        "  <- depends on step 1 (demo) -- at step 9\n")


# -- labels --------------------------------------------------------------------------------------

def _with_labels(run, *labels):
    record = run.to_dict()
    record["extra"]["agentpile"].setdefault("labels", []).extend(labels)
    return ap.Run(record)


def _target(kind, **address):
    return {"kind": kind, "address": address}


def test_labels_that_name_nothing_in_the_record_are_listed_with_the_reason():
    run = next(ap.load(FIX / "atif_multiagent.json"))
    run = _with_labels(
        run,
        {"name": "late", "target": _target("step", trajectory="fx-1", step=99), "value": 1, "label_source": "t"},
        {"name": "ghost", "target": _target("agent", trajectory="fx-1", agent="nobody"), "value": 1, "label_source": "t"},
        {"name": "tl", "target": _target("tool", trajectory="fx-1", tool="no.tool"), "value": 1, "label_source": "t"},
        {"name": "sp", "target": _target("span", trajectory="fx-1", span="s9"), "value": 1, "label_source": "t"},
        {"name": "far", "target": _target("step", trajectory="elsewhere", step=1), "value": 1, "label_source": "t"},
        {"name": "odd", "target": _target("galaxy", trajectory="fx-1"), "value": 1, "label_source": "t"})
    closing = " ".join(view(run).split("labels with nothing to sit beside\n")[1].split())
    for part in ("step label late = 1 (t) -- step 99 has no events",
                 "agent label ghost = 1 (t) -- no event by agent nobody",
                 "tool label tl = 1 (t) -- no call of tool no.tool",
                 "span label sp = 1 (t), span s9 -- span s9 is not in this record",
                 "step label far = 1 (t) -- addresses trajectory elsewhere, which this record does not hold",
                 "galaxy label odd = 1 (t) -- a galaxy target is not one this view places"):
        assert part in closing


def test_call_and_tool_labels_sit_on_the_call_events():
    run = _with_labels(
        next(ap.load(FIX / "atif_multiagent.json")),
        {"name": "risky", "target": _target("call", trajectory="fx-1", step=4, call="c3"), "value": "high",
         "label_source": "human"},
        {"name": "slow", "target": _target("tool", trajectory="fx-1", tool="t.search"), "value": True,
         "label_source": "bench"})
    out = view(run)
    assert "<- call label risky = \"high\" (human)" in row(out, "book_flight")
    assert "<- tool label slow = true (bench)" in row(out, "search_flights")
    assert "slow" not in row(out, "read_calendar")


# -- subagents -------------------------------------------------------------------------------------

def test_subagent_trajectories_follow_the_main_events():
    record = next(ap.load(FIX / "atif_multiagent.json")).to_dict()
    record["subagent_trajectories"] = [{
        "schema_version": "ATIF-v1.7", "session_id": "sub-1", "trajectory_id": "sub-1",
        "agent": {"name": "helper", "version": "0"},
        "steps": [{"step_id": 1, "source": "agent", "message": "sub says hi"}]}]
    out = view(ap.Run(record))
    assert "counts   steps 5, messages 5, calls 3, results 3, subagents 1" in out
    assert out.index("\nevents\n") < out.index("\nsubagent sub-1\n") < out.index("sub says hi")


def test_a_run_label_addressed_to_another_trajectory_stays_out_of_the_header():
    record = next(ap.load(FIX / "atif_multiagent.json")).to_dict()
    record["subagent_trajectories"] = [{
        "schema_version": "ATIF-v1.7", "session_id": "sub-1", "trajectory_id": "sub-1",
        "agent": {"name": "helper", "version": "0"},
        "steps": [{"step_id": 1, "source": "agent", "message": "sub says hi"}]}]
    record["extra"]["agentpile"]["labels"] += [
        {"name": "main_ok", "target": {"kind": "run", "address": {}}, "value": True},
        {"name": "child_failed", "target": {"kind": "run", "address": {"trajectory": "sub-1"}}, "value": True},
        {"name": "ghost_label", "target": {"kind": "run", "address": {"trajectory": "ghost"}}, "value": True}]
    out = view(ap.Run(record))
    header = out[:out.index("\nevents\n")]
    assert "main_ok" in header and "child_failed" not in header and "ghost_label" not in header
    assert out.index("\nsubagent sub-1\n") < out.index("child_failed")
    assert out.index("labels with nothing to sit beside") < out.index("ghost_label")


# -- the function itself ------------------------------------------------------------------------

def test_show_writes_to_stdout_by_default_and_returns_none(capsys):
    assert ap.show(next(ap.load(QUICK_START)), plain=True) is None
    assert capsys.readouterr().out.startswith("agentpile trajectory: conversation-0\n")


def test_arguments_are_checked():
    run = next(ap.load(QUICK_START))
    with pytest.raises(TypeError, match="takes a Run"):
        ap.show(run.record, plain=True)
    with pytest.raises(ValueError, match="max_chars"):
        ap.show(run, max_chars=0, plain=True)
    with pytest.raises(ValueError, match="width"):
        ap.show(run, width=0, plain=True)


def test_show_is_exported_and_the_view_extra_is_declared():
    assert "show" in ap.__all__ and ap.show is not None
    pyproject = (HERE.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^view = \["rich>=13"\]$', pyproject, re.MULTILINE)


def test_importing_agentpile_and_showing_plain_does_not_import_rich():
    code = ("import io, sys\nimport agentpile as ap\n"
            "run = next(ap.load([{'role': 'user', 'content': 'hi'}]))\n"
            "ap.show(run, file=io.StringIO(), plain=True)\n"
            "print('rich' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_without_rich_the_view_falls_back_to_plain_text(monkeypatch):
    monkeypatch.setitem(sys.modules, "rich", None)  # an import of rich now raises ImportError
    monkeypatch.setitem(sys.modules, "rich.console", None)
    run = next(ap.load(QUICK_START))
    buf = io.StringIO()
    ap.show(run, file=buf, width=100)
    assert buf.getvalue() == view(run)
    with pytest.raises(ImportError, match=r"agentpile\[view\]"):
        ap.show(run, file=io.StringIO(), plain=False)


# -- rich ------------------------------------------------------------------------------------------

def test_rich_view_renders_the_same_content_into_a_console_on_a_string_buffer():
    console_module = pytest.importorskip("rich.console")
    buf = io.StringIO()
    console = console_module.Console(file=buf, width=100, highlight=False)
    run = corpus("who_and_when")[0]
    assert ap.show(run, file=console, deps=["full_context_v1"]) is None
    out = buf.getvalue()
    assert "agentpile trajectory: Algorithm-Generated/1" in out
    assert "Kevin355/Who_and_When @ 59b9fcb" in out
    assert "Plan: convert 4 to base 3." in out
    assert "mistake_step = true (who_and_when)" in out
    assert "depends on steps 1-4 (full_context)" in out
    assert "\x1b" not in out  # not a terminal, so no colour codes


def test_rich_view_with_a_text_stream_a_config_and_markup_in_the_source():
    pytest.importorskip("rich")
    buf = io.StringIO()
    ap.show(next(ap.load(FIX / "config_mcp.json")), file=buf, width=100, plain=False)
    assert "read_file (fs.read_file)" in buf.getvalue() and "annotation" in buf.getvalue()
    buf = io.StringIO()
    ap.show(conversation({"role": "user", "content": "[bold red]loud[/] [link=https://x.test]y[/link]"}),
            file=buf, width=100)
    assert "[bold red]loud[/] [link=https://x.test]y[/link]" in buf.getvalue()  # source text is never markup


def test_plain_true_ignores_an_installed_rich():
    pytest.importorskip("rich")
    run = corpus("who_and_when")[0]
    assert view(run).startswith("agentpile trajectory: Algorithm-Generated/1\n  record")
    buf = io.StringIO()
    ap.show(run, file=buf, width=100, plain=True)
    assert "╭" not in buf.getvalue() and buf.getvalue().isascii()


def test_inside_a_notebook_a_text_stream_still_gets_the_view(monkeypatch):
    console_module = pytest.importorskip("rich.console")
    monkeypatch.setattr(console_module, "_is_jupyter", lambda: True)  # what rich detects in a Jupyter kernel
    assert console_module.Console().is_jupyter  # so a Console left to decide sends its output to the cell
    buf = io.StringIO()
    ap.show(corpus("who_and_when")[0], file=buf, width=100)
    assert "mistake_step = true (who_and_when)" in buf.getvalue()
