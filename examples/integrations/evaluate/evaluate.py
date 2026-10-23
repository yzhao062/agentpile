"""Score an agent's tool calls offline with Ragas and DeepEval, on rows from ap.to_chat.

Ragas (ToolCallF1, ToolCallAccuracy) and DeepEval (ToolCorrectnessMetric) both compare the calls an agent made with a
list of reference calls. agentpile gives them one input: ap.to_chat(runs, arguments="dict") turns each trajectory into
OpenAI-style messages whose tool calls carry their arguments as dicts. tau-bench runs hold the task's reference calls
in the run label gold_actions (name and kwargs). A score below 1 says the agent's calls differ from that list. It does
not say the run failed, because tau-bench grades the final database state. Ragas matches arguments exactly, and
DeepEval gives partial credit for the argument keys that match.

tau2-bench has a second caller. Its user simulator makes tool calls too, and to_chat writes them as user text
("[user] tool call check_status {}"), so they never reach either library as agent calls. The script checks this against
run.events(), which holds every call with the id of the agent that made it.

Checks (the script exits non-zero when one fails):
  1. The calls handed to Ragas and to DeepEval equal the agent's call events in run.events(), run by run, and every
     simulator call became user text.
  2. Self-reference control: a run scored against its own calls gives 1.0 on both Ragas metrics and on DeepEval.
  3. The stub judge handed to DeepEval, which raises when asked for a completion, was never called. A second metric
     with available_tools set does ask its judge, which shows that the stub is wired in.

    pip install -e . -r examples/integrations/evaluate/requirements.txt    # from the repository root
    python examples/integrations/evaluate/evaluate.py

It reads the test fixtures of this checkout, so it needs no network, API key or account and calls no model.
"""

import os

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"  # set before either library is imported
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import asyncio  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import agentpile as ap  # noqa: E402
from deepeval.metrics import ToolCorrectnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCallParams, ToolCall as DeepEvalCall  # noqa: E402
from ragas.dataset_schema import MultiTurnSample  # noqa: E402
from ragas.messages import AIMessage, HumanMessage, ToolMessage, ToolCall as RagasCall  # noqa: E402
from ragas.metrics.collections import ToolCallAccuracy, ToolCallF1  # noqa: E402  (ragas.metrics.ToolCallF1 is deprecated)

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
SIMULATOR = "user_simulator"  # the roster name of tau2-bench's simulated user
warnings.filterwarnings("ignore", module="ragas")  # Ragas warns when the two call lists differ in length


class NoJudge(DeepEvalBaseLLM):
    """A stand-in judge model that raises, and counts, every request for a completion."""

    fired = 0

    def load_model(self):
        return None

    def get_model_name(self):
        return "no-judge"

    def generate(self, *args, **kwargs):
        self.fired += 1
        raise RuntimeError("DeepEval asked the judge model for a completion")

    async def a_generate(self, *args, **kwargs):
        self.generate()


def tool_correctness(judge, **kwargs):
    """Ordered matching, because the default scores two identical calls in one run below 1.0 against themselves."""
    return ToolCorrectnessMetric(model=judge, async_mode=False, should_consider_ordering=True,
                                 evaluation_params=[ToolCallParams.INPUT_PARAMETERS], **kwargs)


JUDGE = NoJudge()
DEEPEVAL = tool_correctness(JUDGE)  # no available_tools: that argument makes the metric ask the judge to rate tool selection


def convert(row):
    """One chat row as Ragas messages and DeepEval tool calls. Ragas has no system message, so that one is left out."""
    messages, called = [], []
    for m in row["messages"]:
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls", [])]
        called += [DeepEvalCall(name=name, input_parameters=args) for name, args in calls]
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"] or "", tool_calls=[RagasCall(name=n, args=a) for n, a in calls] or None))
        elif m["role"] == "tool":
            messages.append(ToolMessage(content=m["content"]))
    return messages, called


def score(messages, called, ragas_ref, deepeval_ref):
    """(Ragas F1, Ragas strict accuracy, DeepEval tool correctness) of a run's calls against a reference list."""
    sample = MultiTurnSample(user_input=messages, reference_tool_calls=ragas_ref)
    ragas = [asyncio.run(metric.ascore(user_input=sample.user_input, reference_tool_calls=sample.reference_tool_calls)).value
             for metric in (ToolCallF1(), ToolCallAccuracy(strict_order=True))]
    first = next((m.content for m in messages if isinstance(m, HumanMessage)), "")
    case = LLMTestCase(input=first, tools_called=called, expected_tools=deepeval_ref)
    return (*ragas, DEEPEVAL.measure(case, _show_indicator=False))


def check_dataset(name):
    runs = list(ap.load(name, root=FIXTURES / name))
    report = {}
    rows = ap.to_chat(runs, arguments="dict", report=report)
    kept = [(runs[e["input"]], rows[e["row"]]) for e in report["runs"] if e["row"] is not None]
    skipped = [e["skipped"] for e in report["runs"] if e["row"] is None]
    n_agent = n_sim = n_control = 0
    table = []
    for run, row in kept:
        messages, called = convert(row)
        sims = {a["id"] for a in run.agents if a.get("name") == SIMULATOR}
        agent_events = [(e.name, e.content) for e in run.events() if e.kind == "call" and e.actor not in sims]
        sim_events = [e for e in run.events() if e.kind == "call" and e.actor in sims]
        # 1. what the libraries received is what the record says the agent called, and the simulator's calls are text
        in_ragas = [(c.name, c.args) for m in messages if isinstance(m, AIMessage) for c in m.tool_calls or []]
        assert in_ragas == agent_events, f"{run.id}: Ragas got {in_ragas}, the record has {agent_events}"
        assert [(c.name, c.input_parameters) for c in called] == agent_events, f"{run.id}: DeepEval got other calls"
        as_text = [m for m in row["messages"] if m["role"] == "user" and any(m["content"].startswith(f"[{s}] tool call") for s in sims)]
        assert len(as_text) == len(sim_events), f"{run.id}: {len(sim_events)} simulator calls, {len(as_text)} as user text"
        n_agent, n_sim = n_agent + len(called), n_sim + len(sim_events)
        # 2. self-reference control: the run's own calls as the reference
        if called:
            result = score(messages, called, [RagasCall(name=c.name, args=c.input_parameters) for c in called], list(called))
            assert result == (1.0, 1.0, 1.0), f"{run.id}: self-reference control scored {result}"
            # and its opposite: the same calls under tool names the agent never used score 0 in both libraries
            renamed = [DeepEvalCall(name=c.name + "_never_called", input_parameters=c.input_parameters) for c in called]
            result = score(messages, called, [RagasCall(name=c.name, args=c.input_parameters) for c in renamed], renamed)
            assert result == (0.0, 0.0, 0.0), f"{run.id}: a reference of other tool names scored {result}"
            n_control += 1
        # the reference calls tau-bench keeps in the run label gold_actions
        gold = next((lab["value"] for lab in run.labels if lab["name"] == "gold_actions"), None)
        if gold:
            result = score(messages, called, [RagasCall(name=g["name"], args=g["kwargs"]) for g in gold],
                           [DeepEvalCall(name=g["name"], input_parameters=g["kwargs"]) for g in gold])
            assert all(0.0 <= value <= 1.0 for value in result), f"{run.id}: score outside [0, 1]: {result}"
            table.append((run.id, len(called), len(gold), *result))
    print(f"{name}: {len(runs)} runs, {len(kept)} chat rows (skipped: {', '.join(skipped) or 'none'}); "
          f"{n_agent} agent calls handed to Ragas and DeepEval, {n_sim} simulator calls kept out as user text")
    print(f"  controls on {n_control} runs: own calls as reference score 1.0, renamed tools 0.0 (Ragas F1, Ragas "
          "strict accuracy, DeepEval)")
    if table:
        print(f"  against gold_actions ({len(table)} runs):  calls gold  Ragas F1  Ragas acc  DeepEval  run")
        for run_id, n_calls, n_gold, *values in table:
            print(f"{n_calls:>34}{n_gold:>5}{values[0]:>10.4f}{values[1]:>11.4f}{values[2]:>10.4f}  {run_id}")
    return n_sim, len(table)


if __name__ == "__main__":
    (_, n_gold), (n_sim, _) = check_dataset("tau_bench"), check_dataset("tau2_bench")
    assert n_gold > 0 and n_sim > 0, "the fixtures should hold gold_actions (tau_bench) and simulator calls (tau2_bench)"
    # 3. the stub judge never fired; the same metric with available_tools does ask its judge
    probe, tool = NoJudge(), DeepEvalCall(name="t", input_parameters={})
    try:
        tool_correctness(probe, available_tools=[tool]).measure(
            LLMTestCase(input="x", tools_called=[tool], expected_tools=[tool]), _show_indicator=False)
    except RuntimeError:
        pass
    assert probe.fired > 0 and JUDGE.fired == 0, f"judge calls: {JUDGE.fired}; with available_tools: {probe.fired}"
    print(f"stub judge model calls: {JUDGE.fired} (with available_tools set: {probe.fired})\nall checks passed")
