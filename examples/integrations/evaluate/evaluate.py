"""Score tool calls with Ragas and DeepEval.

Takes runs and compares the tool calls each agent made with a reference list: tau-bench runs keep it in the run label
gold_actions (name and kwargs). ap.to_chat(runs, arguments="dict") gives both libraries their input, and each run gets
three scores from 0 to 1: Ragas ToolCallF1 and strict-order ToolCallAccuracy (arguments must match exactly) and
DeepEval ToolCorrectnessMetric (partial credit for the argument keys that match). A score below 1 says the agent's
calls differ from the list; tau-bench grades the final database state, so the run may still have passed. tau2-bench's
user simulator makes tool calls too, and to_chat writes them as user text, so only the agent's calls are scored.

    pip install -e ".[hf]" -r examples/integrations/evaluate/requirements.txt
    python examples/integrations/evaluate/evaluate.py                        # the bundled fixtures, with the checks
    python examples/integrations/evaluate/evaluate.py tau_bench --limit 20   # your runs: tau_bench, which has gold_actions
"""

import argparse
import asyncio
import warnings
from itertools import islice
from pathlib import Path

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCallParams, ToolCall as DeepEvalCall
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolMessage, ToolCall as RagasCall
from ragas.metrics.collections import ToolCallAccuracy, ToolCallF1  # ragas.metrics.ToolCallF1 is deprecated

import agentpile as ap

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
warnings.filterwarnings("ignore", module="ragas")  # Ragas warns when the two call lists differ in length


def evaluate(runs):
    """(run id, Ragas F1, Ragas strict accuracy, DeepEval tool correctness) of each run that has a gold_actions label."""
    report = {}
    rows = ap.to_chat(runs, arguments="dict", report=report)
    scores = []
    for entry in report["runs"]:
        run = runs[entry["input"]]
        gold = next((lab["value"] for lab in run.labels if lab["name"] == "gold_actions"), None)
        if entry["row"] is not None and gold:
            messages, called = convert(rows[entry["row"]])
            scores.append((run.id, *score(messages, called, [RagasCall(name=g["name"], args=g["kwargs"]) for g in gold],
                                          [DeepEvalCall(name=g["name"], input_parameters=g["kwargs"]) for g in gold])))
    return scores


class NoJudge(DeepEvalBaseLLM):
    """DeepEval wants a model object; this metric never calls it, so this one raises, and counts, any request."""

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


SIMULATOR = "user_simulator"  # the roster name of tau2-bench's simulated user


def check_dataset(name, runs):
    report = {}
    rows = ap.to_chat(runs, arguments="dict", report=report)
    kept = [(runs[e["input"]], rows[e["row"]]) for e in report["runs"] if e["row"] is not None]
    skipped = [e["skipped"] for e in report["runs"] if e["row"] is None]
    n_agent = n_sim = n_control = 0
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
    print(f"{name}: {len(runs)} runs, {len(kept)} chat rows (skipped: {', '.join(skipped) or 'none'}); "
          f"{n_agent} agent calls handed to Ragas and DeepEval, {n_sim} simulator calls kept out as user text")
    print(f"  controls on {n_control} runs: own calls as reference score 1.0, renamed tools 0.0 (Ragas F1, Ragas "
          "strict accuracy, DeepEval)")
    return n_sim


def check(tau, tau2, scores):
    n_sim = check_dataset("tau_bench", tau) + check_dataset("tau2_bench", tau2)
    for run_id, *values in scores:
        assert all(0.0 <= value <= 1.0 for value in values), f"{run_id}: score outside [0, 1]: {values}"
    assert scores and n_sim > 0, "the fixtures should hold gold_actions (tau_bench) and simulator calls (tau2_bench)"
    # 3. the stub judge never fired; the same metric with available_tools does ask its judge
    probe, tool = NoJudge(), DeepEvalCall(name="t", input_parameters={})
    try:
        tool_correctness(probe, available_tools=[tool]).measure(
            LLMTestCase(input="x", tools_called=[tool], expected_tools=[tool]), _show_indicator=False)
    except RuntimeError:
        pass
    assert probe.fired > 0 and JUDGE.fired == 0, f"judge calls: {JUDGE.fired}; with available_tools: {probe.fired}"
    print(f"stub judge model calls: {JUDGE.fired} (with available_tools set: {probe.fired})")


def load_runs(sources, limit):
    """The first `limit` runs of each source; a dataset name takes limit= and downloads only that part, a file path does not."""
    return [run for source in sources
            for run in islice(ap.load(source, **({"limit": limit} if source in ap.datasets.names() else {})), limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="*", help="a dataset name or a trace file (default: the bundled fixtures, with the checks)")
    parser.add_argument("--limit", type=int, default=20, help="runs per source (default: 20)")
    args = parser.parse_args()
    if args.source:
        scores = evaluate(load_runs(args.source, args.limit))
    else:
        tau, tau2 = (list(ap.load(name, root=FIXTURES / name)) for name in ("tau_bench", "tau2_bench"))
        scores = evaluate(tau)
    if scores:
        f1, accuracy, deepeval = (sum(s[i] for s in scores) / len(scores) for i in (1, 2, 3))
        print(f"{len(scores)} runs scored against gold_actions: mean Ragas F1 {f1:.2f}, Ragas accuracy {accuracy:.2f}, DeepEval {deepeval:.2f}")
    else:
        print("no run has a gold_actions label to score against")
    if not args.source:
        check(tau, tau2, scores)
        print("all checks passed")


if __name__ == "__main__":
    main()
