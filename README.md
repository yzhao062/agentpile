<div align="center">

<h1><img src="https://raw.githubusercontent.com/yzhao062/agentpile/main/assets/logo.png" alt="agentpile" width="435"></h1>

**From agent trajectory datasets and traces to the tools you already use.**

[![CI](https://github.com/yzhao062/agentpile/actions/workflows/tests.yml/badge.svg)](https://github.com/yzhao062/agentpile/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/agentpile.svg)](https://pypi.org/project/agentpile/)
[![Python](https://img.shields.io/badge/python-3.9%20to%203.13-blue.svg)](https://github.com/yzhao062/agentpile/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/yzhao062/agentpile/blob/main/LICENSE)

[What You Can Do](#1-what-you-can-do) · [Quick Start](#2-quick-start) · [Datasets](#3-datasets) · [Add a Dataset](#4-add-a-dataset) · [Capture](#5-capturing-a-running-agent) · [Reference](#6-reference)

</div>

```python
import agentpile as ap

run = next(ap.load("who_and_when"))   # one of 14 datasets by name, or a trace file by path
ap.show(run)                          # every step, call and result, with the source's labels beside them
```

<p align="center">
  <img src="https://raw.githubusercontent.com/yzhao062/agentpile/main/assets/show.svg" alt="ap.show output for a Who&When run: the task, four agents, the outcome, and five steps, with the expert's mistake labels in color beside step 1" width="100%">
</p>

Public datasets store agent runs in their own schemas (e.g., chat messages in tau-bench, OpenTelemetry spans in TRAIL, raw log text in MAST), and each analysis tool expects its own input. agentpile sits between the two. It reads datasets, trace files and live runs into [ATIF v1.7](https://github.com/harbor-framework/harbor) records that keep what the source recorded. You can then use those records with pandas, networkx or a tracing UI such as Phoenix. It downloads each dataset from its publisher at a pinned revision and redistributes nothing.

<p align="center">
  <img src="https://raw.githubusercontent.com/yzhao062/agentpile/main/assets/overview.png" alt="Three kinds of input (14 public datasets, trace and tool files, live agent runs) go into one run record, an ATIF v1.7 trajectory or a config record with no steps. Three kinds of use come out: read a run with ap.show, analyze many runs with pandas tables and networkx graphs, and hand a run on as JSON or as OTLP spans to Phoenix or Jaeger." width="100%">
</p>

## 1. What You Can Do

### 1.1 Open Any Run in a Tracing UI

tau-bench stores plain chat logs. `ap.send_otlp` posts a run as OTLP spans to Arize Phoenix, Jaeger, MLflow or any OTLP/HTTP collector.

```python
run = next(ap.load("tau_bench", limit=1))
ap.send_otlp(run, "http://localhost:6006/v1/traces")   # Phoenix; Jaeger listens on :4318
```

<p align="center"><img src="https://raw.githubusercontent.com/yzhao062/agentpile/main/assets/phoenix.png" alt="A tau-bench run in Arize Phoenix: system and user messages, chat steps and tool calls in a span tree, with the customer's request and the agent's final answer" width="85%"></p>

A run whose source has no timestamps gets synthetic ones that only order its spans. The durations a tracing UI shows for such a run are therefore not runtime measurements.

### 1.2 Compare Failures Across Datasets

Who&When blames a step, TRAIL marks error spans, and MAST flags 14 failure modes per run. One table holds all three ([script](https://github.com/yzhao062/agentpile/blob/main/examples/compare_failures.py)).

```python
# TRAIL is gated: accept its terms on Hugging Face and log in first (section 3)
runs = [run for name in ("who_and_when", "trail", "mast") for run in ap.load(name)]
labels = ap.to_pandas(runs, "labels")   # run_id, name, value, value_type, target_kind, step, span, ...
```

<p align="center"><img src="https://raw.githubusercontent.com/yzhao062/agentpile/main/assets/compare_failures.png" alt="Left: cumulative share of labels by the position of their step in the run, for Who&When's decisive step, TRAIL's first error and all TRAIL errors. Right: share of MAST runs flagged with each of 14 failure modes" width="100%"></p>

### 1.3 Follow Values Through a Run

Which earlier result does each call reuse? When a call reuses an id-like value from an earlier result, the `value_flow` rule makes it depend on the call that produced that result. Such values include reservation ids, product ids and dates. The links are inferred from the recorded values ([script](https://github.com/yzhao062/agentpile/blob/main/examples/value_flow.py)).

```python
deps = ap.derive_deps(run, rules=["value_flow"])        # each entry names its rule and version
graph = ap.to_networkx(run, rules=["value_flow"])       # events as nodes, dependencies as edges
```

<p align="center"><img src="https://raw.githubusercontent.com/yzhao062/agentpile/main/assets/value_flow.png" alt="Left: one tau-bench run drawn as a graph, with arcs from the calls whose results supplied reservation ids to the later calls that reused them. Right: how many steps back each dependency reaches in tau-bench and tau2-bench" width="100%"></p>

### 1.4 Works With

The same record feeds tools for evaluation, training, auditing, analysis and viewing. `ap.to_chat` turns runs into OpenAI-style chat rows with their tools, `ap.table` into flat tables, and `run.to_otlp` and `run.to_json` into traces and ATIF files. Each row below is a [recipe](https://github.com/yzhao062/agentpile/tree/main/examples/integrations) that CI runs every week at the version shown, offline on the bundled fixtures. A row turns stale when its check fails and stays stale until the same check passes again. A newer version of a tool counts only once its own check has passed.

| Use | Tool | Check | Tested version | Status | Last passing check | Recipe |
|---|---|---|---|---|---|---|
| Evaluate | Ragas | tool-call F1 and accuracy, no model | 0.4.3 | ![check][s-evaluate] | ![last pass][l-evaluate] | [evaluate][r-evaluate] |
| Evaluate | DeepEval | tool correctness, judge never called | 4.3.1 | ![check][s-evaluate] | ![last pass][l-evaluate] | [evaluate][r-evaluate] |
| Train | Hugging Face datasets, TRL | round trips, assistant-only labels | 5.1.0, 1.16.0 | ![check][s-train] | ![last pass][l-train] | [train][r-train] |
| Audit | Invariant Guardrails | data-flow and ordering rules | 0.3.5 | ![check][s-audit] | ![last pass][l-audit] | [audit][r-audit] |
| Analyze | DuckDB, Polars | same answers, Parquet round trip | 2.0.0, 2.0.0 | ![check][s-analyze] | ![last pass][l-analyze] | [analyze][r-analyze] |
| View | Harbor (`harbor view`) | ATIF loads and is served | 0.24.0 | ![check][s-view-harbor] | ![last pass][l-view-harbor] | [view-harbor][r-view-harbor] |
| View | Inspect AI (`inspect view`) | eval log reads back and is served | 0.3.283 | ![check][s-view-inspect] | ![last pass][l-view-inspect] | [view-inspect][r-view-inspect] |
| View | OTLP receivers | exports parse as OTLP and read back | proto 1.45.1 | ![check][s-view-otlp] | ![last pass][l-view-otlp] | [view-otlp][r-view-otlp] |
| View | Arize Phoenix | every trace sent is found | 20.24.0 | ![check][s-ingest-phoenix] | ![last pass][l-ingest-phoenix] | [view-otlp][r-view-otlp] |
| View | Jaeger | every trace sent is found | 2.22.0 | ![check][s-ingest-jaeger] | ![last pass][l-ingest-jaeger] | [view-otlp][r-view-otlp] |
| View | MLflow | every trace sent is found | 3.17.2 | ![check][s-ingest-mlflow] | ![last pass][l-ingest-mlflow] | [view-otlp][r-view-otlp] |

The last three rows start the platform, send the fixture runs to it and query them back. They run weekly, for a release tag and on demand, and after a change to the code they cover they show stale until they run again. Langfuse has a preset in the OTLP recipe and gets a row once its ingestion check passes.

[s-evaluate]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fevaluate.json
[l-evaluate]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fevaluate-last.json
[s-train]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Ftrain.json
[l-train]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Ftrain-last.json
[s-audit]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Faudit.json
[l-audit]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Faudit-last.json
[s-analyze]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fanalyze.json
[l-analyze]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fanalyze-last.json
[s-view-harbor]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fview-harbor.json
[l-view-harbor]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fview-harbor-last.json
[s-view-inspect]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fview-inspect.json
[l-view-inspect]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fview-inspect-last.json
[s-view-otlp]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fview-otlp.json
[l-view-otlp]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fview-otlp-last.json
[s-ingest-phoenix]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fingest-phoenix.json
[l-ingest-phoenix]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fingest-phoenix-last.json
[s-ingest-jaeger]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fingest-jaeger.json
[l-ingest-jaeger]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fingest-jaeger-last.json
[s-ingest-mlflow]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fingest-mlflow.json
[l-ingest-mlflow]: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fyzhao062%2Fagentpile%2Frecipe-status%2Fstatus%2Fingest-mlflow-last.json
[r-evaluate]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/evaluate
[r-train]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/train
[r-audit]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/audit
[r-analyze]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/analyze
[r-view-harbor]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/view-harbor
[r-view-inspect]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/view-inspect
[r-view-otlp]: https://github.com/yzhao062/agentpile/tree/main/examples/integrations/view-otlp

## 2. Quick Start

```bash
pip install agentpile                    # the core: standard library only
pip install "agentpile[hf,view]"         # adds dataset downloads and the colored ap.show
```

| Extra | Adds |
|---|---|
| `hf` | downloads of the Hugging Face datasets |
| `view` | colored `ap.show` output (rich); without it the view is plain text |
| `pandas`, `graph` | `ap.to_pandas` and `ap.to_networkx`; `ap.table` gives the same rows with the core alone |
| `otel` | protobuf for `ap.send_otlp` to a receiver that takes only protobuf, such as Phoenix (Python 3.10+) |
| `pydantic-ai`, `openai-agents` | `ap.capture` for Pydantic AI (Python 3.11+) and the OpenAI Agents SDK (Python 3.10+) |

A conversation in the OpenAI chat format needs only the core:

```python
import agentpile as ap

messages = [
    {"role": "user", "content": "Is it raining in Paris?"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "light rain, 14 C"},
    {"role": "assistant", "content": "Yes, light rain."},
]
run = next(ap.load(messages))             # the format is detected, and an ambiguous input is an error
for event in run.events():                # messages, tool calls and tool results in source order
    print(event.kind, event.role, event.content)

for dep in ap.derive_deps(run, rules=["tool_call_id"]):   # inferred only when you ask, by a named rule
    print(dep["rule"], "step", dep["src"]["step"], dep["src"]["call"], "-> step", dep["dst"]["step"])

run.to_json("run.json")                   # ATIF v1.7 JSON; run.to_otlp("run.otlp.json") writes OTLP spans
ap.to_chat(run)                           # chat rows with tools for training and evaluation; here, the same messages
```

```text
message user Is it raining in Paris?
message agent None
call agent {'city': 'Paris'}
result tool light rain, 14 C
message agent Yes, light rain.
tool_call_id step 2 call_1 -> step 3
```

The second line is a stand-in made visible. ATIF does not permit the `null` in the source's first assistant message. agentpile therefore stores an empty string in the record and notes that the source held `null`, and the event reports `None`. Rule `tool_call_id` gives the dependency: it makes the next agent step after a result depend on the call that produced it, here the final answer (step 3) on `call_1` in step 2.

A dataset loads by name (`ap.datasets.names()` lists them), and a trace file by its path (`ap.readers.names()` lists the formats):

```python
for run in ap.load("who_and_when", limit=5):   # fetches only the files these five runs come from
    print(run.id, run.labels)                  # the labels the reader emits, each with its target
run = next(ap.load("my_trace.json"))           # ATIF, OpenAI messages, OTLP JSON, a tool list or a Claude Code session
```

## 3. Datasets

Each name loads one dataset at the revision shown, and a full load is checked against the run count in its [manifest](https://github.com/yzhao062/agentpile/tree/main/src/agentpile/corpora/manifests). Harbor's ATIF v1.7 models (Harbor 0.22.0) accept every trajectory checked: the first 3,000 runs of each dataset, or all of a smaller one. `tools/dataset_table.py` regenerates this table.

| Name | Source at pinned revision | Runs | Source annotation levels | License on the card |
|---|---|---|---|---|
| `who_and_when` | [Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When) @ `59b9fcb` | 184 | run, step | not declared |
| `mast` | [mcemri/MAST-Data](https://huggingface.co/datasets/mcemri/MAST-Data) @ `95118ac` | 1,642 | run | cc-by-4.0 |
| `trail` | [PatronusAI/TRAIL](https://huggingface.co/datasets/PatronusAI/TRAIL) @ `b424ce6` | 148 | run, span | mit |
| `tau_bench` | [AgentSuite/tau-bench-trajectories](https://huggingface.co/datasets/AgentSuite/tau-bench-trajectories) @ `382e57d` | 4,950 | run | not declared |
| `tau2_bench` | [AgentSuite/tau2-bench-trajectories](https://huggingface.co/datasets/AgentSuite/tau2-bench-trajectories) @ `07264f1` | 8,340 | run | not declared |
| `agent_reward_bench` | [McGill-NLP/agent-reward-bench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench) @ `b6d17e6` | 1,302 | run | not declared |
| `agentdojo` | [sungjuncho/agentdojo-trajectories](https://huggingface.co/datasets/sungjuncho/agentdojo-trajectories) @ `43df1b8` | 15,095 | call, run, span | not declared |
| `swe_rebench_openhands` | [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) @ `3545538` | 67,074 | run | cc-by-4.0 |
| `swe_agent_nebius` | [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) @ `68195a1` | 80,036 | run, step | cc-by-4.0 |
| `swe_gym_openhands` | [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) @ `baf3a4e` | 6,055 | run | not declared |
| `swe_smith` | [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories) @ `08e109b` | 24,100 | run, step | mit |
| `adp` | [neulab/agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection) @ `31a76bf` | 200,914 | run | not declared |
| `trace_commons` | [trace-commons/agent-traces](https://huggingface.co/datasets/trace-commons/agent-traces) @ `112ebd4` | 28 | none | cc-by-4.0 |
| `catchbench_pre` | [yzhao062/catchbench](https://github.com/yzhao062/catchbench) @ `30a516c` | 1,187 | run, tool | MIT; rows name their upstream license |

> [!WARNING]
> TRAIL's harness logs contain GAIA reference answers and SWE-bench gold patches. agentpile keeps them as recorded. Remove them before you show a TRAIL run to an agent under test, using the `gold_note` of the TRAIL manifest, which names each field.

<details>
<summary>Notes on the table and on single datasets</summary>

<br>

- **Annotation levels.** The column lists the levels at which each source annotates its runs, as its manifest records them. Readers turn part of these annotations into `run.labels`: run labels for every dataset except ADP, plus step and agent labels for Who&When, span labels for TRAIL and tool labels for CatchBench PRE. The rest stay elsewhere in the record (e.g., ADP's results in `run.outcome`, SWE-smith's patch in `run.dataset`).
- **TRAIL** is gated on Hugging Face: accept its terms on the dataset page with the account whose token you use.
- **AgentRewardBench** run files total about 30 GB, because each step keeps the browser state the agent saw. `limit=` fetches only the run files it reads.
- **SWE-smith** loads its `tool` split. 14,984 of its 24,100 rows write actions as text and load as plain messages, and `message_format="native"` returns only the 9,116 rows with structured calls.
- **Trace Commons** loads only its Claude Code sessions, 28 at the pinned revision: enough for tests and examples, too few to draw conclusions from. Its Cursor and opencode files have no reader.
- **ADP** loads the 14 configurations whose records pass validation and a count check, and the manifest lists the rest. The loader follows ADP's own ADP-to-ATIF converter with two changes, both to keep the source's roles. An environment observation that answers no call becomes a user step whose `role` is `environment`, where ADP's converter makes it an empty agent step. A call's result keeps the observation's own `source`, where ADP's converter relabels `user` as `environment`.

</details>

## 4. Add a Dataset

Have agent trajectories or a trace format that agentpile does not load yet? [Open an issue](https://github.com/yzhao062/agentpile/issues/new?template=add-a-dataset.yml) with a link and we write the loader, the manifest and the tests. A pull request with a loader is welcome too (see [Extending](#62-extending)).

## 5. Capturing a Running Agent

`ap.capture` records a run of a Pydantic AI agent or an OpenAI Agents SDK agent into the same record, with the agent's configuration attached:

```bash
pip install "agentpile[pydantic-ai]"     # or "agentpile[openai-agents]"
```

```python
with ap.capture(agent=my_agent) as cap:
    my_agent.run_sync("Book a flight from LAX to SFO.")
cap.run            # the captured run
cap.otlp           # the same spans as an OTLP JSON document
ap.config_from(my_agent)   # the configuration alone: instructions, model, tools with schemas, handoffs
```

Capture is tested with Pydantic AI 2.58.0, 2.59.0 and 2.60.0 and with the OpenAI Agents SDK 0.24.0 through openinference-instrumentation-openai-agents 2.5.3. agentpile checks the version of the instrumentation that wrote the spans, which is Pydantic AI itself or, for the Agents SDK, OpenInference. With any other instrumentation version `cap.run` raises an error, and `ap.capture(my_agent, allow_untested_version=True)` builds the run anyway and marks it `source.dialect_tested` false.

<details>
<summary>Which runs a capture records</summary>

<br>

A capture uses its own tracer provider and never sets the global one. Observers the host program already had keep receiving every span, and the agent's original instrumentation setting is restored when the block exits, including on an exception. Everything a capture records happens while the block is open and only for the agents passed to `capture()`. Runs of other agents in the same window are not recorded. What counts as a selected agent's run depends on the framework. For the Agents SDK, a trace in which a selected agent takes part (it starts the run, receives a handoff, or is called as a tool) is recorded whole, other participants included. Because the SDK names agents only by name in its spans, an unselected agent that shares a selected agent's name is also recorded. For Pydantic AI, each selected agent's own runs are recorded: an agent that a selected agent calls from a tool is recorded only when it is selected too. Pass every agent whose delegated work you want.

</details>

## 6. Reference

### 6.1 The Record

A trajectory record is an ATIF v1.7 trajectory, the Agent Trajectory Interchange Format of the Harbor evaluation framework. What ATIF has no field for sits under `extra["agentpile"]`: where the run came from, the source's labels with their targets, the agents and tools with their permissions, and the source value behind each message or call stand-in. A configuration with no execution (e.g., an MCP tool list) is a config record (`agentpile-config/0`) with the same agents, tools and labels at its top level and no steps.

Loading never adds what the source did not record: it keeps any dependencies the input states and infers none. The few pairings a reader has to infer carry `paired_by` with the name of the rule, and `ap.derive_deps(run, rules=[...])` infers dependencies between steps when you ask.

<details>
<summary>What the extension holds, and how inference and export work</summary>

<br>

- **Source and identity:** dataset, revision, file, record key and reader version, enough to trace every run back to the row it came from.
- **Labels with targets:** each label points at the run, an agent, a step, a call, a span or a tool, with the label's origin (expert, LLM judge, benchmark).
- **Agents and tools:** each agent's instructions, model and tools, and each tool's original definition, its permission, and where that permission came from.
- **Stand-ins made visible:** where ATIF requires a value the source did not have (a non-null message, an object of arguments, a call id, an image part it cannot hold), the record keeps the source value and the event view reports it. A missing value, a null and an empty string therefore stay distinct. A call whose source arguments were neither a JSON object nor null keeps them in `arguments_raw` and carries `arguments_state: "unparsed"`, and in `run.to_otlp()` its model span and its tool span both show that source value. For any other call the model span shows the parsed arguments, and the tool span shows the source text where the reader kept it.
- **Source order:** `run.events()` follows the order the source recorded, which differs from ATIF where ATIF stores a result on the step that issued the call (a delegated agent's steps can come in between). A reader whose references are not integers (e.g., TRAIL's span addresses) stores an integer sort position beside each reference (`source_pos`, `result_pos`), and the reference itself keeps pointing back into the source.

CatchBench PRE rows load as config records, and so do tool definitions: an MCP `tools/list` result, an OpenAI tool list and an Anthropic tool list (readers `mcp-tools-list`, `openai-tools` and `anthropic-tools`), each definition kept verbatim. An MCP tool's permission comes from its annotations when it states them and is `unknown` otherwise.

A Claude Code session file (`~/.claude/projects/<project>/<session>.jsonl`) loads by its path with reader `claude-code-jsonl`, and its subagent files load with it, each under the call that started it. Its format is Claude Code's own and has no public specification. The reader was built on 2.1.x files; it records the versions it saw, and `run.dataset["accounting"]` counts every line it did not map. A session that was edited or rewound keeps every branch, so an export in file order can hold alternatives, which `run.dataset["branches"]` lists. Local sessions can hold secrets, and agentpile writes what it reads.

Inferred pairings include a result matched to its call by position, a TRAIL history call placed on the latest agent step, and a TRAIL TOOL span tied to its call by a unique match. Every dependency from `ap.derive_deps` names the rule and version that produced it, and `run.with_deps(rules=[...])` returns a copy whose steps carry those entries, which `to_otlp()` then writes as span links. The run you loaded is not changed.

`run.to_otlp()` writes a run as OTLP JSON spans in the layout of Harbor's ATIF-to-OpenTelemetry converter. A run loaded from a trace writes each of its source spans once, and a telemetry tool therefore counts the same model and tool operations as in the original trace. The spans agentpile adds around them (the trajectory, user and system steps) are CHAIN spans, not model or tool operations. Reading an OTLP trace whose instrumentation version differs from the tested ones raises an error unless you pass `allow_untested_version=True`, which marks the record as read from an untested version.

</details>

### 6.2 Extending

A reader is any object with `sniff(source) -> bool` and `read(source) -> Iterator[Run]`. Register one for a session with `ap.register("my-format", reader)`, or ship it in a package through the entry-point group `agentpile.readers` (datasets and writers have `agentpile.datasets` and `agentpile.writers`). Pass `reader="..."` to `ap.load` to skip detection.

### 6.3 Downloads and Cache

Downloaded files live in one place, `~/.cache/agentpile/<dataset>/<revision>/` (set `AGENTPILE_CACHE` to move it), and never in the shared Hugging Face cache. A load with `limit=` fetches only the files its runs come from, and a later full load fetches the rest. Until a full download completes, each limited load still asks the hub for its file listing. `ap.cache_info()` lists what is there and `ap.clear_cache()` removes it.

## 7. Status

Alpha. Version 0.2.0 covers the 13 datasets above, each checked against a manifest of pinned counts, the ATIF, OpenAI-style message, OTLP JSON, tool-definition and agentpile config formats, capture for Pydantic AI and the OpenAI Agents SDK, and chat rows for training and evaluation; the tools in section 1.4 are checked weekly. The record format may still change before 1.0, and the [changelog](https://github.com/yzhao062/agentpile/blob/main/CHANGELOG.md) lists every change.

## 8. License

Apache-2.0 for the code. Each dataset keeps its publisher's terms. Several declare no license on their dataset card, and agentpile records what each card says in its manifest. The tests use synthetic fixtures in each dataset's file layout and copy no dataset content. The name follows the habit of calling a mixed collection a pile and has no connection to EleutherAI's The Pile.

<p align="right"><a href="#agentpile">back to top</a></p>
