# Changelog

All notable changes to agentpile are listed here. Versions follow [Semantic Versioning](https://semver.org/); before 1.0, a minor version may change the record format, and the entry says how.

## 0.3.0 (2026-10-24)

A reader for the session files Claude Code writes, with their subagent files, and the `trace_commons` dataset of 28 public sessions read with it. The record format and every existing reader are unchanged.

### Added

- A reader for the session files Claude Code writes under `~/.claude/projects/<project>/<session>.jsonl`, registered as `claude-code-jsonl`, so `ap.load(path)` reads a local session and every existing output (`run.events()`, tables, `to_chat`, `to_otlp`, `show`) works on it. Claude Code writes one record per content block; the records of one message become one agent step, placed at the message's first line, and each tool result pairs with its call by id even when Claude Code wrote the result before the rest of the message. Subagent files under `<session>/subagents/` load with the session as subagent trajectories, linked to the Agent call that names their id or the Workflow call that names their folder. Attachments are kept verbatim; other bookkeeping records are counted, every line of the file is accounted for in `run.dataset["accounting"]`, and an index of every record's `uuid` and parent keeps the session's tree, including edited or rewound branches, which are labelled and never pruned. The format is Claude Code's own and has no public specification; the reader was built on 2.1.x files.
- The `trace_commons` dataset: the 28 Claude Code sessions of `trace-commons/agent-traces`, read with the new reader at a pinned revision.

## 0.2.0 (2026-10-23)

Chat rows for training and evaluation, OTLP output that follows the OpenInference and OTel GenAI conventions, a labels table that saves to Parquet, and recipes for downstream tools. The labels table's `value` column changes type, which can silently empty an existing filter (see Changed).

### Added

- `ap.to_chat(runs, path=None, *, perspective=None, arguments="json", report=None)` turns trajectories into OpenAI-style chat rows, `{"messages": [...], "tools": [...]}`, returned as a list or written as JSON Lines to `path`; the `chat-jsonl` writer does the same for one run. The rows have the message shape that OpenAI fine-tuning files, Hugging Face chat templates and message-list auditors use, and writing them needs only the standard library. Messages follow `run.events()`, each tool result follows the assistant turn that issued its call, and tool definitions in the OpenAI, MCP or Anthropic shape become OpenAI function tools. With `perspective="X"`, agent X is the assistant and every other agent's turns become user text that starts with the agent's id; tau2-bench's simulated user is user text either way. Where the chat shape needs a value the source lacks, the row holds `{}`, `{"_raw": ...}` or `""`. The optional `report` counts each such change and gives the position and address of every event behind each message, so a row can be traced back to the record.
- Seven recipes in `examples/integrations/` hand agentpile output to downstream tools and check the result: tool-call scoring with Ragas and DeepEval, fine-tuning input with Hugging Face `datasets` and TRL, rule checks with Invariant Guardrails, table queries with DuckDB and Polars, and viewing in Harbor, Inspect AI and OTLP trace viewers (Phoenix, Jaeger, MLflow). Each pins its own environment and runs offline on the bundled fixtures. A CI workflow runs them weekly, and the README's Works With table shows each check's status and the date it last passed. The recipes are examples, not part of the package.

### Changed

- The labels table of `ap.table(runs, "labels")` and `ap.to_pandas(runs, "labels")` stores each `value` as text, or `None` for a null value. A new `value_type` column records `bool`, `int`, `float`, `str`, `json` or `None`. Keep `value` as it is when `value_type` is `str` or `None`, and use `json.loads(value)` otherwise; in `to_pandas` these two columns keep `None` rather than NaN, so the same rule works on a DataFrame. A value JSON cannot hold (a date, a tuple) comes back in its JSON form, and `run.labels` keeps the original typed values. Datasets label with booleans, numbers and strings alike, so the old column mixed types, and saving the labels table to Parquet (or any other Arrow conversion) failed for 7 of the 13 dataset fixtures and for all 13 together. A filter written for the old types needs the new form: `df[df.value == True]` now matches nothing, and `df[(df.value_type == "bool") & (df.value == "true")]` replaces it. `df.value.astype(float)` still works on numeric labels.
- OTLP export follows two pinned conventions: the attribute names of OpenInference 0.1.41, and the OTel GenAI attribute names and value schemas at revision `9ffc7e00c639be7712e544e3e54331e885dccf84`. The record attributes (`agentpile.*`) are unchanged, and the reader still rebuilds the exact record, from new exports and from those 0.1.x wrote. What a trace viewer shows changes:
  - Tool spans that agentpile writes sit under the span of the step that made the call instead of directly under the trajectory span. Every span agentpile adds lies inside its parent and takes time, except under a source span that itself takes none; a source span without an end time counts as one, and a missing start time no longer lands after the recorded end. In 0.1.x, 187 of the 535 spans agentpile added across the fixtures lasted zero time: every tool span, and steps such as a trajectory's last one when it called no tool. A step whose source timestamp falls outside the source span or call it is drawn under moves inside it, marked `agentpile.time_synthetic`; the record keeps the timestamp.
  - On a tool span, `input.value` and `output.value` give the call's arguments and result as the source recorded them. `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result` now appear only for a JSON object, as the GenAI schemas require; on the dataset fixtures this leaves out 6 of 94 argument values, the ones that did not parse, and 72 of 78 results, mostly plain text. Code that read either attribute for text should read `input.value` or `output.value`. A new `tool.parameters` holds the tool's parameter schema from an OpenAI, MCP or Anthropic definition, left out where two tools of the call's name could have different ones.
  - Model spans show in `input.value` what came in since the previous model call: system and user messages, the text of a step that made no model call (a scripted greeting), and tool results with their call ids. The value is marked `agentpile.input_reconstructed`, since ATIF does not record the prompt a model was given; `llm.input_messages` and `gen_ai.input.messages` stay absent.
  - The resource's `service.name` falls back to the run's dataset when the agent's name is missing or `unknown`, and to `agentpile` without a dataset. `openinference.project.name` is the dataset, so Phoenix files each dataset under its own project. The dataset goes by the name `ap.load` takes, such as `tau_bench`; a record from a dataset agentpile does not register keeps the source's own id, such as a hub repository.
  - Attributes the record does not hold stay absent, including ones the conventions require, such as `gen_ai.provider.name`. Spans adopted from a source trace keep their attributes, parents and times.

### Fixed

- A model span in an OTLP export names a model when its step records none: the model the record's roster gives the step's agent, else the trajectory's model. tau-bench records its model only once per run, so its model spans named none before. A step the harness scripted, such as tau2-bench's fixed greeting, still names none, and the record itself is unchanged. The span is then named `chat <model>`.

## 0.1.2 (2026-10-22)

### Added

- Capture and the OTLP reader accept spans from Pydantic AI 2.60.0, which builds the same records as 2.58.0 and 2.59.0 from the fixture scripts. With 0.1.1, `cap.run` raised `UnsupportedVersion` for it, and 2.60.0 is what `pip install "agentpile[pydantic-ai]"` now installs.

## 0.1.1 (2026-10-22)

Fixes found by running 0.1.0 on Google Colab, where the preinstalled google-adk caps OpenTelemetry at 1.42.1.

### Changed

- The `otel` extra installs only `opentelemetry-proto`, which `ap.send_otlp` uses for a receiver that takes only protobuf. The OpenTelemetry SDK comes with the `pydantic-ai` and `openai-agents` extras, since capture is the only code that uses it.
- The OpenTelemetry lower bound of the `otel`, `pydantic-ai` and `openai-agents` extras is 1.42.1, down from 1.45.1, so installing them on Colab no longer upgrades OpenTelemetry past what google-adk accepts. The tests pass on both versions.

### Fixed

- Loading a gated dataset (TRAIL) without access raises `ap.GatedDataset`, which names the page where you accept the terms and the ways to give huggingface_hub a token (`hf auth login`, `HF_TOKEN`, Colab Secrets). The hub's own error is kept as the cause.
- `ap.show(run, file=...)` writes to `file` inside a Jupyter notebook too. Before, rich drew the view as cell output and the file stayed empty. Without `file`, a notebook still shows the rich view as cell output.
- The capture error for a missing OpenTelemetry SDK names the extra of the framework being captured instead of `otel`.
- The README's compare-failures example says that TRAIL is gated.

## 0.1.0 (2026-10-22)

The first release.

### Added

- `ap.load(name)` for 13 public agent-trace datasets, each at a pinned revision and checked against the run count in its manifest: Who&When, MAST, TRAIL, tau-bench, tau2-bench, AgentRewardBench, AgentDojo, SWE-rebench OpenHands, SWE-agent (Nebius), SWE-Gym OpenHands, SWE-smith, ADP and CatchBench PRE. With `limit=N`, a Hugging Face dataset downloads only the files its first N runs come from (five AgentRewardBench runs fetch about 25 MB of its 30 GB); a later full load fetches the rest and runs the count check.
- `ap.load(path)` and `ap.load(record)` for ATIF v1.7, OpenAI-style message lists, OTLP JSON (OpenInference and OTel GenAI attributes), MCP `tools/list` results, OpenAI and Anthropic tool lists, and agentpile config records. The format is detected, and an input that more than one reader claims is an error.
- The run record: an ATIF v1.7 trajectory with an `extra["agentpile"]` extension for source identity, labels with targets, agents and tools with permissions, source spans, and the source values behind every stand-in ATIF requires. A configuration without execution is an `agentpile-config/0` record. Harbor 0.22.0's ATIF models accept every trajectory checked: the first 3,000 runs of each dataset, or all of a smaller one.
- `Run.events()`: messages, tool calls and tool results in the order the source recorded them, with each event's address in the record.
- `ap.derive_deps(run, rules=[...])` with six named, versioned rules (`tool_call_id_v1`, `full_context_v1`, `resource_chain_v1`, `last_writer_v1`, `prior_reads_v1`, `value_flow_v1`), and `Run.with_deps(rules=[...])`, which returns a copy whose steps carry the entries.
- `run.to_json()` and `run.to_otlp()`; an OTLP export reads back to an equal record, and a run loaded from a trace writes each source span once.
- `ap.show(run)`: a terminal view of one run, with labels beside the events they target and the record's stand-in and pairing marks; rich output with the `view` extra, plain text without it.
- `ap.table(runs, level)` for flat rows of runs, events or labels (standard library), `ap.to_pandas(runs, level)` (`pandas` extra), and `ap.to_networkx(run, rules)`, an event graph with pairings and inferred dependencies as edges (`graph` extra).
- `ap.send_otlp(runs, endpoint)`: posts each trajectory's OTLP JSON to a tracing backend such as Jaeger or MLflow, and switches to protobuf for one that answers 415, such as Arize Phoenix (needs `opentelemetry-proto` from the `otel` extra).
- `ap.capture(agent=...)` and `ap.config_from(agent)` for Pydantic AI (2.58.0 and 2.59.0) and the OpenAI Agents SDK (0.24.0, through openinference-instrumentation-openai-agents 2.5.3). Spans from any other version of the instrumentation (Pydantic AI's own, or OpenInference's for the Agents SDK) raise `UnsupportedVersion` unless `allow_untested_version=True`, which builds the run and marks it.
- `ap.validate(record)`, the JSON schema of the extension, and plugin registries for readers, writers and datasets (`ap.register` and the `agentpile.readers`, `agentpile.writers` and `agentpile.datasets` entry-point groups).
- One download cache, `~/.cache/agentpile/` (`AGENTPILE_CACHE`), with `ap.cache_info()` and `ap.clear_cache()`.
