# Changelog

All notable changes to agentpile are listed here. Versions follow [Semantic Versioning](https://semver.org/); before 1.0, a minor version may change the record format, and the entry says how.

## 0.1.0 (2026-10-21)

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
