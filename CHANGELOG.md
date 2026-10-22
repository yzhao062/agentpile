# Changelog

All notable changes to agentpile are listed here. Versions follow [Semantic Versioning](https://semver.org/); before 1.0, a minor version may change the record format, and the entry says how.

## 0.1.0 (2026-10-20)

The first release.

### Added

- `ap.load(name)` for 13 public agent-trace datasets, each at a pinned revision and checked against the run count in its manifest: Who&When, MAST, TRAIL, tau-bench, tau2-bench, AgentRewardBench, AgentDojo, SWE-rebench OpenHands, SWE-agent (Nebius), SWE-Gym OpenHands, SWE-smith, ADP and CatchBench PRE.
- `ap.load(path)` and `ap.load(record)` for ATIF v1.7, OpenAI-style message lists, OTLP JSON (OpenInference and OTel GenAI attributes), MCP `tools/list` results, OpenAI and Anthropic tool lists, and agentpile config records. The format is detected, and an input that more than one reader claims is an error.
- The run record: an ATIF v1.7 trajectory with an `extra["agentpile"]` extension for source identity, labels with targets, agents and tools with permissions, source spans, and the source values behind every stand-in ATIF requires. A configuration without execution is an `agentpile-config/0` record. Harbor 0.22.0's ATIF models accept every trajectory checked: the first 3,000 runs of each dataset, or all of a smaller one.
- `Run.events()`: messages, tool calls and tool results in the order the source recorded them, with each event's address in the record.
- `ap.derive_deps(run, rules=[...])` with six named, versioned rules (`tool_call_id_v1`, `full_context_v1`, `resource_chain_v1`, `last_writer_v1`, `prior_reads_v1`, `value_flow_v1`), and `Run.with_deps(rules=[...])`, which returns a copy whose steps carry the entries.
- `run.to_json()` and `run.to_otlp()`; an OTLP export reads back to an equal record, and a run loaded from a trace writes each source span once.
- `ap.capture(agent=...)` and `ap.config_from(agent)` for Pydantic AI and the OpenAI Agents SDK.
- `ap.validate(record)`, the JSON schema of the extension, and plugin registries for readers, writers and datasets (`ap.register` and the `agentpile.readers`, `agentpile.writers` and `agentpile.datasets` entry-point groups).
- One download cache, `~/.cache/agentpile/` (`AGENTPILE_CACHE`), with `ap.cache_info()` and `ap.clear_cache()`.
