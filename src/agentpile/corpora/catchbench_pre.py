"""CatchBench PRE (github.com/yzhao062/catchbench, data/pre): agent configurations with excess-permission labels.

Each row describes one agent's declared capabilities before it runs (a CrewAI
agent, an MCP server, an n8n workflow, a SWE-agent config, an InjecAgent
roster, or a synthetic case), so it becomes a config record: no steps are
invented. Capability names are unique within a row and become the tool ids;
the row's ``permission_level`` is the tool permission, declared by the
benchmark. ``excess_set`` gives one tool-level label per capability;
``minimal_reference`` and InjecAgent's ``attack_type`` are run-level labels.
The committed rows carry ``spec_tokens`` instead of the original prose, so
the agent's instructions stay unknown rather than rebuilt from tokens. The
whole row is kept in ``dataset``.

The six files are fetched from GitHub at the pinned commit (standard library
only) into ``<cache root>/catchbench_pre/<commit>/``.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from ..cache import dataset_dir
from ..record import CONFIG, CONFIG_SCHEMA, Run
from ._base import Dataset

CORPORA = ("crewai", "injecagent", "mcp", "n8n", "sweagent", "synthetic")
RAW_URL = "https://raw.githubusercontent.com/{repo}/{revision}/data/pre/{corpus}.json"


class CatchBenchPre(Dataset):
    name = "catchbench_pre"
    format = "catchbench-pre-json"

    @property
    def repo_id(self) -> str:
        return self.manifest["github"]

    def files(self) -> Path:
        root = dataset_dir(self.name, self.revision)
        target = root / "data" / "pre"
        target.mkdir(parents=True, exist_ok=True)
        for corpus in CORPORA:
            path = target / f"{corpus}.json"
            if path.is_file():
                continue
            url = RAW_URL.format(repo=self.repo_id, revision=self.revision, corpus=corpus)
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            json.loads(data)  # refuse to cache a non-JSON response
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        return root

    def records(self, root: Path, corpus: str = None, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"catchbench_pre takes corpus= only, got {sorted(options)}")
        if corpus is not None and corpus not in CORPORA:
            raise ValueError(f"corpus must be one of {list(CORPORA)}, got {corpus!r}")
        for name in CORPORA:
            if corpus is not None and name != corpus:
                continue
            rel = f"data/pre/{name}.json"
            with (root / rel).open("r", encoding="utf-8") as fh:
                rows = json.load(fh)
            for row in rows:
                yield str(row.get("instance_id")), row, {"file": rel, "config": name, "split": None}

    def convert(self, key: str, raw: Dict[str, Any], where: Dict[str, Any]) -> Run:
        caps = raw.get("declared_capabilities") or []
        tools = {}
        for cap in caps:
            tools[cap["name"]] = {
                "name": cap["name"], "agent_or_server": "agent", "type": cap.get("type"),
                "definition": None, "definition_format": None, "atif_index": None,
                "permission": cap.get("permission_level", "unknown"), "permission_from": "declared",
            }
        labels_raw = raw.get("labels") or {}
        label_source = labels_raw.get("label_source") or "unknown"
        labels: List[Dict[str, Any]] = [
            {"name": "excess_capability", "target": {"kind": "tool", "address": {"trajectory": key, "tool": name}},
             "value": True, "label_source": label_source, "source_ref": {"field": "labels.excess_set"}}
            for name in labels_raw.get("excess_set") or []]
        labels.append({"name": "minimal_reference", "target": {"kind": "run", "address": {"trajectory": key}},
                       "value": list(raw.get("minimal_reference") or []), "label_source": "catchbench",
                       "source_ref": {"field": "minimal_reference"}})
        if "attack_type" in labels_raw:
            labels.append({"name": "attack_type", "target": {"kind": "run", "address": {"trajectory": key}},
                           "value": labels_raw["attack_type"], "label_source": label_source,
                           "source_ref": {"field": "labels.attack_type"}})
        source = self.source(key, where)
        source["config"] = raw.get("source", where.get("config"))  # the row's own corpus label (synthetic.json rows say synth_crewai)
        record = {
            "schema": CONFIG_SCHEMA,
            "source": source,
            "task": {"id": key, "text": None},
            "agents": [{"id": "agent", "name": None, "instructions": None, "model": None,
                        "tools": [cap["name"] for cap in caps], "handoffs": []}],
            "tools": tools,
            "labels": labels,
            "dataset": raw,
        }
        return Run(record, CONFIG)


dataset = CatchBenchPre()
