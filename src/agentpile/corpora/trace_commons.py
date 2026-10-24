"""Trace Commons (trace-commons/agent-traces): coding-agent sessions donated as each agent's own session file.

One run per Claude Code session file under ``sessions/claude_code/``, read by
the ``claude-code-jsonl`` reader (``agentpile.formats.claude_code``); the run id
is the file name without ``.jsonl``. The Cursor, opencode and other harness
folders are not read, since agentpile has no reader for those formats. Each
session was anonymized by its contributor before upload (dataset card) and
carries no labels. At the pinned revision there are 28 sessions, enough for
tests and examples and too few to draw conclusions from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from ..formats.claude_code import session_to_trajectory
from ..record import TRAJECTORY, Run
from ._base import Dataset

PATTERN = "sessions/claude_code/*.jsonl"


class TraceCommons(Dataset):
    name = "trace_commons"
    format = "claude-code-jsonl"
    incremental = True

    def records(self, root: Path, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        if options:
            raise TypeError(f"trace_commons takes no options, got {sorted(options)}")
        for path in self.paths(root, PATTERN):
            rel = path.relative_to(root).as_posix()
            yield path.name[: -len(".jsonl")], path, {"file": rel, "config": "claude_code", "split": None}

    def convert(self, key: str, raw: Path, where: Dict[str, Any]) -> Run:
        return Run(session_to_trajectory(self.ready(raw), trajectory_id=key, ext={"source": self.source(key, where)}),
                   TRAJECTORY)


dataset = TraceCommons()
