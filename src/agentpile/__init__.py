"""agentpile: load public agent-trajectory datasets and agent traces into one run record.

    import agentpile as ap

    for run in ap.load("who_and_when"):   # a dataset at its pinned revision
        run.labels, run.agents, run.steps
    run = next(ap.load("trace.json"))     # a file; the format is detected, never guessed
    run.to_json("run.json"); run.to_otlp()
    ap.to_chat(ap.load("tau_bench"), "chat.jsonl")   # OpenAI-style chat rows with tools

A trajectory record is an ATIF v1.7 trajectory with agentpile fields under
``extra["agentpile"]``; a config record is ``agentpile-config/0`` JSON.
``ap.datasets.names()`` lists the datasets, ``ap.readers.names()`` the file
formats. Downloads go to ``ap.cache_root()`` (``~/.cache/agentpile`` unless
``AGENTPILE_CACHE`` is set); ``ap.cache_info()`` and ``ap.clear_cache()``
list and remove them.
"""

from ._capture import CaptureError, capture, config_from
from ._chat import to_chat
from ._graph import to_networkx
from ._send import send_otlp
from ._show import show
from ._tables import table, to_pandas
from .cache import cache_info, cache_root, clear_cache
from .deps import derive_deps
from .errors import AgentpileError, AmbiguousSource, GatedDataset, UnrecognizedSource, UnsupportedVersion
from .loading import load
from .record import ATIF_VERSION, CONFIG_SCHEMA, EXT, MISSING, Event, Run
from .registry import datasets, readers, register, writers
from .validate import validate

__version__ = "0.3.0"

# Built-in plugins go in the same registries as installed ones, as lazy
# module strings, so importing agentpile imports none of them. They are added
# before entry points are read, so a plugin cannot shadow them by name.
readers._builtin("atif", "agentpile.formats.atif")
readers._builtin("agentpile-config", "agentpile.formats.config")
readers._builtin("openai-messages", "agentpile.formats.messages")
readers._builtin("otlp-json", "agentpile.formats.otlp")
readers._builtin("mcp-tools-list", "agentpile.formats.tools:mcp")
readers._builtin("openai-tools", "agentpile.formats.tools:openai")
readers._builtin("anthropic-tools", "agentpile.formats.tools:anthropic")
readers._builtin("claude-code-jsonl", "agentpile.formats.claude_code")
writers._builtin("json", "agentpile.formats.jsonfile")
writers._builtin("otlp-json", "agentpile.formats.otlp")
writers._builtin("chat-jsonl", "agentpile.formats.chat")
datasets._builtin("who_and_when", "agentpile.corpora.who_and_when:dataset")
datasets._builtin("mast", "agentpile.corpora.mast:dataset")
datasets._builtin("catchbench_pre", "agentpile.corpora.catchbench_pre:dataset")
datasets._builtin("tau_bench", "agentpile.corpora.tau_bench:dataset")
datasets._builtin("tau2_bench", "agentpile.corpora.tau2_bench:dataset")
datasets._builtin("swe_gym_openhands", "agentpile.corpora.swe_gym_openhands:dataset")
datasets._builtin("swe_rebench_openhands", "agentpile.corpora.swe_rebench_openhands:dataset")
datasets._builtin("agentdojo", "agentpile.corpora.agentdojo:dataset")
datasets._builtin("trail", "agentpile.corpora.trail:dataset")
datasets._builtin("agent_reward_bench", "agentpile.corpora.agent_reward_bench:dataset")
datasets._builtin("swe_agent_nebius", "agentpile.corpora.swe_agent_nebius:dataset")
datasets._builtin("swe_smith", "agentpile.corpora.swe_smith:dataset")
datasets._builtin("adp", "agentpile.corpora.adp:dataset")
datasets._builtin("trace_commons", "agentpile.corpora.trace_commons:dataset")

__all__ = [
    "ATIF_VERSION",
    "AgentpileError",
    "AmbiguousSource",
    "CaptureError",
    "CONFIG_SCHEMA",
    "EXT",
    "Event",
    "GatedDataset",
    "MISSING",
    "Run",
    "UnrecognizedSource",
    "UnsupportedVersion",
    "cache_info",
    "cache_root",
    "capture",
    "clear_cache",
    "config_from",
    "datasets",
    "derive_deps",
    "load",
    "readers",
    "register",
    "send_otlp",
    "show",
    "table",
    "to_chat",
    "to_networkx",
    "to_pandas",
    "validate",
    "writers",
]
