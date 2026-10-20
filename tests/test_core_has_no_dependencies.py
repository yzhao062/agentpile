"""Gate 6: the core imports and reads files with the standard library alone."""

import subprocess
import sys
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"

SCRIPT = f"""
import sys
import agentpile as ap
for path in ("atif_multiagent.json", "atif_two.jsonl", "config_mcp.json"):
    for run in ap.load(r"{FIX}/" + path):
        assert ap.validate(run) == [], path
        list(run.events())
third_party = sorted(m for m in ("pyarrow", "huggingface_hub", "opentelemetry", "pydantic", "yaml", "numpy", "pandas")
                     if m in sys.modules)
print(",".join(third_party))
"""


def test_import_and_read_pull_in_no_third_party_module():
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
