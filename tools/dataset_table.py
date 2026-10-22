"""Print the README's dataset table from the bundled manifests, so the table cannot drift from the pinned facts.

Usage: python tools/dataset_table.py
"""

import json
from pathlib import Path

MANIFESTS = Path(__file__).resolve().parents[1] / "src" / "agentpile" / "corpora" / "manifests"

ORDER = ["who_and_when", "mast", "trail", "tau_bench", "tau2_bench", "agent_reward_bench", "agentdojo",
         "swe_rebench_openhands", "swe_agent_nebius", "swe_gym_openhands", "swe_smith", "adp", "catchbench_pre"]


def label_summary(m):
    levels = {}
    for lab in m.get("labels") or []:
        levels.setdefault(lab.get("level") or "?", []).append(lab.get("field"))
    return ", ".join(sorted(levels)) or "none"


def main():
    rows = ["| Name | Source at pinned revision | Runs | Source annotation levels | License on the card |", "|---|---|---|---|---|"]
    found = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in MANIFESTS.glob("*.json")}
    for name in [n for n in ORDER if n in found] + sorted(set(found) - set(ORDER)):
        m = found[name]
        if m.get("hf_id"):
            src = f"[{m['hf_id']}](https://huggingface.co/datasets/{m['hf_id']})"
        else:
            src = f"[{m['github']}](https://github.com/{m['github']})" if m.get("github") else "?"
        runs = ((m.get("counts") or {}).get("runs_after_load") or {}).get("value")
        rows.append(f"| `{name}` | {src} @ `{m['revision'][:7]}` | {runs:,} | {label_summary(m)} | {m.get('license_on_card') or 'not declared'} |"
                    if isinstance(runs, int) else
                    f"| `{name}` | {src} @ `{m['revision'][:7]}` | ? | {label_summary(m)} | {m.get('license_on_card') or 'not declared'} |")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
