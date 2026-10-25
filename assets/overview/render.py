"""Render the README overview figure: assets/overview/overview.html to assets/overview.png.

Usage: python assets/overview/render.py   (needs Google Chrome or Chromium; set CHROME to its path or name if it is not found)

The circle areas in the figure are proportional to the run counts in the page's data-runs attributes.
Before rendering, the script checks those counts and the figure's other dataset claims against the bundled manifests.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = HERE / "overview.html"
OUT = HERE.parent / "overview.png"
MANIFESTS = HERE.parents[1] / "src" / "agentpile" / "corpora" / "manifests"
WIDTH, HEIGHT, SCALE = 1280, 600, 2
# datasets without a circle, as the figure names them
LISTED = {"mast": "MAST", "agent_reward_bench": "AgentRewardBench", "catchbench_pre": "CatchBench",
          "who_and_when": "Who&amp;When", "trail": "TRAIL", "trace_commons": "Trace Commons"}


def check_counts(page):
    runs = {path.stem: json.loads(path.read_text())["counts"]["runs_after_load"]["value"] for path in MANIFESTS.glob("*.json")}
    circles = re.findall(r'data-dataset="(\w+)"[^>]*data-runs="(\d+)"', page)
    drawn = {key: int(n) for key, n in circles}
    assert len(drawn) == len(circles) == page.count('class="bub"'), "every circle needs one data-dataset and data-runs, each dataset once"
    wrong = {key: (n, runs.get(key)) for key, n in drawn.items() if runs.get(key) != n}
    assert not wrong, f"circle counts differ from the manifests (figure, manifest): {wrong}"
    assert set(runs) == set(drawn) | set(LISTED), f"each dataset needs a circle or a place in the list: {sorted(set(runs) - set(drawn) - set(LISTED))}"
    assert max(runs[key] for key in LISTED) < 2000 and min(drawn.values()) >= 2000, "the listed datasets are the ones under 2,000 runs"
    assert all(name in page for name in LISTED.values()), "the figure names every listed dataset"
    assert len(LISTED) == 6 and "and six with under 2,000 runs each" in page, "the figure says six datasets have under 2,000 runs"
    assert f"{len(runs)} public datasets" in page, f"the figure should say {len(runs)} public datasets"
    total = sum(runs.values())
    assert "about 410,000 runs" in page and abs(total - 410000) <= 5000, f"the figure says about 410,000 runs; the manifests hold {total:,}"


def chrome():
    named = os.environ.get("CHROME")
    for candidate in (named and (shutil.which(named) or named), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                      shutil.which("google-chrome"), shutil.which("google-chrome-stable"), shutil.which("chromium"),
                      shutil.which("chromium-browser")):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("Google Chrome or Chromium not found; set CHROME to its path")


def main():
    check_counts(PAGE.read_text())
    subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--allow-file-access-from-files",
                    f"--force-device-scale-factor={SCALE}", f"--window-size={WIDTH},{HEIGHT}", "--virtual-time-budget=3000",
                    f"--screenshot={OUT}", PAGE.as_uri()], check=True, capture_output=True)
    print(f"wrote {OUT} ({WIDTH * SCALE} x {HEIGHT * SCALE}); pixels depend on the installed fonts")


if __name__ == "__main__":
    main()
