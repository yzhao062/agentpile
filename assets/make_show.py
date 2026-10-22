"""Draw the README's ap.show screenshot: one Who&When run, rendered by rich and saved as SVG.

Usage: python assets/make_show.py   (needs agentpile[hf,view]; downloads the few Who&When files it reads;
writes assets/show.svg)
"""

import os
from pathlib import Path

from rich.console import Console

import agentpile as ap

RUN = "Algorithm-Generated/57"  # a short run whose mistake labels sit on its first step
OUT = Path(__file__).with_name("show.svg")


def main():
    run = next(r for r in ap.load("who_and_when", limit=80) if r.id == RUN)
    console = Console(record=True, width=110, file=open(os.devnull, "w"))
    ap.show(run, max_chars=150, file=console)
    svg = console.export_svg(title="ap.show(run)")
    OUT.write_text("\n".join(line.rstrip() for line in svg.splitlines()) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
