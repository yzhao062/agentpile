"""Draw the README overview figure: three kinds of input, one run record, three ways to use it.

Usage: python assets/make_overview.py   (needs matplotlib; writes assets/overview.png)

The palette and type follow the GRADE and auditable README figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]

TEAL, CORAL = "#4098AC", "#ED8D5A"
TEAL_FILL, CORAL_FILL, GRAY_FILL = "#E3F1F4", "#FBEAE1", "#F3F3F3"
NEAR_BLACK, SUBTITLE, GRAY_EDGE = "#1a1a1a", "#666666", "#9A9A9A"
CODE = "#9A4A33"

OUT = Path(__file__).with_name("overview.png")

INPUTS = [
    ("13 public datasets", "Who&When, TRAIL, tau-bench, ADP\nand nine more: about 410,000 runs", 'ap.load("trail")'),
    ("Trace and tool files", "ATIF, OpenAI messages, OTLP JSON;\nMCP, OpenAI and Anthropic tool lists", 'ap.load("trace.json")'),
    ("Live agent runs", "Pydantic AI and the OpenAI Agents SDK,\nwith the agent's configuration", "ap.capture(agent)"),
]
OUTPUTS = [
    ("Read a run", "messages, calls and results in source\norder, labels beside what they target", "ap.show(run)"),
    ("Analyze many runs", "pandas tables, networkx graphs,\ndependencies inferred on request", "ap.to_pandas(runs)"),
    ("Hand it on", "JSON for either kind of record;\ntrajectory spans to Phoenix or Jaeger", "run.to_json()  ap.send_otlp(run)"),
]


W, H = 10.0, 5.1       # figure size in inches; the axes use inches too, so font sizes and spacing agree
CW, CH, GAP = 3.15, 1.2, 0.18
TOP = H - 1.08         # top edge of the card columns
LEFT, RIGHT = 0.15, W - 0.15 - CW
RX, RW = 3.75, 2.5     # the record in the middle


def card(ax, x, y, title, body, code, edge, fill):
    ax.add_patch(FancyBboxPatch((x, y), CW, CH, boxstyle="round,pad=0,rounding_size=0.08",
                                linewidth=1.4, edgecolor=edge, facecolor=fill))
    ax.add_patch(FancyBboxPatch((x + 0.08, y + 0.1), 0.055, CH - 0.2, boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=0, facecolor=edge))
    ax.text(x + 0.24, y + CH - 0.15, title, fontsize=12.5, fontweight="bold", color=NEAR_BLACK, va="top")
    ax.text(x + 0.24, y + CH - 0.45, body, fontsize=10, color=SUBTITLE, va="top", linespacing=1.3)
    ax.text(x + 0.24, y + 0.12, code, fontsize=10, color=CODE, family="monospace", va="bottom")


def arrow(ax, start, end, color):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.7, color=color,
                                 connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0))


def band(ax, y0, hh, head, lines):
    """A white band in the record with a monospace head over centered lines, the pair centered in the band."""
    cx = RX + RW / 2
    ax.add_patch(FancyBboxPatch((RX + 0.15, y0), RW - 0.3, hh, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=1.0, edgecolor=CORAL, facecolor="white"))
    block = 0.30 + 0.19 * (lines.count("\n") + 1)
    top = y0 + (hh + block) / 2
    ax.text(cx, top, head, fontsize=10, fontweight="bold", color=CODE, family="monospace", ha="center", va="top")
    ax.text(cx, top - 0.30, lines, fontsize=9.6, color=NEAR_BLACK, ha="center", va="top", linespacing=1.4)


def main():
    fig = plt.figure(figsize=(W, H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(LEFT, H - 0.12, "Many trace formats in, one record format out", fontsize=17, fontweight="bold",
            color=NEAR_BLACK, va="top")
    ax.text(LEFT, H - 0.50, "agentpile keeps what each source recorded, marks where it stands in for a message or a call\n"
            "and which pairings it inferred, and derives dependencies only on request.", fontsize=10.5, color=SUBTITLE,
            va="top", linespacing=1.3)

    ys = [TOP - CH - i * (CH + GAP) for i in range(3)]
    for (title, body, code), y in zip(INPUTS, ys):
        card(ax, LEFT, y, title, body, code, TEAL, TEAL_FILL)
    for (title, body, code), y in zip(OUTPUTS, ys):
        card(ax, RIGHT, y, title, body, code, GRAY_EDGE, GRAY_FILL)

    bottom = ys[2]
    ax.add_patch(FancyBboxPatch((RX, bottom), RW, TOP - bottom, boxstyle="round,pad=0,rounding_size=0.1",
                                linewidth=1.8, edgecolor=CORAL, facecolor=CORAL_FILL))
    cx = RX + RW / 2
    ax.text(cx, TOP - 0.18, "The run record", fontsize=14, fontweight="bold", color=NEAR_BLACK, ha="center", va="top")
    ax.text(cx, TOP - 0.50, "an ATIF v1.7 trajectory per run;\na config record has no steps", fontsize=10,
            color=SUBTITLE, ha="center", va="top", linespacing=1.25)
    band(ax, bottom + 1.75, 1.1, "steps", "messages\ntool calls\nresults")
    band(ax, bottom + 0.15, 1.45, 'extra["agentpile"]',
         "source, revision and row\nlabels, each with its target\nagents and tools, with permissions\n"
         "source values behind stand-ins")

    for y in ys:
        arrow(ax, (LEFT + CW + 0.06, y + CH / 2), (RX - 0.06, y + CH / 2), TEAL)
        arrow(ax, (RX + RW + 0.06, y + CH / 2), (RIGHT - 0.06, y + CH / 2), CORAL)

    fig.savefig(OUT, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
