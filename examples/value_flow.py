"""How far back does an agent reach for the values it uses?

The datasets record which tool call produced which result, but not where a call's arguments came from.
ap.derive_deps(run, rules=["value_flow"]) infers it: a later call depends on the earlier call whose result holds the
same id-like value (a reservation or product number, a date), the latest one if several do. It reads only JSON results,
so it finds no dependency in the plain-text results of OpenHands. Panel (a) draws one small tau_bench run as the graph
that ap.to_networkx returns, one column per step with the dependencies as arcs. Panel (b) counts, for tau_bench,
tau2_bench and the first 1,000 runs of swe_gym_openhands, how many steps lie between a result and the call that reuses
its value, and how many runs have a dependency reaching back three steps or more, counting either every step (user
turns included) or the agent's tool calls alone.

    pip install "agentpile[hf,graph]" matplotlib
    python examples/value_flow.py [OUT]

Downloads about 470 MB once, runs in about 6 s and writes value_flow.png.
"""

import sys
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as BezierPath

import agentpile as ap

SHOWN = "claude-4-opus-thinking-off/airline/1"   # panel (a): a small run with four dependencies, one 5 steps back
FAR = 3                                          # "reaches far" means a dependency at least this many steps back

run = next(r for r in ap.load("tau_bench") if r.id == SHOWN)
deps = ap.derive_deps(run, rules=["value_flow"])        # each entry names its rule and version
graph = ap.to_networkx(run, rules=["value_flow"])       # events as nodes, dependencies as edges


def reach(r):
    """How far back each dependency of a run reaches, in steps and in agent tool calls."""
    found = ap.derive_deps(r, rules=["value_flow"])       # inferred on request, never stored in the data
    calls = [e for e in r.events() if e.kind == "call" and e.actor != "user"]   # not the user simulator's
    order = {(e.address["step"], e.address["call"]): i for i, e in enumerate(calls)}
    steps = [d["dst"]["step"] - d["src"]["step"] for d in found]                # step ids count 1, 2, 3, ...
    in_calls = [order[(d["dst"]["step"], d["dst"]["call"])] - order[(d["src"]["step"], d["src"]["call"])] for d in found]
    return steps, in_calls


def far(per_run):
    """The number of runs with a dependency that reaches back FAR or more."""
    return sum(max(distances, default=0) >= FAR for distances in per_run)


stats = {}
for name, limit in (("tau_bench", None), ("tau2_bench", None), ("swe_gym_openhands", 1000)):   # 1,000 runs come from one shard
    steps, calls = zip(*(reach(r) for r in ap.load(name, limit=limit)))      # downloads the pinned files once
    pooled = [s for per_run in steps for s in per_run]
    stats[name] = {"runs": len(steps), "deps": len(pooled), "steps": pooled, "far_runs": far(steps), "far_call_runs": far(calls)}

# ---- figure ----

TEAL, CORAL, MINT, GOLD = "#4098AC", "#ED8D5A", "#7BC8A4", "#E8B04B"
NEAR_BLACK, SUBTITLE, SPINE, DOT, CORAL_FILL = "#1a1a1a", "#666666", "#9A9A9A", "#C4C4C4", "#FBEAE1"
TEAL_TEXT, MINT_TEXT = "#2F7F91", "#3A936C"          # the same hues, darker, for text on white
CAP = 13                                             # the histogram groups every distance from CAP up
out = sys.argv[1] if len(sys.argv) > 1 else "value_flow.png"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9.5,
                     "axes.edgecolor": SPINE, "axes.linewidth": 1.0, "xtick.color": NEAR_BLACK, "ytick.color": NEAR_BLACK,
                     "axes.labelcolor": SUBTITLE, "text.color": NEAR_BLACK})


def draw_run(ax, graph):
    """Panel (a): the run as ap.to_networkx returns it, one column per step, dependencies as arcs (1 unit = 1 inch)."""
    column = {step: i for i, step in enumerate(sorted({a["step"] for _, a in graph.nodes(data=True)}))}
    in_step = {}
    for key, attrs in graph.nodes(data=True):
        in_step.setdefault(attrs["step"], []).append(key)
    x = {key: column[step] + (j - (len(keys) - 1) / 2) * 0.34 for step, keys in in_step.items() for j, key in enumerate(keys)}
    arcs = [(u, v, a) for u, v, a in graph.edges(data=True) if a["kind"] == "dep"]
    value = {v: a["resource"][0].split(":", 1)[1] for _, v, a in arcs}          # the id that links the call to its source
    last = max(column.values())

    call = {"marker": "s", "color": NEAR_BLACK, "ms": 6.5}
    result = {"marker": "o", "mfc": "white", "mec": NEAR_BLACK, "mew": 1.2, "ms": 5.5}
    message = {"marker": "o", "color": DOT, "ms": 4}
    ax.plot([min(x.values()), max(x.values())], [0, 0], color=DOT, lw=1.0, zorder=1)
    for key, attrs in graph.nodes(data=True):
        style = {"call": call, "result": result}.get(attrs["kind"], message)
        ax.plot(x[key], 0, zorder=4 if attrs["kind"] != "message" else 3, **style)
        if attrs["kind"] == "call":
            label = attrs["name"] + (f"\n({value[key]})" if key in value else "")
            ax.text(x[key] + 0.12, -0.14, label, rotation=45, ha="right", va="top", rotation_mode="anchor", fontsize=9.5,
                    multialignment="right", linespacing=1.1)
    for u, v, attrs in arcs:
        back = graph.nodes[v]["step"] - graph.nodes[u]["step"]
        top = 0.26 * back
        mid = (x[u] + x[v]) / 2
        path = BezierPath([(x[u], 0), (mid, 2 * top), (x[v], 0)], [BezierPath.MOVETO, BezierPath.CURVE3, BezierPath.CURVE3])
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>,head_length=4,head_width=2.2", color=CORAL, lw=1.7,
                                     shrinkA=5, shrinkB=5, zorder=2))
        ax.text(mid, top, f"+{back}", ha="center", va="center", fontsize=9.5, color=CORAL, fontweight="bold", zorder=5,
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none"})
    # a one-line key for the marks, along the empty top left
    cursor = -0.3
    for label, style, gap in (("tool call", call, 3.2), ("tool result", result, 3.7), ("message", message, 3.0)):
        ax.plot(cursor, 1.58, **style)
        ax.text(cursor + 0.3, 1.58, label, va="center", fontsize=9.5)
        cursor += gap
    ax.text(cursor, 1.58, "+k: k steps back", va="center", fontsize=9.5, color=CORAL, fontweight="bold")
    ax.set_xlim(-0.5, last + 0.6)
    ax.set_ylim(-1.5, 1.72)
    ax.axis("off")


def draw_reach(ax, tau, tau2, swe):
    """Panel (b): share of a dataset's dependencies by steps back, and the share of runs with a far dependency."""
    xs = list(range(1, CAP + 1))
    top = 46                                                   # the bars live below this; the key to the runs sits above it
    ax.axvspan(FAR - 0.5, CAP + 0.5, ymin=0, ymax=top / 66, color=CORAL_FILL, zorder=0, lw=0)
    for color, offset, table in ((TEAL, -0.2, tau), (MINT, 0.2, tau2)):
        counts = Counter(min(s, CAP) for s in table["steps"])
        ax.bar([x + offset for x in xs], [100 * counts[x] / table["deps"] for x in xs], width=0.4, color=color, zorder=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_bounds(0, top)
    ax.tick_params(length=0)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs[:-1]] + [f"{CAP}+"])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_yticklabels(["0", "10", "20", "30", "40%"])
    ax.set_xlim(0.4, CAP + 0.6)
    ax.set_ylim(0, 66)
    ax.set_xlabel("steps back (a user turn counts as a step)")
    ax.set_ylabel("share of the dataset's dependencies", y=top / 132)
    ax.text(CAP + 0.45, top - 1, f"{FAR}+ steps back", color=CORAL, fontsize=9.5, fontweight="bold", va="top", ha="right")
    ax.text(1.65, 39.5, "tau2_bench", color=MINT_TEXT, fontsize=9.5, fontweight="bold", va="center")
    ax.text(2.3, 22.5, "tau_bench", color=TEAL_TEXT, fontsize=9.5, fontweight="bold", va="center")
    ax.text(CAP + 0.45, 37, f"swe_gym_openhands has no bars:\n{swe['deps']} dependencies in its first {swe['runs']:,} runs,"
            "\nwhose results are plain text", color=SUBTITLE, fontsize=9.5, va="top", ha="right", linespacing=1.2)
    # the share of runs with a dependency FAR or more back, counted in steps and in tool calls, one row per dataset
    for x, text in ((3.0, "in steps"), (4.9, "in calls")):
        ax.text(x, 65, text, fontsize=9.5, color=SUBTITLE, va="top", ha="right")
    ax.text(5.15, 65, f"runs with a dependency {FAR}+ back", fontsize=9.5, color=SUBTITLE, va="top")
    for row, (name, color, table) in enumerate((("tau_bench", TEAL, tau), ("tau2_bench", MINT, tau2),
                                                ("swe_gym_openhands", GOLD, swe))):
        y = 58.2 - 4.4 * row
        ax.plot(0.9, y, "s", color=color, ms=7, clip_on=False)
        for x, key in ((3.0, "far_runs"), (4.9, "far_call_runs")):
            ax.text(x, y, f"{100 * table[key] / table['runs']:.0f}%", fontsize=9.5, fontweight="bold", va="center", ha="right")
        ax.text(5.15, y, f"{name}, {table['runs']:,} runs", fontsize=9.5, va="center")


fig = plt.figure(figsize=(10, 4.2), dpi=200, facecolor="white")
fig.text(0.015, 0.955, "How far back does an agent reach for the values it uses?", fontsize=15, fontweight="bold",
         va="top", ha="left")
fig.text(0.015, 0.885, "Dependencies inferred by ap.derive_deps(run, rules=[\"value_flow\"]): a later call repeats an "
         "id-like value that an earlier tool result held.", fontsize=9.5, color=SUBTITLE,
         va="top", ha="left")
draw_run(fig.add_axes([0.015, 0.02, 0.55, 0.725]), graph)
draw_reach(fig.add_axes([0.655, 0.165, 0.335, 0.58]), stats["tau_bench"], stats["tau2_bench"], stats["swe_gym_openhands"])
fig.text(0.015, 0.80, "(a) One run as a graph", fontsize=11, fontweight="bold", va="bottom")
fig.text(0.015, 0.765, f"tau_bench, {run.id}: {graph.number_of_nodes()} events, {len(deps)} dependencies",
         fontsize=9.5, color=SUBTITLE, va="bottom")
fig.text(0.585, 0.80, "(b) How far back, per dataset", fontsize=11, fontweight="bold", va="bottom")
fig.text(0.585, 0.765, f"{stats['tau_bench']['deps']:,} tau_bench and {stats['tau2_bench']['deps']:,} tau2_bench dependencies",
         fontsize=9.5, color=SUBTITLE, va="bottom")
fig.savefig(out, dpi=200, facecolor="white")
print(f"wrote {out}")
