"""How far back does an agent reach for the values it uses?

The datasets record which tool call produced which result, but not where a call's arguments came from.
``ap.derive_deps(run, rules=["value_flow"])`` infers it: a later call depends on the earlier call whose result
holds the same id-like value. A value is id-like when the whole string matches ``agentpile.deps.ID_LIKE``: up to
four letters, an optional ``-`` or ``_``, a digit and two or more further characters. That pattern matches
product and item numbers, airline reservation codes, telecom customer and line ids, and also dates, zip codes and
phone numbers. It does not match tau-bench order ids (``#W2378156``), user ids (``olivia_gonzalez_2305``) or file
paths, and it reads a result only when the result is JSON. When a value appears in several earlier results, the
rule picks the latest, so a dependency reaches the most recent holder of the value, not its first appearance.

Panel (a) draws one small tau_bench run as the networkx graph that ``ap.to_networkx(run, rules=["value_flow"])``
returns: one column per step, dependencies as arcs. Panel (b) shows, for tau_bench and tau2_bench (all runs) and the
first 1,000 runs of swe_gym_openhands, how many steps lie between the step that holds the producing result and the
step of the consuming call, and how many runs have a dependency reaching back three steps or more. Steps include the
user turns of a dialogue, so the table also counts the same distance in the agent's tool calls, which is shorter.
OpenHands results are plain text, so the rule finds no dependency there.

Run it from the repository root. It needs the extras hf, pandas and graph, and matplotlib:

    pip install "agentpile[hf,pandas,graph]" matplotlib
    python examples/value_flow.py

It downloads the tau_bench files (112 MB), the tau2_bench files (290 MB) and one swe_gym_openhands shard (70 MB),
about 470 MB, into ap.cache_root() once. It then runs in about 10 seconds, prints the numbers behind the figure,
and writes assets/value_flow.png.
"""

import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.path import Path as BezierPath  # noqa: E402

import agentpile as ap  # noqa: E402
from agentpile.deps import ID_LIKE  # noqa: E402

TEAL, CORAL, MINT, GOLD = "#4098AC", "#ED8D5A", "#7BC8A4", "#E8B04B"
NEAR_BLACK, SUBTITLE, SPINE, DOT, CORAL_FILL = "#1a1a1a", "#666666", "#9A9A9A", "#C4C4C4", "#FBEAE1"
TEAL_TEXT, MINT_TEXT = "#2F7F91", "#3A936C"          # the same hues, darker, for text on white
OUT = Path(__file__).resolve().parents[1] / "assets" / "value_flow.png"

SWE_LIMIT = 1000   # the first 1,000 runs of swe_gym_openhands (one shard of three is downloaded)
FAR = 3            # "reaches far" means a dependency at least this many steps back
CAP = 13           # the histogram groups every distance from CAP up
DATES_PHONES = re.compile(r"\d{4}-\d{2}-\d{2}|\d{3}-\d{3}-\d{4}")


# ---- the agentpile part: load, infer dependencies, measure ---------------------------------------------------

def measure(name, **options):
    """Per run: how far back each inferred dependency reaches, in steps and in tool calls; the run table; a run to draw."""
    reach, calls_back, rows, linked_by = {}, {}, [], {}
    smallest = None
    for run in ap.load(name, **options):                          # downloads the pinned files once
        deps = ap.derive_deps(run, rules=["value_flow"])           # inferred on request, never stored in the data
        reach[run.id] = [d["dst"]["step"] - d["src"]["step"] for d in deps]   # step ids count 1, 2, 3, ...
        agent_calls = [e for e in run.events() if e.kind == "call" and e.actor != "user"]   # not the user simulator's
        order = {(e.address["step"], e.address["call"]): i for i, e in enumerate(agent_calls)}
        calls_back[run.id] = [order[(d["dst"]["step"], d["dst"]["call"])] - order[(d["src"]["step"], d["src"]["call"])]
                              for d in deps]                       # the same distance in agent tool calls, no user turns
        linked_by[run.id] = [d["resource"] for d in deps]
        rows += ap.table(run, "runs")                              # one row per run: n_steps, n_calls, n_results
        if len(deps) >= 4 and max(reach[run.id]) >= FAR and (smallest is None or (len(run.steps), run.id) < smallest[0]):
            smallest = ((len(run.steps), run.id), run)             # the smallest run with four dependencies, one far
    return reach, calls_back, pd.DataFrame(rows)[["run_id", "n_steps", "n_calls", "n_results"]], linked_by, smallest


# ---- numbers --------------------------------------------------------------------------------------------------

def summarize(name, reach, calls_back, runs, linked_by):
    """Print the numbers behind panel (b) for one dataset and return them."""
    runs = runs.assign(deps=runs.run_id.map(lambda r: len(reach[r])),
                       far=runs.run_id.map(lambda r: max(reach[r], default=0) >= FAR),
                       far_calls=runs.run_id.map(lambda r: max(calls_back[r], default=0) >= FAR))
    steps = pd.Series([s for r in reach.values() for s in r], dtype="int64")
    calls = pd.Series([s for r in calls_back.values() for s in r], dtype="int64")
    n, with_deps = len(runs), int((runs.deps > 0).sum())
    none = runs[runs.deps == 0]
    no_call = int((none.n_calls == 0).sum())
    no_result = int(((none.n_calls > 0) & (none.n_results == 0)).sum())
    only_dates = sum(all(DATES_PHONES.fullmatch(v[len("value:"):]) for v in vs)
                     for per_run in linked_by.values() for vs in per_run)
    print(f"\n{name}: {n:,} runs, {len(steps):,} dependencies, {with_deps:,} runs with at least one")
    print(f"  runs with none: {n - with_deps:,} = {no_call:,} with no tool call + {no_result:,} with calls but no "
          f"paired result + {n - with_deps - no_call - no_result:,} whose results share no id-like value with a later call")
    if len(steps):
        print(f"  steps back: median {steps.median():.0f}, mean {steps.mean():.1f}, max {steps.max()}; "
              f"{100 * (steps >= FAR).mean():.1f}% of dependencies reach back {FAR} steps or more")
        print(f"  runs with a dependency {FAR}+ steps back: {int(runs.far.sum()):,} = {100 * runs.far.mean():.1f}% of "
              f"all runs, {100 * runs.far.sum() / with_deps:.1f}% of the runs with a dependency")
        print(f"  counted in tool calls instead of steps (user turns ignored): median {calls.median():.0f}; "
              f"{100 * (calls >= FAR).mean():.1f}% of dependencies and {100 * runs.far_calls.mean():.1f}% of runs "
              f"reach back {FAR} calls or more")
        print(f"  dependencies held by a date or a phone number alone: {only_dates:,} = {100 * only_dates / len(steps):.1f}%")
    else:
        print("  no dependency found")
    if name != "swe_gym_openhands":
        by_model = runs.assign(model=runs.run_id.str.split("/").str[0]).groupby("model").far.mean().mul(100).sort_values()
        print(f"  share of runs with a {FAR}+ step dependency, by model ({len(by_model)} models): "
              f"{by_model.index[0]} {by_model.iloc[0]:.0f}% ... {by_model.index[-1]} {by_model.iloc[-1]:.0f}%")
    return {"runs": n, "deps": len(steps), "with_deps": with_deps, "far_runs": int(runs.far.sum()),
            "far_call_runs": int(runs.far_calls.sum()), "steps": steps}


def why_none(name, **options):
    """Why a dataset has no dependency: its results that are JSON, and its string arguments that match ID_LIKE."""
    results = as_json = args = id_args = 0
    for run in ap.load(name, **options):
        for event in run.events():
            if event.kind == "result":
                results += 1
                try:
                    as_json += isinstance(json.loads(event.content), (dict, list))
                except (TypeError, ValueError):
                    pass
            elif event.kind == "call" and isinstance(event.content, dict):
                strings = [v for v in event.content.values() if isinstance(v, str)]
                args += len(strings)
                id_args += sum(bool(ID_LIKE.match(v)) for v in strings)
    print(f"  {name}: {as_json:,} of {results:,} results are JSON objects or lists; "
          f"{id_args:,} of {args:,} string arguments match ID_LIKE")


def pattern_coverage():
    """What ID_LIKE matches among tau_bench arguments: the share of each argument's string values it accepts."""
    seen, hit = Counter(), Counter()
    for run in ap.load("tau_bench"):
        for event in run.events():
            if event.kind == "call" and isinstance(event.content, dict):
                for key, value in event.content.items():
                    for item in value if isinstance(value, list) else [value]:
                        if isinstance(item, str):
                            seen[key] += 1
                            hit[key] += bool(ID_LIKE.match(item))
    print("\nID_LIKE on tau_bench call arguments (share of string values it accepts):")
    print("  " + ", ".join(f"{k} {100 * hit[k] / seen[k]:.0f}%" for k in
                           ("product_id", "item_ids", "reservation_id", "zip", "date", "order_id", "user_id", "payment_method_id")))


# ---- the figure -----------------------------------------------------------------------------------------------

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9.5,
                     "axes.edgecolor": SPINE, "axes.linewidth": 1.0, "xtick.color": NEAR_BLACK, "ytick.color": NEAR_BLACK,
                     "axes.labelcolor": SUBTITLE, "text.color": NEAR_BLACK})


def draw_run(ax, run):
    """Panel (a): the run as ap.to_networkx returns it, one column per step, dependencies as arcs (1 unit = 1 inch)."""
    graph = ap.to_networkx(run, rules=["value_flow"])
    column = {step: i for i, step in enumerate(sorted({a["step"] for _, a in graph.nodes(data=True)}))}
    in_step = {}
    for key, attrs in graph.nodes(data=True):
        in_step.setdefault(attrs["step"], []).append(key)
    x = {key: column[step] + (j - (len(keys) - 1) / 2) * 0.34 for step, keys in in_step.items() for j, key in enumerate(keys)}
    deps = [(u, v, a) for u, v, a in graph.edges(data=True) if a["kind"] == "dep"]
    value = {v: a["resource"][0].split(":", 1)[1] for _, v, a in deps}          # the id that links the call to its source
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
    for u, v, attrs in deps:
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
    return graph, len(deps)


def draw_reach(ax, tau, tau2, swe):
    """Panel (b): share of a dataset's dependencies by steps back, and the share of runs with a far dependency."""
    xs = list(range(1, CAP + 1))
    top = 46                                                   # the bars live below this; the key to the runs sits above it
    ax.axvspan(FAR - 0.5, CAP + 0.5, ymin=0, ymax=top / 66, color=CORAL_FILL, zorder=0, lw=0)
    for color, offset, table in ((TEAL, -0.2, tau), (MINT, 0.2, tau2)):
        counts = table["steps"].clip(upper=CAP).value_counts().reindex(xs, fill_value=0)
        ax.bar([x + offset for x in xs], 100 * counts / counts.sum(), width=0.4, color=color, zorder=3)
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


def figure(run, tau, tau2, swe):
    fig = plt.figure(figsize=(10, 4.2), dpi=200, facecolor="white")
    fig.text(0.015, 0.955, "How far back does an agent reach for the values it uses?", fontsize=15, fontweight="bold",
             va="top", ha="left")
    fig.text(0.015, 0.885, "Dependencies inferred by ap.derive_deps(run, rules=[\"value_flow\"]): a later call repeats an "
             "id-like value that an earlier tool result held.", fontsize=9.5, color=SUBTITLE,
             va="top", ha="left")
    ax_a = fig.add_axes([0.015, 0.02, 0.55, 0.725])
    ax_b = fig.add_axes([0.655, 0.165, 0.335, 0.58])
    graph, n_deps = draw_run(ax_a, run)
    draw_reach(ax_b, tau, tau2, swe)
    fig.text(0.015, 0.80, "(a) One run as a graph", fontsize=11, fontweight="bold", va="bottom")
    fig.text(0.015, 0.765, f"tau_bench, {run.id}: {graph.number_of_nodes()} events, {n_deps} dependencies",
             fontsize=9.5, color=SUBTITLE, va="bottom")
    fig.text(0.585, 0.80, "(b) How far back, per dataset", fontsize=11, fontweight="bold", va="bottom")
    fig.text(0.585, 0.765, f"{tau['deps']:,} tau_bench and {tau2['deps']:,} tau2_bench dependencies",
             fontsize=9.5, color=SUBTITLE, va="bottom")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    plt.close(fig)
    return graph


if __name__ == "__main__":
    start = time.time()
    results, smallest = {}, None
    for name, options in (("tau_bench", {}), ("tau2_bench", {}), ("swe_gym_openhands", {"limit": SWE_LIMIT})):
        reach, calls_back, runs, linked_by, picked = measure(name, **options)
        print(f"{name} at {ap.datasets.get(name).revision[:12]}")
        results[name] = summarize(name, reach, calls_back, runs, linked_by)
        if name == "tau_bench":
            smallest = picked[1]
    pattern_coverage()
    print("\nwhy the rule finds nothing in OpenHands:")
    why_none("swe_gym_openhands", limit=SWE_LIMIT)
    graph = figure(smallest, results["tau_bench"], results["tau2_bench"], results["swe_gym_openhands"])
    print(f"\nPanel (a): {smallest.id}, the run with the fewest steps among those with 4 or more dependencies "
          f"and one reaching back {FAR}+ steps ({graph.number_of_nodes()} events, {len(smallest.steps)} steps)")
    for u, v, attrs in graph.edges(data=True):
        if attrs["kind"] == "dep":
            back = graph.nodes[v]["step"] - graph.nodes[u]["step"]
            print(f"  {graph.nodes[u]['name']} (step {graph.nodes[u]['step']}) -> {graph.nodes[v]['name']} "
                  f"(step {graph.nodes[v]['step']}): {', '.join(attrs['resource'])}, {back} step{'s' * (back != 1)} back")
    print(f"\nwrote {OUT} in {time.time() - start:.0f} s")
