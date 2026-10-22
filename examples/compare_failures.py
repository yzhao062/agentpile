"""Where do annotated failures sit in an agent run? Three datasets, three annotation shapes, one table call.

Who&When labels the decisive error step of each failed run (one step per run), TRAIL labels every error on a
span of the trace (several per run, each with a category and an impact), and MAST labels each run with 14 failure
modes and no location (judged by an LLM). Without agentpile that is three parsers and three schemas. With it,
the three datasets are three ap.load calls and one ap.to_pandas(runs, "labels") call, and every label has the
same columns: run_id, name, value, target_kind (run, step or span) and the address it points at.

Panel (a) puts the step that each Who&When and TRAIL label points at on one axis, its position in the run (step
number divided by the number of steps, so 1.0 is the last step), as the share of labels at or before each
position. A TRAIL error sits on a span, and run.spans says which step that span became; labels on spans that
became no step are counted and left out. Panel (b) shows how often the judge flagged each MAST failure mode, as
a share of the runs it labelled.

    pip install "agentpile[hf,pandas,graph]" matplotlib   # graph is for one cross-check; TRAIL is gated on Hugging Face
    python examples/compare_failures.py                   # downloads about 260 MB once; writes assets/compare_failures.png
"""

import time
from pathlib import Path

import matplotlib
import numpy as np

import agentpile as ap

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]

TEAL, CORAL, GOLD = "#4098AC", "#ED8D5A", "#E8B04B"
NEAR_BLACK, SUBTITLE, SPINE, GRID = "#1a1a1a", "#666666", "#9A9A9A", "#E6E6E6"
OUT = Path(__file__).resolve().parents[1] / "assets" / "compare_failures.png"


def load():
    """The agentpile part: three datasets in, one runs table and one labels table out."""
    who, trail, mast = (list(ap.load(name)) for name in ("who_and_when", "trail", "mast"))
    runs = ap.to_pandas(who + trail + mast, "runs")        # one row per run, whatever its dataset
    labels = ap.to_pandas(who + trail + mast, "labels")    # one row per label; target_kind says where it points

    def position(frame):   # where a labelled step sits in its run: step number / number of steps
        frame = frame.merge(runs[["run_id", "config", "n_steps"]], on="run_id")
        return frame.assign(pos=frame.step.astype(float) / frame.n_steps.astype(float))

    # Who&When: the label mistake_step targets a step.
    ww = position(labels.query("name == 'mistake_step'"))

    # TRAIL: the label error targets a span, and run.spans gives the step each span became (none for some).
    step_of = {(r.id, sid): s["address"]["step"] for r in trail for sid, s in r.spans.items() if s.get("address")}
    tr = labels.query("label_source == 'trail' and name == 'error'")
    tr = tr.assign(step=[step_of.get(key) for key in zip(tr.run_id, tr.span)])      # NaN where the span became no step
    mapped = position(tr.dropna(subset=["step"]))

    # MAST: 14 labels per run, one per failure mode; value 1 (flagged), 0, or null (no annotation).
    modes = labels.query("label_source == 'llm_judge'").assign(value=lambda d: d.value.astype(float))
    share = modes.groupby("name").value.mean().sort_values(ascending=False)   # share of labelled runs
    names = {label["name"]: label["source_ref"]["mode"] for label in mast[0].labels}   # the taxonomy names
    return who, trail, mast, labels, ww, tr, mapped, modes, share, names


def agent_position(run, step):
    """Position of a step among the run's agent steps only, leaving out system and user steps."""
    ids = [s["step_id"] for s in run.steps if s["source"] == "agent"]
    return (ids.index(step) + 1) / len(ids)


def report(who, trail, mast, labels, ww, tr, mapped, modes, share, names):
    """Print every number the figure shows, plus a check of the position measure."""
    first = mapped.groupby("run_id").pos.min()
    print(f"runs loaded: who_and_when {len(who)}, trail {len(trail)}, mast {len(mast)}")
    print("labels by dataset and the kind of target they point at:")
    print(labels.groupby(["label_source", "target_kind"]).size().to_string())
    print("\nWho&When (mistake_step, one per run)")
    print(f"  labels {len(ww)} in {ww.run_id.nunique()} runs; steps outside the run: {int((ww.step > ww.n_steps).sum())}")
    print(f"  position: median {ww.pos.median():.2f}, mean {ww.pos.mean():.2f}, share at or before 0.5: {(ww.pos <= 0.5).mean():.1%}")
    for config, group in ww.groupby("config"):
        print(f"  {config}: {len(group)} runs, median position {group.pos.median():.2f}")

    what = {(r.id, sid): f"{s['kind']} {s['name']}" for r in trail for sid, s in r.spans.items()}
    unmapped = tr[tr.step.isna()]
    kinds = unmapped.apply(lambda row: what.get((row.run_id, row.span), "no span"), axis=1).value_counts().to_dict()
    print("\nTRAIL (error labels on spans, several per run)")
    print(f"  error labels {len(tr)} in {tr.run_id.nunique()} of {len(trail)} runs; mapped to a step {len(mapped)}; "
          f"mapped to no step {len(unmapped)} (by span: {kinds})")
    print(f"  mapped errors per run with one: {len(mapped) / mapped.run_id.nunique():.1f}")
    print(f"  position of every error: median {mapped.pos.median():.2f}, mean {mapped.pos.mean():.2f}, "
          f"share at or before 0.5: {(mapped.pos <= 0.5).mean():.1%}")
    early = int((mapped.pos <= 0.5).sum())
    print(f"  share at or before 0.5 if the {len(unmapped)} left-out labels all fell after / all before: "
          f"{early / len(tr):.1%} to {(early + len(unmapped)) / len(tr):.1%}")
    print(f"  position of the first error of each run ({len(first)} runs): median {first.median():.2f}, "
          f"share at or before 0.5: {(first <= 0.5).mean():.1%}")

    print(f"  {int(tr.span.notna().sum())} of the labels sit on {tr.span.nunique()} different spans")

    print("\nChecks on the position measure")
    by_id = {r.id: r for r in list(who) + list(trail)}
    for label, frame in (("Who&When", ww), ("TRAIL", mapped)):
        pos = [agent_position(by_id[run], int(step)) for run, step in zip(frame.run_id, frame.step)]
        print(f"  position among agent steps only, {label}: median {np.median(pos):.2f} (all steps: {frame.pos.median():.2f})")
    try:  # the graph extra: ap.to_networkx puts each label on a node, which is another route from span to step
        per_run = mapped.groupby("run_id").size()
        on_nodes = {r.id: sum(lab["name"] == "error" for _, node in ap.to_networkx(r).nodes(data=True) for lab in node["labels"])
                    for r in trail}
        same = all(on_nodes[r.id] == per_run.get(r.id, 0) for r in trail)
        print(f"  ap.to_networkx puts {sum(on_nodes.values())} error labels on a node; same count as above in every run: {same}")
    except ImportError:
        print("  ap.to_networkx check skipped (pip install 'agentpile[graph]')")

    judged = modes.groupby("run_id").value.agg(["sum", "count"]).query("count > 0")
    print("\nMAST (14 failure modes, LLM judge)")
    print(f"  runs with judge labels {len(judged)} of {len(mast)}; modes flagged per run: mean {judged['sum'].mean():.1f}; "
          f"runs with none flagged {int((judged['sum'] == 0).sum())}")
    for code, value in share.items():
        print(f"  {code} {names[code]:<36s}{value:6.1%}")
    return judged


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(length=0, labelsize=9, colors=SUBTITLE)
    ax.set_facecolor("white")


def ecdf(pos):
    x = np.sort(np.asarray(pos, dtype=float))
    return np.concatenate([[0.0], x]), np.arange(0, len(x) + 1) / len(x)


def draw(ww, tr, mapped, share, names, judged, n_mast):
    first = mapped.groupby("run_id").pos.min()
    W, H = 10.0, 4.2
    fig = plt.figure(figsize=(W, H), dpi=200, facecolor="white")

    def put(x, y, text, **kw):  # figure text at a position in inches from the bottom left
        return fig.text(x / W, y / H, text, **kw)

    put(0.3, H - 0.2, "Where annotated failures sit in agent runs", fontsize=15, fontweight="bold", color=NEAR_BLACK, va="top")
    put(0.3, H - 0.52, "Who&When labels one step per run, TRAIL labels every error on a span, MAST labels the whole run. "
        'One table call reads all three: ap.to_pandas(runs, "labels").', fontsize=9.5, color=SUBTITLE, va="top")

    # (a) share of labels whose step sits at or before each position in the run (a TRAIL step with two errors counts twice)
    ax = fig.add_axes([0.8 / W, 0.62 / H, 4.0 / W, 2.16 / H])
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], color=SPINE, linewidth=1, linestyle=(0, (2, 3)))
    ax.axvline(0.5, color=SPINE, linewidth=0.8, linestyle=(0, (1, 3)))
    at_mid = {}
    for key, pos, color, line, width in (("ww", ww.pos, TEAL, "-", 2.4), ("first", first, CORAL, "--", 1.9),
                                         ("all", mapped.pos, CORAL, "-", 2.4)):
        x, y = ecdf(pos)
        ax.step(x, y, where="post", color=color, linestyle=line, linewidth=width)
        at_mid[key] = float((np.asarray(pos, dtype=float) <= 0.5).mean())
        ax.plot([0.5], [at_mid[key]], "o", color=color, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    halo = [pe.withStroke(linewidth=3, foreground="white")]
    n_err_runs = mapped.run_id.nunique()
    for x, y, text, color, ha, va in (
        (0.03, 0.97, f"Who&When: the decisive step\n{at_mid['ww']:.0%} by mid-run", TEAL, "left", "top"),
        (0.315, 0.6, f"TRAIL: first error\nof each run\n{at_mid['first']:.0%} by mid-run", CORAL, "right", "center"),
        (0.97, 0.33, f"TRAIL: every error,\n{len(mapped) / n_err_runs:.1f} per run\n{at_mid['all']:.0%} by mid-run", CORAL, "right", "top"),
    ):
        ax.text(x, y, text, color=color, fontsize=9.5, fontweight="bold", va=va, ha=ha, linespacing=1.25, path_effects=halo, zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Position in the run (step number / number of steps; 1 is the last step)", fontsize=9, color=SUBTITLE, labelpad=5)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_ylabel("Share of labels\nat or before this position", fontsize=9, color=SUBTITLE, labelpad=4)
    put(0.8, 3.33, "(a) Position of the annotated step in the run", fontsize=10.5, fontweight="bold", color=NEAR_BLACK, va="bottom")
    put(0.8, 2.97, f"Who&When: {len(ww)} steps, {ww.run_id.nunique()} runs. TRAIL: {len(mapped)} error labels, {n_err_runs} runs.\n"
        f"Left out: {len(tr) - len(mapped)} of {len(tr)} TRAIL labels that map to no step.",
        fontsize=9, color=SUBTITLE, va="bottom", linespacing=1.3)

    # (b) share of MAST runs per failure mode, as the judge labelled them
    ax2 = fig.add_axes([7.25 / W, 0.3 / H, 2.3 / W, 2.48 / H])
    style(ax2)
    ordered = share.sort_values()
    y = np.arange(len(ordered))
    ax2.barh(y, ordered.values, color=GOLD, height=0.68)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{code} {names[code]}" for code in ordered.index], fontsize=9, color=NEAR_BLACK)
    for yi, value in zip(y, ordered.values):
        ax2.text(value + 0.008, yi, f"{value:.0%}", va="center", fontsize=9, color=NEAR_BLACK)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(-0.6, len(ordered) - 0.4)
    ax2.set_xticks([])
    ax2.spines["bottom"].set_visible(False)
    put(5.0, 3.33, "(b) MAST failure modes, share of runs flagged", fontsize=10.5, fontweight="bold", color=NEAR_BLACK, va="bottom")
    put(5.0, 2.97, f"{len(judged):,} of {n_mast:,} runs have judge labels (run level, no position).\n"
        f"The LLM judge flags {judged['sum'].mean():.1f} modes per run on average.", fontsize=9, color=SUBTITLE, va="bottom", linespacing=1.3)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    start = time.perf_counter()
    who, trail, mast, labels, ww, tr, mapped, modes, share, names = load()
    judged = report(who, trail, mast, labels, ww, tr, mapped, modes, share, names)
    draw(ww, tr, mapped, share, names, judged, len(mast))
    print(f"elapsed {time.perf_counter() - start:.0f} s")
