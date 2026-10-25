"""Where do annotated failures sit in an agent run?

Who&When labels the decisive error step of each failed run, TRAIL labels every error on a span of the trace, and MAST
labels each run with 14 failure modes judged by an LLM, with no location. Without agentpile that is three parsers and
three schemas. With it, three ap.load calls and one ap.to_pandas(runs, "labels") call give every label the same
columns. Panel (a) then puts the step that each Who&When and TRAIL label points at on one axis, as its position in the
run (step number divided by the number of steps, so 1.0 is the last step), and panel (b) shows how often the judge
flagged each MAST failure mode.

    pip install "agentpile[hf,pandas]" matplotlib   # TRAIL is gated on Hugging Face: accept its terms and log in first
    python examples/compare_failures.py [OUT]

Downloads about 260 MB once, runs in about 2 s and writes compare_failures.png.
"""

import sys

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import agentpile as ap

runs = [run for name in ("who_and_when", "trail", "mast") for run in ap.load(name)]
labels = ap.to_pandas(runs, "labels")   # run_id, name, value, target_kind, label_source, step, span, ...  (step is empty for span labels)
print(labels.groupby(["label_source", "target_kind"]).size())   # three annotation shapes in one table

# The labels table has no step for a TRAIL span and no mode name for a MAST code; both come from the runs.
step_of = {(run.id, sid): s["address"]["step"] for run in runs for sid, s in run.spans.items() if s.get("address")}
names = {label["name"]: label["source_ref"]["mode"] for run in runs for label in run.labels if "mode" in label["source_ref"]}

info = ap.to_pandas(runs, "runs")   # one row per run: n_steps, dataset, ...


def position(frame):   # where a labelled step sits in its run: step number / number of steps
    frame = frame.merge(info[["run_id", "n_steps"]], on="run_id")
    return frame.assign(pos=frame.step.astype(float) / frame.n_steps.astype(float))


# Who&When: the label mistake_step targets a step.
ww = position(labels.query("name == 'mistake_step'"))

# TRAIL: the label error targets a span, and run.spans gives the step each span became (none for some).
tr = labels.query("label_source == 'trail' and name == 'error'")
tr = tr.assign(step=[step_of.get(key) for key in zip(tr.run_id, tr.span)])      # NaN where the span became no step
mapped = position(tr.dropna(subset=["step"]))
first = mapped.groupby("run_id").pos.min()

# MAST: 14 labels per run, one per failure mode; value 1 (flagged), 0, or null (no annotation).
modes = labels.query("label_source == 'llm_judge'").assign(value=lambda d: d.value.astype(float))
share = modes.groupby("name").value.mean().sort_values()   # share of labelled runs
judged = modes.groupby("run_id").value.agg(["sum", "count"]).query("count > 0")
n_mast = int((info.dataset == "mcemri/MAST-Data").sum())

# ---- figure ----

TEAL, CORAL, GOLD = "#4098AC", "#ED8D5A", "#E8B04B"
NEAR_BLACK, SUBTITLE, SPINE, GRID = "#1a1a1a", "#666666", "#9A9A9A", "#E6E6E6"
out = sys.argv[1] if len(sys.argv) > 1 else "compare_failures.png"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]


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
y = np.arange(len(share))
ax2.barh(y, share.values, color=GOLD, height=0.68)
ax2.set_yticks(y)
ax2.set_yticklabels([f"{code} {names[code]}" for code in share.index], fontsize=9, color=NEAR_BLACK)
for yi, value in zip(y, share.values):
    ax2.text(value + 0.008, yi, f"{value:.0%}", va="center", fontsize=9, color=NEAR_BLACK)
ax2.set_xlim(0, 0.5)
ax2.set_ylim(-0.6, len(share) - 0.4)
ax2.set_xticks([])
ax2.spines["bottom"].set_visible(False)
put(5.0, 3.33, "(b) MAST failure modes, share of runs flagged", fontsize=10.5, fontweight="bold", color=NEAR_BLACK, va="bottom")
put(5.0, 2.97, f"{len(judged):,} of {n_mast:,} runs have judge labels (run level, no position).\n"
    f"The LLM judge flags {judged['sum'].mean():.1f} modes per run on average.", fontsize=9, color=SUBTITLE, va="bottom", linespacing=1.3)

fig.savefig(out, dpi=200, facecolor="white")
print(f"wrote {out}")
