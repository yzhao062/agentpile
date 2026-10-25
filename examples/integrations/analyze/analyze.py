"""Query runs as tables with DuckDB and Polars.

Takes runs and writes the three tables that ap.table builds (runs, events and labels: flat rows of scalars, the same
columns whatever the dataset) to Parquet in --out, a new temporary directory by default. DuckDB SQL and Polars then ask
the same two questions of the files:

  1. How many labels are there per dataset, label name and target kind? (labels joined to runs on run_id)
  2. How many messages, calls and results does each run have, and what share of its calls has a result?
     (a result row names its call in source_call_id, which equals the call's `call` in the same run)

    pip install -e ".[hf]" -r examples/integrations/analyze/requirements.txt
    python examples/integrations/analyze/analyze.py                        # the bundled fixtures, with the checks
    python examples/integrations/analyze/analyze.py tau_bench --limit 20   # your runs: a dataset name or a trace file
"""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import agentpile as ap

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
LABELS_SQL = """
    SELECT r.dataset, l.name, l.target_kind, count(*) AS n
    FROM labels l JOIN runs r USING (run_id) GROUP BY 1, 2, 3"""
EVENTS_SQL = """
    WITH answered AS (SELECT DISTINCT run_id, source_call_id AS call FROM events WHERE kind = 'result'),
    counts AS (
        SELECT r.run_id,
               count(*) FILTER (WHERE e.kind = 'message') AS messages,
               count(*) FILTER (WHERE e.kind = 'call') AS calls,
               count(*) FILTER (WHERE e.kind = 'result') AS results,
               count(*) FILTER (WHERE e.kind = 'call' AND a.call IS NOT NULL) AS answered
        FROM runs r LEFT JOIN events e USING (run_id)
        LEFT JOIN answered a ON a.run_id = e.run_id AND a.call = e.call
        GROUP BY r.run_id)
    SELECT *, answered::DOUBLE / NULLIF(calls, 0) AS share FROM counts"""


def to_parquet(runs, out):
    """Write the runs, events and labels tables of runs to Parquet files in out; returns their paths by level.
    A level with no rows gets no file: runs read from a trace file usually carry no labels."""
    files = {}
    for level in ("runs", "events", "labels"):
        rows = ap.table(runs, level)
        if rows:
            files[level] = out / f"{level}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), files[level])
    return files


def with_duckdb(files):
    con = duckdb.connect()
    for level, path in files.items():
        con.read_parquet(str(path)).create_view(level)
    labels = con.sql(LABELS_SQL).fetchall() if "labels" in files else []
    return labels, con.sql(EVENTS_SQL).fetchall()


def with_polars(files):
    runs, events = (pl.read_parquet(files[level]) for level in ("runs", "events"))
    counts = []
    if "labels" in files:
        counts = (pl.read_parquet(files["labels"]).join(runs.select("run_id", "dataset"), on="run_id")
                  .group_by("dataset", "name", "target_kind").agg(n=pl.len()).rows())
    answered = (events.filter(pl.col("kind") == "result").select("run_id", call="source_call_id")
                .unique().with_columns(hit=pl.lit(True)))     # null keys never join, as in SQL
    kind = pl.col("kind")
    per_run = (events.join(answered, on=["run_id", "call"], how="left")
               .group_by("run_id").agg(messages=(kind == "message").sum(), calls=(kind == "call").sum(),
                                       results=(kind == "result").sum(),
                                       answered=((kind == "call") & pl.col("hit").fill_null(False)).sum()))
    per_run = (runs.select("run_id").join(per_run, on="run_id", how="left")   # a run with no events counts 0
               .with_columns(pl.col("messages", "calls", "results", "answered").fill_null(0))
               .with_columns(share=pl.when(pl.col("calls") > 0).then(pl.col("answered") / pl.col("calls"))))
    return counts, per_run.rows()


def summarize(runs, duck_labels, duck_events):
    name = lambda dataset: (dataset or "no dataset").split("/")[-1]   # runs read from a file have none
    short = {r["run_id"]: name(r["dataset"]) for r in ap.table(runs, "runs")}
    kinds, calls = defaultdict(Counter), defaultdict(lambda: [0, 0])
    for dataset, _, kind, n in duck_labels:
        kinds[name(dataset)][kind] += n
    for run_id, _, n_calls, _, n_answered, _ in duck_events:
        calls[short[run_id]][0] += n_answered
        calls[short[run_id]][1] += n_calls
    print(f"\n{'dataset':<26}{'labels by target kind':<30}calls with a result")
    for dataset in sorted(set(short.values())):
        label_text = ", ".join(f"{kind} {n}" for kind, n in sorted(kinds[dataset].items())) or "none"
        print(f"{dataset:<26}{label_text:<30}{calls[dataset][0]} of {calls[dataset][1]}")


DATASETS = ["who_and_when", "trail", "mast", "tau_bench", "agentdojo"]


def check(runs, files, duck, polars):
    rows = {level: ap.table(runs, level) for level in files}
    assert len({r["run_id"] for r in rows["runs"]}) == len(runs), "run ids must be unique for the join on run_id"

    # (1) Parquet round trip with pyarrow: same rows, same types
    for level, level_rows in rows.items():
        table = pa.Table.from_pylist(level_rows)
        back = pq.read_table(files[level])
        assert back.schema == table.schema and back.equals(table), f"{level}: Parquet round trip changed the table"
        assert back.to_pylist() == level_rows, f"{level}: rows differ after the round trip"
        print(f"round trip ok: {level:<6} {back.num_rows:>3} rows x {back.num_columns:>2} columns")
    labels = pq.read_table(files["labels"])
    assert all(labels.schema.field(c).type == pa.string() for c in ("value", "value_type"))
    typed = [json.loads(v) if t not in (None, "str") else v
             for v, t in zip(labels["value"].to_pylist(), labels["value_type"].to_pylist())]
    original = [label.get("value") for run in runs for label in run.labels]
    assert [(type(x), x) for x in typed] == [(type(x), x) for x in original], "value + value_type lose a value"
    print(f"label values are text with value_type ({dict(Counter(labels['value_type'].to_pylist()))}); "
          "decoding them gives back every typed value")

    # (2) the same two queries in DuckDB and in Polars
    (duck_labels, duck_events), (pl_labels, pl_events) = duck, polars
    assert sorted(duck_labels) == sorted(pl_labels), "label counts differ between DuckDB and Polars"
    assert sorted(duck_events) == sorted(pl_events), "per-run event counts differ between DuckDB and Polars"
    print(f"DuckDB == Polars: {len(duck_labels)} (dataset, name, target kind) groups, {len(duck_events)} runs")

    # both engines against the tables themselves: no label lost in the join, counts as the runs table states them
    assert sum(n for *_, n in duck_labels) == len(rows["labels"])
    stated = {r["run_id"]: (r["n_messages"], r["n_calls"], r["n_results"]) for r in rows["runs"]}
    assert {run_id: (m, c, r) for run_id, m, c, r, *_ in duck_events} == stated, "events disagree with the runs table"
    print("checked against ap.table: label total and per-run message, call and result counts agree")


def load_runs(sources, limit):
    """The first `limit` runs of each source; a dataset name takes limit= and downloads only that part, a file path does not."""
    return [run for source in sources
            for run in islice(ap.load(source, **({"limit": limit} if source in ap.datasets.names() else {})), limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="*", help="a dataset name or a trace file (default: the bundled fixtures, with the checks)")
    parser.add_argument("--limit", type=int, default=20, help="runs per source (default: 20)")
    parser.add_argument("--out", type=Path, help="directory for the Parquet files (default: a new temporary one)")
    args = parser.parse_args()
    out = args.out or Path(tempfile.mkdtemp(prefix="agentpile-analyze-"))
    out.mkdir(parents=True, exist_ok=True)
    if args.source:
        runs = load_runs(args.source, args.limit)
    else:
        runs = [run for name in DATASETS for run in ap.load(name, root=FIXTURES / name)]

    files = to_parquet(runs, out)
    duck, polars = with_duckdb(files), with_polars(files)
    print(f"{len(runs)} runs; Parquet files in {out}")
    summarize(runs, *duck)
    if not args.source:
        print()
        check(runs, files, duck, polars)
        print("all checks passed")


if __name__ == "__main__":
    main()
