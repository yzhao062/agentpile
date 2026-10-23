"""One table analysis, run by DuckDB and by Polars, over the tables agentpile builds from five datasets.

ap.table gives flat rows of scalars at three levels (runs, events, labels), the same columns whatever the dataset.
This example writes the three tables to Parquet with pyarrow, reads them back and checks that nothing changed
(same rows, same types, label values still text next to value_type). It then asks two questions of the files in
DuckDB SQL and in Polars and checks that the two engines agree to the last digit:

  1. How many labels are there per dataset, label name and target kind? (labels joined to runs on run_id)
  2. How many messages, calls and results does each run have, and what share of its calls has a result?
     (a result row names its call in source_call_id, which equals the call's `call` in the same run)

The default run reads the small fixtures in tests/corpora/fixtures of the checkout: no network, no API key, no model.

    pip install -e ".[hf]" -r examples/integrations/analyze/requirements.txt   # from the checkout root
    python examples/integrations/analyze/analyze.py [--out DIR]                # --out keeps the Parquet files there
"""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import agentpile as ap

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
DATASETS = ["who_and_when", "trail", "mast", "tau_bench", "agentdojo"]
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


def with_duckdb(files):
    con = duckdb.connect()
    for level, path in files.items():
        con.read_parquet(str(path)).create_view(level)
    return con.sql(LABELS_SQL).fetchall(), con.sql(EVENTS_SQL).fetchall()


def with_polars(files):
    runs, events, labels = (pl.read_parquet(files[level]) for level in ("runs", "events", "labels"))
    counts = (labels.join(runs.select("run_id", "dataset"), on="run_id")
              .group_by("dataset", "name", "target_kind").agg(n=pl.len()))
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
    return counts.rows(), per_run.rows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, help="directory for the Parquet files (default: a new temporary one)")
    out = parser.parse_args().out or Path(tempfile.mkdtemp(prefix="agentpile-analyze-"))
    out.mkdir(parents=True, exist_ok=True)

    runs = [run for name in DATASETS for run in ap.load(name, root=str(FIXTURES / name))]
    rows = {level: ap.table(runs, level) for level in ("runs", "events", "labels")}
    assert len({r["run_id"] for r in rows["runs"]}) == len(runs), "run ids must be unique for the join on run_id"
    print(f"loaded {len(runs)} runs from {len(DATASETS)} datasets; Parquet files in {out}")

    # (1) Parquet round trip with pyarrow: same rows, same types
    files = {}
    for level, level_rows in rows.items():
        table = pa.Table.from_pylist(level_rows)
        files[level] = out / f"{level}.parquet"
        pq.write_table(table, files[level])
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
    (duck_labels, duck_events), (pl_labels, pl_events) = with_duckdb(files), with_polars(files)
    assert sorted(duck_labels) == sorted(pl_labels), "label counts differ between DuckDB and Polars"
    assert sorted(duck_events) == sorted(pl_events), "per-run event counts differ between DuckDB and Polars"
    print(f"\nDuckDB == Polars: {len(duck_labels)} (dataset, name, target kind) groups, {len(duck_events)} runs")

    # both engines against the tables themselves: no label lost in the join, counts as the runs table states them
    assert sum(n for *_, n in duck_labels) == len(rows["labels"])
    stated = {r["run_id"]: (r["n_messages"], r["n_calls"], r["n_results"]) for r in rows["runs"]}
    assert {run_id: (m, c, r) for run_id, m, c, r, *_ in duck_events} == stated, "events disagree with the runs table"
    print("checked against ap.table: label total and per-run message, call and result counts agree")

    short = {r["run_id"]: r["dataset"].split("/")[-1] for r in rows["runs"]}
    kinds, calls = defaultdict(Counter), defaultdict(lambda: [0, 0])
    for dataset, _, kind, n in duck_labels:
        kinds[dataset.split("/")[-1]][kind] += n
    for run_id, _, n_calls, _, n_answered, _ in duck_events:
        calls[short[run_id]][0] += n_answered
        calls[short[run_id]][1] += n_calls
    print(f"\n{'dataset':<26}{'labels by target kind':<30}calls with a result")
    for dataset, count in sorted(kinds.items()):
        label_text = ", ".join(f"{kind} {n}" for kind, n in sorted(count.items()))
        print(f"{dataset:<26}{label_text:<30}{calls[dataset][0]} of {calls[dataset][1]}")


if __name__ == "__main__":
    main()
