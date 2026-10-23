# Integration Recipes

Each directory here is one recipe: a short script that hands agentpile output to a downstream tool and checks the result, with the exact versions it was checked against. The README's [Works With](../../README.md#14-works-with) table lists them with the status of their last check.

| Recipe | Use | Tools | What the script checks |
|---|---|---|---|
| [evaluate](evaluate/) | score tool calls | Ragas, DeepEval | self-reference scores of 1.0; scores against tau-bench's reference calls; no judge-model call |
| [train](train/) | fine-tuning input | Hugging Face datasets, TRL | exact JSONL and Parquet round trips; assistant-only labels from TRL |
| [audit](audit/) | rule checks | Invariant Guardrails | data-flow and ordering rules on rows put back in source order |
| [analyze](analyze/) | tables | DuckDB, Polars | exact Parquet round trip; two queries that both engines answer alike |
| [view-harbor](view-harbor/) | browse runs | Harbor | trajectories load with Harbor's models and `harbor view` serves them |
| [view-inspect](view-inspect/) | browse runs | Inspect AI | the `.eval` log reads back and `inspect view` serves it |
| [view-otlp](view-otlp/) | trace viewers | Phoenix, Jaeger, MLflow, Langfuse | exports parse as OTLP and read back; with `--send`, every trace arrives |

## Running One

From the repository root, in a fresh environment with the Python version named in the first line of the recipe's `requirements.txt`:

```bash
pip install -e ".[hf]" -r examples/integrations/train/requirements.txt
bash examples/integrations/train/setup.sh          # only where a recipe has one: files fetched at install time
python examples/integrations/train/train.py
```

A recipe reads the small dataset fixtures in `tests/corpora/fixtures`, so its default run needs no network, account, API key or model. To run it the way CI does, put `examples/integrations/_offline` first on `PYTHONPATH`: every connection beyond this machine then raises, and loopback stays open for a viewer the recipe starts. The OTLP ingestion check needs the platforms themselves; `view-otlp/ingest.sh <platform>` starts one locally, sends the runs, queries them back and stops it.

## How They Are Checked

`.github/workflows/recipes.yml` runs every recipe at its pins, offline, weekly, on demand, for a release tag, and whenever a recipe or the code it uses changes. The OTLP ingestion check against Phoenix, Jaeger and MLflow runs weekly, for a release tag and on demand. Results on `main` go to the `recipe-status` branch. A failed check marks its row stale until the same check passes again, a passing row turns stale when the code it covers changes and its check has not run since, and a failed weekly run opens or updates an issue labeled `recipe-check`. A version of a tool that no check has run gets no row.

## Adding a Recipe

A new recipe is a directory with a script of the same name (`view-harbor/` holds `view_harbor.py`), a `requirements.txt` that pins every package the script imports, and a `setup.sh` if it needs files at install time. The script runs offline on the fixtures by default, asserts its checks and prints what it checked. Add the directory to the matrix in `recipes.yml` and a row to the README table.
