"""Turn chat rows into supervised fine-tuning input with Hugging Face datasets and TRL.

Takes runs and gives a datasets.Dataset of ap.to_chat rows that survive JSONL and Parquet, and an SFTTrainer that
builds assistant-only labels on Qwen2.5's chat template (TRL's training variant, bundled here). Nothing is trained and
no model is called: SFTTrainer wants a model object, so a randomly initialised Qwen2 of 2 million parameters stands in
for the one you would train. With your own runs the script writes the dataset to --out and counts the tokens TRL
trains; the checks compare every label with the rendered text.

    pip install -e ".[hf]" -r examples/integrations/train/requirements.txt
    bash examples/integrations/train/setup.sh                            # the Qwen2.5 tokenizer files, 11 MB
    python examples/integrations/train/train.py                          # the bundled fixtures, with the checks
    python examples/integrations/train/train.py tau_bench --limit 20     # your runs: a dataset name or a trace file
"""

import argparse
import json
import re
import tempfile
from itertools import islice
from pathlib import Path

import transformers
import trl
from datasets import Dataset, Features, List, Value, disable_progress_bars
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import get_training_chat_template, has_generation_markers

import agentpile as ap

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parents[2] / "tests" / "corpora" / "fixtures"
MODEL, REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
TEMPLATE = HERE / "qwen2_5_training.jinja"  # assistant-only loss needs a template with generation blocks
CALL = {"id": Value("string"), "type": Value("string"), "function": {"name": Value("string"), "arguments": Value("string")}}
FEATURES = Features({"messages": List({"role": Value("string"), "content": Value("string"), "name": Value("string"),
                                       "tool_call_id": Value("string"), "tool_calls": List(CALL)}),
                     "tools": Value("string")})


def to_sft(runs):
    """A Dataset with one chat row per run that has an assistant turn."""
    rows = [row for row in ap.to_chat(runs) if any(m["role"] == "assistant" for m in row["messages"])]
    # datasets infers features from the first row and would lose the tools column, so FEATURES declares them
    return Dataset.from_list([encode(row) for row in rows], features=FEATURES)


def make_trainer(model, dataset, tokenizer, output_dir):
    """An SFTTrainer that trains on the assistant turns of dataset only."""
    # TRL hands the template the stored messages, whose call arguments are JSON text, and the template needs objects
    apply_chat_template = tokenizer.apply_chat_template
    tokenizer.apply_chat_template = lambda messages, *a, **kw: apply_chat_template(messages_of(messages, objects=True), *a, **kw)
    args = SFTConfig(output_dir=str(output_dir), assistant_only_loss=True, chat_template_path=str(TEMPLATE),
                     max_length=None,  # no truncation, so every assistant turn is checked
                     per_device_train_batch_size=4, use_cpu=True, bf16=False, report_to="none",
                     save_strategy="no", disable_tqdm=True)
    return SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)


def encode(row):
    """Tool schemas differ in shape from row to row, so the tool list is stored as JSON text."""
    return {"messages": row["messages"], "tools": json.dumps(row["tools"], ensure_ascii=False) if "tools" in row else None}


def messages_of(stored, objects=False):
    """Stored messages as to_chat gave them (absent keys dropped); objects=True decodes the arguments of each call."""
    out = []
    for m in stored:
        m = {k: v for k, v in m.items() if v is not None or k == "content"}  # content is null only on a call-only turn
        if objects and "tool_calls" in m:
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": json.loads(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def tiny_model(tok):
    """SFTTrainer wants a model object: a random Qwen2 of 2 million parameters, which never runs, stands in for yours."""
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True))


# Four fixtures, plus swe_rebench_openhands (a second tool schema) and swe_agent_nebius (a row with no assistant turn).
SOURCES = ("tau_bench", "tau2_bench", "swe_gym_openhands", "who_and_when", "swe_rebench_openhands", "swe_agent_nebius")
TRL_TEMPLATES = Path(trl.__file__).parent / "chat_templates"
ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant(.*?<\|im_end\|>\n)", re.S)  # the text that a generation block covers


def decode(stored):
    """The to_chat row that encode() stored: absent keys dropped, tools back to a list, arguments still JSON text."""
    row = {"messages": messages_of(stored["messages"])}
    return row if stored["tools"] is None else {**row, "tools": json.loads(stored["tools"])}


def round_trip(rows, work, label):
    """Rows through JSONL and Parquet under the declared features; they must come back equal, as stored text and as rows."""
    ds = Dataset.from_list([encode(r) for r in rows], features=FEATURES)
    ds.to_json(str(work / f"{label}.jsonl"))
    ds.to_parquet(str(work / f"{label}.parquet"))
    cache = str(work / "cache")
    for back in (Dataset.from_json(str(work / f"{label}.jsonl"), features=FEATURES, cache_dir=cache),
                 Dataset.from_parquet(str(work / f"{label}.parquet"), features=FEATURES, cache_dir=cache)):
        assert back.to_list() == ds.to_list() and [decode(s) for s in back] == rows, label
    return len(rows)


def check_labels(tok, template, example):
    """Compare TRL's labels with the rendered text; returns (tokens, trained tokens, tool-call tokens)."""
    messages, tools = example["messages"], json.loads(example["tools"]) if example["tools"] else None
    text = tok.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == example["input_ids"], "TRL tokenized something other than the rendered text"
    turns = [m.span(1) for m in ASSISTANT_TURN.finditer(text)]
    assistant = [m for m in messages_of(messages, objects=True) if m["role"] == "assistant"]
    assert len(turns) == len(assistant) == text.count("<|im_start|>assistant"), "assistant turns not found in the text"
    labels = example["labels"]
    trained = [label != -100 for label in labels]
    assert any(trained), "no trained token"
    assert all(label in (-100, i) for label, i in zip(labels, example["input_ids"])), "a label is not its token"
    # exactly the tokens inside assistant turns are trained: every system, user and tool-result token is -100
    assert trained == [any(s <= a < e for s, e in turns) for a, _ in enc["offset_mapping"]], "labels do not follow the assistant turns"
    n_call = 0
    for m, (start, end) in zip(assistant, turns):  # and the tokens of each tool call are among them
        at = start
        for call in m.get("tool_calls", []):
            args = json.dumps(call["function"]["arguments"], ensure_ascii=False)
            body = '{"name": "%s", "arguments": %s}' % (call["function"]["name"], args)
            at = text.find(body, at, end)
            assert at >= 0, "tool call missing from the rendered turn"
            span = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < at + len(body) and b > at]
            assert span and all(trained[i] for i in span), "a tool-call token is masked"
            n_call, at = n_call + len(span), at + len(body)
    return len(labels), sum(trained), n_call


def check(fixtures, tok, work):
    """fixtures: the runs of each fixture dataset, by name."""
    runs = [run for group in fixtures.values() for run in group]
    report = {}
    rows = ap.to_chat(runs, report=report)  # arguments="json" is the default
    entries = [e for e in report["runs"] if e["row"] is not None]
    assert all(e["assistant_turns"] == sum(m["role"] == "assistant" for m in r["messages"]) for e, r in zip(entries, rows))
    named = ap.to_chat(fixtures["who_and_when"])  # no perspective: turns carry the agent name
    print(f"{len(rows)} chat rows from {len(fixtures)} fixtures ({len(runs) - len(rows)} runs without events give no row); "
          f"{sum('tools' in r for r in rows)} rows record tools")

    # (1) storage: explicit features, JSON text for tools and arguments, exact round trips
    varied = [r for r in rows if "tools" in r]
    schemas = {json.dumps(r["tools"], sort_keys=True) for r in varied}
    assert len(schemas) > 1 and all("name" in m for r in named for m in r["messages"] if m["role"] == "assistant")
    groups = (("no tools", [r for r in rows if "tools" not in r]), ("tool schemas", varied), ("all rows", rows))
    print("JSONL and Parquet round trips are exact:", {label: round_trip(group, work, label) for label, group in groups},
          f"(the tool batch holds {len(schemas)} different tool lists)")

    # (2) the rows without an assistant turn are dropped
    train = to_sft(runs)
    dropped = [e["id"] for e in entries if e["assistant_turns"] == 0]
    assert len(train) == len(rows) - len(dropped)
    print(f"dropped {len(dropped)} rows without an assistant turn: {dropped}")

    # (3) template: TRL's training variant of the tokenizer's own template, bundled and checked against the pinned TRL
    template = TEMPLATE.read_text()
    shipped = (TRL_TEMPLATES / "qwen2_5_training.jinja").read_text()
    assert template.endswith(shipped) and tok.chat_template == (TRL_TEMPLATES / "qwen2_5.jinja").read_text()
    assert get_training_chat_template(tok) == shipped and has_generation_markers(template)
    print(f"trl {trl.__version__} ships {len(list(TRL_TEMPLATES.glob('*_training.jinja')))} training templates; the bundled "
          f"qwen2_5_training.jinja is its copy, a patch of the template that {MODEL} ships")
    sample = next(r for r in train if r["tools"] and any(m["tool_calls"] for m in r["messages"]))
    text = tok.apply_chat_template(messages_of(sample["messages"]), tools=json.loads(sample["tools"]), chat_template=template, tokenize=False)
    assert '"arguments": "{' in text  # arguments left as JSON text would be encoded a second time by the template

    # (4) TRL: its own tokenization, label building and collation, with assistant-only loss
    trainer = make_trainer(tiny_model(tok), train, tok, work / "out")
    assert tok.chat_template == template
    totals = [check_labels(tok, template, example) for example in trainer.train_dataset]
    n_tokens, n_trained, n_call = map(sum, zip(*totals))
    batches = list(trainer.get_train_dataloader())
    assert sum(int(b["attention_mask"].sum()) for b in batches) == n_tokens
    assert sum(int((b["labels"] != -100).sum()) for b in batches) == n_trained
    assert all(((b["labels"] == -100) | (b["attention_mask"] == 1)).all() for b in batches)  # padding is never trained
    print(f"SFTTrainer prepared {len(totals)} rows ({n_tokens} tokens): {n_trained} trained, {n_call} of them in tool calls, "
          f"{n_tokens - n_trained} masked (-100); the collator gave {len(batches)} padded batches with the same counts")

    # (5) the template drops the name field of a multi-agent row
    render = lambda r: tok.apply_chat_template(r["messages"], chat_template=template, tokenize=False)
    nameless = [{**r, "messages": [{k: v for k, v in m.items() if k != "name"} for m in r["messages"]]} for r in named]
    assert all(render(a) == render(b) for a, b in zip(named, nameless))
    print(f"names are dropped: {len(named)} multi-agent rows render the same text with and without the name field")


def load_runs(sources, limit):
    """The first `limit` runs of each source; a dataset name takes limit= and downloads only that part, a file path does not."""
    return [run for source in sources
            for run in islice(ap.load(source, **({"limit": limit} if source in ap.datasets.names() else {})), limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="*", help="a dataset name or a trace file (default: the bundled fixtures, with the checks)")
    parser.add_argument("--limit", type=int, default=20, help="runs per source (default: 20)")
    parser.add_argument("--out", type=Path, help="directory for the files (default: a new temporary one)")
    args = parser.parse_args()
    disable_progress_bars()
    transformers.logging.set_verbosity_error()
    out = args.out or Path(tempfile.mkdtemp(prefix="agentpile-train-"))
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)  # setup.sh fetched it
    if args.source:
        dataset = to_sft(load_runs(args.source, args.limit))
        dataset.to_parquet(str(out / "sft.parquet"))
        trainer = make_trainer(tiny_model(tok), dataset, tok, out / "trainer")
        labels = [label for example in trainer.train_dataset for label in example["labels"]]
        print(f"{len(dataset)} rows written to {out / 'sft.parquet'}; SFTTrainer prepared {len(labels)} tokens, "
              f"{sum(label != -100 for label in labels)} of them in assistant turns")
    else:
        print(f"scratch files in {out}")
        check({name: list(ap.load(name, root=FIXTURES / name)) for name in SOURCES}, tok, out)
        print("all checks passed")


if __name__ == "__main__":
    main()
