"""Chat rows into supervised fine-tuning with Hugging Face datasets and TRL, without a training run.

ap.to_chat gives OpenAI-style rows. This recipe follows them to the point where a trainer would start, and checks
the places where they usually break. Storage: tool schemas and tool-call arguments differ in shape from row to row,
and type inference would merge the arguments of different calls into one struct with null keys; both are kept as
JSON text under explicit datasets.Features instead, and the rows survive JSONL and Parquet exactly. Template: chat
templates take argument objects, not text, and assistant-only loss needs a template with generation blocks (TRL's
Qwen2.5 training variant, bundled here). Masks: TRL's SFTTrainer tokenizes the rows and builds the labels, and every
label is compared with the rendered text. Nothing is trained and no model is called; SFTTrainer wants a model object,
so a randomly initialised Qwen2 of 2 million parameters, built from a config in this script, stands in and never runs.

The default run reads the fixtures of this checkout and makes no network call. From the repository root:

    pip install -e ".[hf]" -r examples/integrations/train/requirements.txt
    hf download Qwen/Qwen2.5-0.5B-Instruct config.json tokenizer.json tokenizer_config.json vocab.json merges.txt \\
        --revision 7ae557604adf67be50417f59c2c2f167def9a775
    python examples/integrations/train/train.py

The hf line puts the tokenizer files (11 MB, no weights) in the Hugging Face cache; the script loads them offline and
writes its scratch files to a new temporary directory, whose path it prints.
"""

import json
import re
import sys
import tempfile
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
TRL_TEMPLATES = Path(trl.__file__).parent / "chat_templates"
MODEL, REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"
# Four fixtures, plus swe_rebench_openhands (a second tool schema) and swe_agent_nebius (a row with no assistant turn).
SOURCES = ("tau_bench", "tau2_bench", "swe_gym_openhands", "who_and_when", "swe_rebench_openhands", "swe_agent_nebius")
CALL = {"id": Value("string"), "type": Value("string"), "function": {"name": Value("string"), "arguments": Value("string")}}
FEATURES = Features({"messages": List({"role": Value("string"), "content": Value("string"), "name": Value("string"),
                                       "tool_call_id": Value("string"), "tool_calls": List(CALL)}),
                     "tools": Value("string")})
ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant(.*?<\|im_end\|>\n)", re.S)  # the text that a generation block covers


def chat_rows(name):
    """([(source, row, report entry)], number of runs) for one fixture; who_and_when speaks as the first agent of a run."""
    runs = list(ap.load(name, root=FIXTURES / name))
    items = []
    for group in ([[run] for run in runs] if name == "who_and_when" else [runs]):
        report = {}
        speaker = group[0].agents[0]["id"] if name == "who_and_when" else None
        rows = ap.to_chat(group, perspective=speaker, report=report)  # arguments="json" is the default
        items += zip([name] * len(rows), rows, [e for e in report["runs"] if e["row"] is not None])
    return items, len(runs)


def encode(row):
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


def main():
    disable_progress_bars()
    transformers.logging.set_verbosity_error()
    work = Path(tempfile.mkdtemp(prefix="agentpile-train-"))  # round-trip files, loader cache, trainer output

    items, n_runs = [], 0
    for name in SOURCES:
        got, runs = chat_rows(name)
        items, n_runs = items + got, n_runs + runs
    rows = [row for _, row, _ in items]
    assert all(e["assistant_turns"] == sum(m["role"] == "assistant" for m in r["messages"]) for _, r, e in items)
    named = ap.to_chat(ap.load("who_and_when", root=FIXTURES / "who_and_when"))  # no perspective: turns carry the agent name
    print(f"{len(rows)} chat rows from {len(SOURCES)} fixtures ({n_runs - len(rows)} runs without events give no row); "
          f"{sum('tools' in r for r in rows)} rows record tools")

    # (1) storage: explicit features, JSON text for tools and arguments, exact round trips
    varied = [r for r in rows if "tools" in r]
    schemas = {json.dumps(r["tools"], sort_keys=True) for r in varied}
    assert len(schemas) > 1 and all("name" in m for r in named for m in r["messages"] if m["role"] == "assistant")
    groups = (("no tools", [r for r in rows if "tools" not in r] + named), ("tool schemas", varied), ("all rows", rows + named))
    print("JSONL and Parquet round trips are exact:", {label: round_trip(group, work, label) for label, group in groups},
          f"(the tool batch holds {len(schemas)} different tool lists)")

    # (2) drop the rows without an assistant turn
    dropped = [f"{name}:{e['id']}" for name, _, e in items if e["assistant_turns"] == 0]
    train = Dataset.from_list([encode(r) for _, r, e in items if e["assistant_turns"] > 0], features=FEATURES)
    print(f"dropped {len(dropped)} rows without an assistant turn: {dropped}")

    # (3) template: TRL's training variant of the tokenizer's own template, bundled and checked against the pinned TRL
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    except OSError:
        sys.exit("tokenizer files are not in the cache: run the hf download line from the docstring first")
    template = (HERE / "qwen2_5_training.jinja").read_text()
    shipped = (TRL_TEMPLATES / "qwen2_5_training.jinja").read_text()
    assert template.endswith(shipped) and tok.chat_template == (TRL_TEMPLATES / "qwen2_5.jinja").read_text()
    assert get_training_chat_template(tok) == shipped and has_generation_markers(template)
    print(f"trl {trl.__version__} ships {len(list(TRL_TEMPLATES.glob('*_training.jinja')))} training templates; the bundled "
          f"qwen2_5_training.jinja is its copy, a patch of the template that {MODEL} ships")
    # TRL decodes a tools string itself but hands the template the stored messages, so arguments become objects only here
    apply_chat_template = tok.apply_chat_template
    tok.apply_chat_template = lambda messages, *a, **kw: apply_chat_template(messages_of(messages, objects=True), *a, **kw)
    sample = next(r for r in train if r["tools"] and any(m["tool_calls"] for m in r["messages"]))
    text = apply_chat_template(messages_of(sample["messages"]), tools=json.loads(sample["tools"]), chat_template=template, tokenize=False)
    assert '"arguments": "{' in text  # arguments left as JSON text would be encoded a second time by the template

    # (4) TRL: its own tokenization, label building and collation, with assistant-only loss
    config = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True)
    args = SFTConfig(output_dir=str(work / "out"), assistant_only_loss=True, chat_template_path=str(HERE / "qwen2_5_training.jinja"),
                     max_length=None,  # no truncation, so every assistant turn is checked
                     per_device_train_batch_size=4, use_cpu=True, bf16=False, report_to="none",
                     save_strategy="no", disable_tqdm=True)
    trainer = SFTTrainer(model=Qwen2ForCausalLM(config), args=args, train_dataset=train, processing_class=tok)
    assert tok.chat_template == template
    totals = [check_labels(tok, template, example) for example in trainer.train_dataset]
    n_tokens, n_trained, n_call = map(sum, zip(*totals))
    batches = list(trainer.get_train_dataloader())
    assert sum(int(b["attention_mask"].sum()) for b in batches) == n_tokens
    assert sum(int((b["labels"] != -100).sum()) for b in batches) == n_trained
    assert all(((b["labels"] == -100) | (b["attention_mask"] == 1)).all() for b in batches)  # padding is never trained
    print(f"SFTTrainer prepared {len(totals)} rows ({n_tokens} tokens): {n_trained} trained, {n_call} of them in tool calls, "
          f"{n_tokens - n_trained} masked (-100); the collator gave {len(batches)} padded batches with the same counts")

    # (5) templates that fail on this data, and what the Qwen template does to the rest
    print("templates that cannot give this data assistant masks (one row with tools):")
    objects, n_calls = messages_of(sample["messages"], objects=True), sum(len(m["tool_calls"] or []) for m in sample["messages"])
    for file in ("qwen2_5.jinja", "qwen3.jinja", "qwen3_instruct_2507.jinja", "qwen2_5_vl_training.jinja"):
        tried = (TRL_TEMPLATES / file).read_text()
        try:
            out = apply_chat_template(objects, tools=json.loads(sample["tools"]), chat_template=tried, return_dict=True,
                                      return_assistant_tokens_mask=True)
            calls = tok.decode(out["input_ids"]).count('", "arguments": ')
            why = (f"has a generation block but renders {calls} of {n_calls} tool calls" if has_generation_markers(tried)
                   else "no generation block") + f"; mask covers {sum(out['assistant_masks'])} of {len(out['input_ids'])} tokens"
        except Exception as exc:  # a template without tool-call support can fail on a call-only turn
            why = f"has a generation block but fails on this row: {type(exc).__name__}: {exc}"
        print(f"  {file}{' (the tokenizer own template)' if file == 'qwen2_5.jinja' else ''}: {why}")
    plain = apply_chat_template([{"role": "user", "content": "hi"}], chat_template=template, tokenize=False)
    print("default system prompt of the Qwen2.5 template:", repr(re.match(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", plain).group(1)),
          f"(inserted in {sum(r['messages'][0]['role'] != 'system' for r in train)} of {len(train)} training rows, which have no system message)")
    render = lambda r: apply_chat_template(r["messages"], chat_template=template, tokenize=False)
    nameless = [{**r, "messages": [{k: v for k, v in m.items() if k != "name"} for m in r["messages"]]} for r in named]
    assert all(render(a) == render(b) for a, b in zip(named, nameless))
    print(f"names are dropped: {len(named)} multi-agent rows render the same text with and without the name field")
    print(f"ok (scratch files in {work})")


if __name__ == "__main__":
    main()
