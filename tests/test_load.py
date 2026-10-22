import json
from pathlib import Path

import pytest

import agentpile as ap

FIX = Path(__file__).parent / "fixtures"


def test_load_atif_file():
    runs = list(ap.load(FIX / "atif_multiagent.json"))
    assert len(runs) == 1
    run = runs[0]
    assert run.kind == "trajectory" and run.id == "fx-1"
    assert len(run.steps) == 5


def test_load_accepts_str_path_and_dict():
    run = next(ap.load(str(FIX / "atif_multiagent.json")))
    record = json.loads((FIX / "atif_multiagent.json").read_text())
    assert next(ap.load(record)) == run


def test_load_jsonl_yields_each_trajectory():
    assert [r.id for r in ap.load(FIX / "atif_two.jsonl")] == ["fx-1", "fx-2"]


def test_load_config_record():
    run = next(ap.load(FIX / "config_mcp.json"))
    assert run.kind == "config"
    assert set(run.tools) == {"fs.read_file", "fs.delete_file"}


def test_unknown_name_is_not_guessed():
    with pytest.raises(FileNotFoundError, match="neither a registered dataset"):
        ap.load("no_such_dataset")


def test_a_misspelled_dataset_name_gets_a_suggestion():
    with pytest.raises(FileNotFoundError, match="did you mean 'who_and_when'"):
        ap.load("who_and_whenn")


def test_a_missing_path_is_reported_as_a_file(tmp_path):
    for missing in ("trace.json", str(tmp_path / "trace"), tmp_path / "x"):
        with pytest.raises(FileNotFoundError, match="no such file"):
            ap.load(missing)


def test_a_dataset_that_cannot_be_fetched_fails_at_call_time(tmp_path, monkeypatch):
    import agentpile.corpora._base as base

    def no_hub():
        raise ImportError("loading a Hugging Face dataset needs the hf extra: pip install 'agentpile[hf]'")

    monkeypatch.setenv("AGENTPILE_CACHE", str(tmp_path))
    monkeypatch.setattr(base, "require_hub", no_hub)
    with pytest.raises(ImportError, match="hf extra"):
        ap.load("who_and_when")


def test_unrecognized_file():
    with pytest.raises(ap.UnrecognizedSource, match="pass reader="):
        ap.load(FIX / "not_a_trace.json")


class _Greedy:
    @staticmethod
    def sniff(source):
        return True

    @staticmethod
    def read(source, **options):
        yield ap.Run({"schema": ap.CONFIG_SCHEMA, "source": {"format": "greedy"}})


def test_ambiguous_input_needs_explicit_reader():
    ap.register("greedy", _Greedy)
    try:
        with pytest.raises(ap.AmbiguousSource, match="atif, greedy"):
            ap.load(FIX / "atif_multiagent.json")
        assert next(ap.load(FIX / "atif_multiagent.json", reader="atif")).id == "fx-1"
        assert next(ap.load(FIX / "atif_multiagent.json", reader="greedy")).source["format"] == "greedy"
    finally:
        ap.readers.unregister("greedy")


def test_register_rejects_incomplete_plugin():
    with pytest.raises(TypeError, match="lacks read"):
        ap.register("half", type("Half", (), {"sniff": staticmethod(lambda s: False)}))


def test_register_refuses_silent_override():
    with pytest.raises(ValueError, match="already registered"):
        ap.register("atif", _Greedy)


def test_registered_dataset_loads_by_name():
    class Toy:
        @staticmethod
        def load(**options):
            return ap.load(FIX / "atif_two.jsonl")

    ap.register("toy", Toy, kind="dataset")
    try:
        assert [r.id for r in ap.load("toy")] == ["fx-1", "fx-2"]
    finally:
        ap.datasets.unregister("toy")


def test_untested_atif_version_is_reported():
    record = json.loads((FIX / "atif_multiagent.json").read_text())
    record["schema_version"] = "ATIF-v1.2"
    with pytest.raises(ap.UnsupportedVersion, match="ATIF-v1.2"):
        list(ap.load(record))


def test_reader_options_are_not_swallowed():
    with pytest.raises(TypeError, match="takes no options"):
        list(ap.load(FIX / "atif_multiagent.json", typo_option=1))
