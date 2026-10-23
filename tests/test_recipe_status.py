"""tools/recipe_status.py, which keeps the README's Works With badges: when a row is passing, stale, or left alone.

The tool is not part of the package; the test reads it from the checkout and skips where the checkout has no tools/.
"""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "recipe_status.py"


@pytest.fixture
def status_tool(tmp_path, monkeypatch):
    if not TOOL.exists():
        pytest.skip("tools/recipe_status.py is not in this checkout")
    spec = importlib.util.spec_from_file_location("recipe_status", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "writer.py").write_text("v1\n", encoding="utf-8")
    (tmp_path / "req.txt").write_text("# pins\nno-such-package-for-agentpile==0\n", encoding="utf-8")
    return module


def checked(tool, monkeypatch, row, outcome, run_id, results):
    """Record one check the way the CI job does, as run ``run_id`` of the workflow."""
    monkeypatch.setenv("GITHUB_RUN_ID", str(run_id))
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    results.mkdir(exist_ok=True)
    tool.main(["record", "--row", row, "--outcome", outcome, "--requirements", "req.txt", "--version", "jaeger=2.22.0",
               "--covers", "code", "--out", str(results / f"{row}.json")])


def folded(tool, results, capsys):
    tool.main(["update", str(results), "status", "--root", "."])
    return capsys.readouterr().out.split()


def badge(row, suffix=""):
    return json.loads(Path("status", f"{row}{suffix}.json").read_text(encoding="utf-8"))["message"]


def test_a_row_is_stale_after_a_failure_or_a_code_change_until_its_check_passes(status_tool, monkeypatch, capsys):
    first = Path("r1")
    checked(status_tool, monkeypatch, "ingest-jaeger", "success", 10, first)
    checked(status_tool, monkeypatch, "view-otlp", "success", 10, first)
    assert folded(status_tool, first, capsys) == []
    rec = json.loads(Path("status/ingest-jaeger.record.json").read_text(encoding="utf-8"))
    assert badge("ingest-jaeger") == "passing"
    assert rec["last_run"]["versions"] == {"no-such-package-for-agentpile": None, "jaeger": "2.22.0"}
    date = badge("ingest-jaeger", "-last")

    Path("code/writer.py").write_text("v2\n", encoding="utf-8")  # the writer changes; only the offline check runs
    second = Path("r2")
    checked(status_tool, monkeypatch, "view-otlp", "success", 11, second)
    assert folded(status_tool, second, capsys) == []
    assert (badge("view-otlp"), badge("ingest-jaeger"), badge("ingest-jaeger", "-last")) == ("passing", "stale", date)

    third = Path("r3")
    checked(status_tool, monkeypatch, "view-otlp", "failure", 12, third)
    assert folded(status_tool, third, capsys) == ["view-otlp"]
    assert (badge("view-otlp"), badge("view-otlp", "-last")) == ("stale", date)

    fourth = Path("r4")
    checked(status_tool, monkeypatch, "ingest-jaeger", "success", 13, fourth)
    assert folded(status_tool, fourth, capsys) == []
    assert badge("ingest-jaeger") == "passing" and badge("view-otlp") == "stale"


def test_a_result_from_an_older_run_is_ignored(status_tool, monkeypatch, capsys):
    newer, older = Path("newer"), Path("older")
    checked(status_tool, monkeypatch, "train", "failure", 21, newer)
    checked(status_tool, monkeypatch, "train", "success", 20, older)
    assert folded(status_tool, newer, capsys) == ["train"]
    assert folded(status_tool, older, capsys) == []  # finished late: does not turn the newer failure green
    assert badge("train") == "stale" and badge("train", "-last") == "never"


def test_a_run_that_folds_late_from_an_older_checkout_is_judged_by_the_current_main(status_tool, monkeypatch, capsys):
    """Both directions of the race: run 10 checked ingestion on v1 but folds after run 11 checked view-otlp on v2."""
    Path("code/writer.py").write_text("v1\n", encoding="utf-8")
    checked(status_tool, monkeypatch, "ingest-jaeger", "success", 9, Path("r9"))
    checked(status_tool, monkeypatch, "ingest-jaeger", "success", 10, Path("r10"))
    checked(status_tool, monkeypatch, "train", "success", 10, Path("r10"))
    assert folded(status_tool, Path("r9"), capsys) == []

    Path("code/writer.py").write_text("v2\n", encoding="utf-8")  # main moves on; root is always a checkout of main
    checked(status_tool, monkeypatch, "view-otlp", "success", 11, Path("r11"))
    checked(status_tool, monkeypatch, "train", "success", 11, Path("r11"))
    assert folded(status_tool, Path("r11"), capsys) == []
    assert (badge("ingest-jaeger"), badge("view-otlp"), badge("train")) == ("stale", "passing", "passing")

    assert folded(status_tool, Path("r10"), capsys) == []  # newer than ingestion's run 9, older than train's run 11
    assert (badge("ingest-jaeger"), badge("view-otlp"), badge("train")) == ("stale", "passing", "passing")
    rec = json.loads(Path("status/ingest-jaeger.record.json").read_text(encoding="utf-8"))
    assert rec["last_run"]["run_id"] == 10 and rec["reason"] == "the code it covers changed after its last check"


def test_a_platform_records_its_own_version_and_an_unchecked_row_says_so(status_tool, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_RUN_ID", "30")
    Path("r").mkdir()
    status_tool.main(["record", "--row", "ingest-mlflow", "--outcome", "success", "--package", "pytest",
                      "--covers", "code", "--out", "r/ingest-mlflow.json"])
    status_tool.main(["update", "r", "status", "--expect", "ingest-mlflow", "--expect", "ingest-phoenix"])
    capsys.readouterr()
    rec = json.loads(Path("status/ingest-mlflow.record.json").read_text(encoding="utf-8"))
    assert list(rec["last_run"]["versions"]) == ["pytest"] and badge("ingest-mlflow") == "passing"
    assert (badge("ingest-phoenix"), badge("ingest-phoenix", "-last")) == ("not run yet", "never")
