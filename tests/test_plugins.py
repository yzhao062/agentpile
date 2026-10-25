"""Gate 5 extensibility: a separately installed package adds a reader through an entry point, nested targets included."""

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import agentpile as ap

SRC = Path(ap.__file__).resolve().parents[1]
README = Path(__file__).resolve().parents[1] / "README.md"

PLUGIN = '''
from agentpile import Run, CONFIG_SCHEMA


class _Reader:
    name = "toy"

    @staticmethod
    def sniff(source):
        return getattr(source, "suffix", None) == ".toy"

    @staticmethod
    def read(source, **options):
        yield Run({"schema": CONFIG_SCHEMA, "source": {"format": "toy", "record_key": source.read_text().strip()}})


class plugins:  # a nested target: toy_agentpile_plugin:plugins.reader
    reader = _Reader
'''


def _install_toy_plugin(root: Path) -> None:
    pkg = root / "toy_agentpile_plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(PLUGIN)
    dist = root / "toy_agentpile_plugin-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: toy-agentpile-plugin\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text("[agentpile.readers]\ntoy = toy_agentpile_plugin:plugins.reader\n")


def test_installed_plugin_is_discovered_through_its_entry_point(tmp_path):
    _install_toy_plugin(tmp_path)
    data = tmp_path / "run.toy"
    data.write_text("toy-run-1\n")
    script = textwrap.dedent(f"""
        import agentpile as ap
        assert "toy" in ap.readers.names()
        run = next(ap.load(r"{data}"))
        print(run.source["format"], run.id)
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), str(SRC)]))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.split() == ["toy", "toy-run-1"]


def test_nested_string_target_resolves(tmp_path, monkeypatch):
    _install_toy_plugin(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    ap.register("toy-nested", "toy_agentpile_plugin:plugins.reader")
    try:
        assert ap.readers.get("toy-nested").name == "toy"
    finally:
        ap.readers.unregister("toy-nested")


def test_a_dependency_may_name_several_resources():
    record = json.loads((Path(__file__).parent / "fixtures" / "atif_multiagent.json").read_text())
    record["steps"][3]["extra"]["agentpile"]["deps"] = [
        {"src": {"trajectory": "fx-1", "step": 3, "call": "c1"}, "dst": {"trajectory": "fx-1", "step": 4},
         "rule": "value_flow", "rule_version": "1", "resource": ["7", "LAX"]}]
    assert ap.validate(record) == []


def _readme_block(marker):
    """The README's Python block that contains ``marker``; blocks are found by content so they can move."""
    blocks = [b for b in re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S) if marker in b]
    assert blocks, f"no Python block in README.md contains {marker!r}"
    return blocks[0]


def test_readme_core_example_runs_without_extras(tmp_path, monkeypatch):
    """The README's in-memory example needs only the core and writes run.json to the working directory."""
    monkeypatch.chdir(tmp_path)
    exec(compile(_readme_block("messages = ["), "README.md", "exec"), {"print": lambda *a, **k: None})
    record = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert record["schema_version"] == "ATIF-v1.7" and ap.validate(record) == []


@pytest.mark.network
def test_readme_dataset_example_runs(tmp_path):
    """The README's dataset example, with the file it loads pointed at a real trace."""
    trace = Path(__file__).parent / "fixtures" / "atif_multiagent.json"
    code = _readme_block('ap.load("run.json")').replace('"run.json"', repr(str(trace)))
    data = os.environ.get("AGENTPILE_DATA")
    if data:  # read the pinned copy instead of downloading
        code = code.replace('ap.load("who_and_when", ', f'ap.load("who_and_when", root={str(Path(data) / "who_and_when")!r}, ')
    exec(compile(code, "README.md", "exec"), {"ap": ap, "print": lambda *a, **k: None})  # the block follows the first
