"""``ap.load(name, limit=N)`` downloads only the files the first N runs need.

A fake hub serves the synthetic corpus fixtures as the pinned files of each dataset and records every call, so
nothing here uses the network.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agentpile as ap
import agentpile.corpora._base as base
from agentpile.corpora._base import Dataset
from agentpile.formats.messages import messages_to_trajectory

FIXTURES = Path(__file__).parent / "corpora" / "fixtures"
MARKER = ".agentpile-complete.json"
PARQUET = {"swe_agent_nebius", "swe_gym_openhands", "swe_rebench_openhands", "swe_smith", "trail"}
NAMES = ["adp", "agent_reward_bench", "agentdojo", "mast", "swe_agent_nebius", "swe_gym_openhands",
         "swe_rebench_openhands", "swe_smith", "tau2_bench", "tau_bench", "trail", "who_and_when"]
REV = "1" * 40


class FakeHub:
    """A directory served as the pinned files of one hub dataset, with a record of what is fetched."""

    EXTRA = {"README.md": b"readme\n", ".gitattributes": b"*.parquet filter=lfs\n", "config.toml": b"x = 1\n",
             "notes/extra.txt": b"notes\n"}  # files a repository has that the manifests mostly do not pin

    def __init__(self, src, revision):
        self.src, self.revision = Path(src), revision
        self.names = sorted([p.relative_to(self.src).as_posix() for p in self.src.rglob("*") if p.is_file()]
                            + list(self.EXTRA))
        self.offline = False
        self.reset()

    def reset(self):
        self.listings, self.downloads, self.snapshots = 0, [], []

    def pinned(self, patterns):
        from huggingface_hub.utils import filter_repo_objects

        return set(filter_repo_objects(self.names, allow_patterns=patterns))

    def _read(self, name):
        return self.EXTRA[name] if name in self.EXTRA else (self.src / name).read_bytes()

    def _write(self, name, local_dir):
        target = Path(local_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self._read(name))
        return str(target)

    def _check(self, repo_type, revision):
        assert not self.offline, "the hub was used although the files should be on disk"
        assert repo_type == "dataset" and revision == self.revision

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, revision=None, local_dir=None, **_):
        self._check(repo_type, revision)
        assert filename in self.names, filename
        self.downloads.append(filename)
        return self._write(filename, local_dir)

    def snapshot_download(self, repo_id, *, repo_type=None, revision=None, allow_patterns=None, local_dir=None, **_):
        self._check(repo_type, revision)
        fresh = [n for n in sorted(self.pinned(allow_patterns)) if not (Path(local_dir) / n).is_file()]
        for name in fresh:
            self._write(name, local_dir)
        self.snapshots.append(fresh)
        return str(local_dir)

    def install(self, monkeypatch):
        import huggingface_hub

        hub = self

        class Api:
            def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, expand=False, revision=None,
                               repo_type=None, token=None):
                hub._check(repo_type, revision)
                assert recursive
                hub.listings += 1
                folders = {"/".join(n.split("/")[:i]) for n in hub.names for i in range(1, len(n.split("/")))}
                entries = ([SimpleNamespace(path=n, size=len(hub._read(n))) for n in hub.names]
                           + [SimpleNamespace(path=f) for f in folders])  # a folder has no size
                return iter(sorted(entries, key=lambda e: e.path, reverse=True))  # the hub promises no order

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", self.hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", self.snapshot_download)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTPILE_CACHE", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _hub(monkeypatch, src, revision):
    pytest.importorskip("huggingface_hub")
    hub = FakeHub(src, revision)
    hub.install(monkeypatch)
    return hub


def _local(cache, ds):
    return cache / ds.name / ds.revision


def _present(local):
    return {p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file() and p.name != MARKER}


@pytest.fixture(params=NAMES)
def served(request, cache, monkeypatch):
    """(dataset, its runs from the fixture directory, the fake hub serving that directory)."""
    name = request.param
    if name in PARQUET:
        pytest.importorskip("pyarrow")
    ds = ap.datasets.get(name)
    src = FIXTURES / name
    full = list(ap.load(name, root=str(src)))
    assert len(full) >= 2
    return ds, full, _hub(monkeypatch, src, ds.revision)


def test_every_hub_dataset_but_one_can_fetch_files_one_at_a_time():
    assert all(ap.datasets.get(n).incremental for n in NAMES)
    assert not ap.datasets.get("catchbench_pre").incremental  # fetched from GitHub, not the hub


def test_a_limited_load_fetches_only_the_files_of_its_runs(served, cache):
    ds, full, hub = served
    pinned = hub.pinned(ds.allow_patterns)
    for k in sorted({1, 2, len(full)}):
        ap.clear_cache()
        hub.reset()
        got = list(ap.load(ds.name, limit=k, verify=False))
        assert got == full[:k]
        needed = set(ds.upfront) | {r.source["file"] for r in full[:k]}
        assert set(hub.downloads) == needed and len(hub.downloads) == len(needed)
        assert needed <= pinned
        if k == 1:
            assert needed < pinned  # the README, config and the other data files stay on the hub
        assert hub.listings == 1 and not hub.snapshots
        assert not (_local(cache, ds) / MARKER).exists()
        assert _present(_local(cache, ds)) == needed


def test_a_later_full_load_reuses_the_files_a_limited_load_left(served, cache):
    ds, full, hub = served
    list(ap.load(ds.name, limit=1, verify=False))
    first = _present(_local(cache, ds))
    hub.reset()
    assert list(ap.load(ds.name, verify=False)) == full
    assert not hub.downloads and len(hub.snapshots) == 1
    assert set(hub.snapshots[0]) == hub.pinned(ds.allow_patterns) - first  # only what was still missing
    assert json.loads((_local(cache, ds) / MARKER).read_text())["revision"] == ds.revision
    # a finished download answers a limited load without any network
    hub.offline = True
    assert list(ap.load(ds.name, limit=1, verify=False)) == full[:1]


def test_a_full_download_that_falls_back_to_the_partial_copy_is_not_marked_complete(served, cache, monkeypatch):
    ds, full, hub = served
    list(ap.load(ds.name, limit=1, verify=False))
    if hub.pinned(ds.allow_patterns) <= _present(_local(cache, ds)):
        pytest.skip("the first run's files are the whole pinned set")
    import huggingface_hub

    online = huggingface_hub.snapshot_download
    # Offline, snapshot_download can hand back the local directory as it is.
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, local_dir=None, **k: str(local_dir))
    with pytest.raises(FileNotFoundError, match="not marked complete"):
        list(ap.load(ds.name, verify=False))
    assert not (_local(cache, ds) / MARKER).exists()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", online)
    hub.reset()
    assert list(ap.load(ds.name, verify=False)) == full  # back online: the rest is fetched (fixtures hold fewer runs)
    assert hub.snapshots and hub.snapshots[0] and (_local(cache, ds) / MARKER).exists()


def test_the_cache_commands_see_the_partial_download(served, cache):
    ds, _, _ = served
    list(ap.load(ds.name, limit=1, verify=False))
    (info,) = ap.cache_info()
    assert (info["dataset"], info["revision"]) == (ds.name, ds.revision) and info["bytes"] > 0
    assert ap.clear_cache(ds.name) == [str((cache / ds.name).resolve())]
    assert ap.cache_info() == []


def test_the_shared_dataset_object_keeps_no_state(served):
    ds, full, _ = served
    one, two = ap.load(ds.name, limit=1, verify=False), ap.load(ds.name, limit=2, verify=False)
    assert [next(one), next(two), next(two)] == [full[0], full[0], full[1]]  # two loads at once do not mix
    assert "_partial" not in ds.__dict__


@pytest.mark.parametrize("name, options, files", [
    ("adp", {"config": "synatra"}, {"synatra/full_std.jsonl"}),
    ("tau_bench", {"model": "model-gem"}, {"model-gem.jsonl"}),
    ("who_and_when", {"subset": "Hand-Crafted"}, {"Who&When/Hand-Crafted/1.json"}),
    ("agent_reward_bench", {"benchmark": "workarena"}, {
        "data/annotations.csv", "cleaned/workarena/GenericAgent-synth-claude/GenericAgent-synth-claude_on_workarena.servicenow/"
        "workarena.servicenow.order-synth-laptop.json"}),
])
def test_a_filter_keeps_files_it_skips_off_the_disk(name, options, files, cache, monkeypatch):
    ds = ap.datasets.get(name)
    hub = _hub(monkeypatch, FIXTURES / name, ds.revision)
    want = list(ap.load(name, root=str(FIXTURES / name), **options))
    assert list(ap.load(name, limit=1, verify=False, **options)) == want[:1]
    assert set(hub.downloads) == files


def test_a_failed_download_surfaces_when_the_reader_reaches_the_file(cache, monkeypatch):
    ds = ap.datasets.get("who_and_when")
    hub = _hub(monkeypatch, FIXTURES / "who_and_when", ds.revision)

    def broken(*args, **kwargs):
        raise OSError("connection reset")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", broken)
    runs = ap.load("who_and_when", limit=1)  # the listing is fetched here, no data file yet
    assert hub.listings == 1
    with pytest.raises(OSError, match="connection reset"):
        next(runs)
    assert not (_local(cache, ds) / MARKER).exists()


def _refusal():
    from huggingface_hub.utils import GatedRepoError

    exc = GatedRepoError.__new__(GatedRepoError)  # the hub builds it from an HTTP response; none is needed here
    Exception.__init__(exc, "Access to dataset is restricted. You must have access to it and be authenticated.")
    return exc


@pytest.mark.parametrize("call, limit", [("HfApi", 1), ("hf_hub_download", 1), ("snapshot_download", None)])
def test_a_gated_dataset_says_how_to_get_access(call, limit, cache, monkeypatch):
    ds = ap.datasets.get("who_and_when")
    _hub(monkeypatch, FIXTURES / "who_and_when", ds.revision)
    import huggingface_hub

    def refuse(*args, **kwargs):
        raise _refusal()

    def refuse_while_listing(*args, **kwargs):  # the hub's listing is lazy: its request runs during iteration
        yield SimpleNamespace(path="README.md", size=7)
        raise _refusal()

    if call == "HfApi":  # the listing, which a limited and a full load both start with
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(list_repo_tree=refuse_while_listing))
    else:
        monkeypatch.setattr(huggingface_hub, call, refuse)
    options = {} if limit is None else {"limit": limit}
    with pytest.raises(ap.GatedDataset) as info:
        list(ap.load("who_and_when", verify=False, **options))
    message = str(info.value)
    assert "https://huggingface.co/datasets/Kevin355/Who_and_When" in message and "HF_TOKEN" in message
    assert "restart the session" in message  # the hub reads a Colab secret once per session
    assert isinstance(info.value, PermissionError)
    assert type(info.value.__cause__).__name__ == "GatedRepoError"
    assert not (_local(cache, ds) / MARKER).exists()


# -- the generic mechanism, on a tree that is not one of the datasets --------------------------------------------

class Tree(Dataset):
    name = "tree"
    format = "tree-json"
    incremental = True
    upfront = ("meta.json",)
    manifest = {"hf_id": "someone/tree", "revision": REV, "allow_patterns": ["*.json"]}

    def records(self, root, **options):
        meta = json.loads((root / "meta.json").read_text())
        for path in self.paths(root, "*/*.json"):
            rel = path.relative_to(root).as_posix()
            yield rel, json.loads(self.ready(path).read_text()), {"file": rel, "meta": meta}

    def convert(self, key, raw, where):
        return ap.Run(messages_to_trajectory(raw["messages"], trajectory_id=key, ext={"source": self.source(key, where)}))


@pytest.fixture
def tree(tmp_path):
    src = tmp_path / "tree"
    for rel in ("a-b/x.json", "a/x.json", "a/y.json", "z/1.json"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(json.dumps({"messages": [{"role": "user", "content": rel}]}))
    (src / "meta.json").write_text("{}")
    return src


def test_files_come_in_the_order_pathlib_gives_them_on_disk(tree, cache, monkeypatch):
    hub = _hub(monkeypatch, tree, REV)
    ds = Tree()
    disk = [p.relative_to(tree).as_posix() for p in sorted(tree.glob("*/*.json"))]
    assert disk == ["a/x.json", "a/y.json", "a-b/x.json", "z/1.json"]  # not the order of the strings
    assert [r.id for r in ds.load(root=str(tree))] == disk
    assert [r.id for r in ds.load(limit=10, verify=False)] == disk
    assert hub.downloads == ["meta.json"] + disk
    hub.reset()
    ap.clear_cache()
    assert [r.id for r in ds.load(limit=2, verify=False)] == disk[:2]
    assert hub.downloads == ["meta.json", "a/x.json", "a/y.json"]


def test_a_pattern_means_what_path_glob_means(tree, cache, monkeypatch):
    (tree / "a" / "x.txt").write_text("-")
    (tree / "deep" / "er").mkdir(parents=True)
    (tree / "deep" / "er" / "x.json").write_text("{}")
    (tree / "ab").mkdir()
    (tree / "ab" / "x.json").write_text("{}")
    _hub(monkeypatch, tree, REV)
    partial = base.Partial("tree", "someone/tree", REV, None)
    for pattern in ("*.json", "*/*.json", "a/*", "?/x.json", "[az]/*.json", "a*/x.json", "*/*/*.json", "meta.json"):
        want = sorted(p.relative_to(tree).as_posix() for p in tree.glob(pattern))
        assert partial.match(pattern) == want, pattern
    with pytest.raises(ValueError, match="recursive"):
        partial.match("**/*.json")


def test_an_upfront_file_the_revision_lacks_is_an_error(tree, cache, monkeypatch):
    class NeedsMore(Tree):
        upfront = ("meta.json", "absent.json")

    _hub(monkeypatch, tree, REV)
    with pytest.raises(FileNotFoundError, match="absent.json"):
        NeedsMore().load(limit=1)


# -- cases that need no hub library -----------------------------------------------------------------------------

def test_limit_with_root_never_touches_the_hub(monkeypatch):
    def no_hub():
        raise AssertionError("the hub was used")

    monkeypatch.setattr(base, "require_hub", no_hub)
    root = FIXTURES / "who_and_when"
    assert [r.id for r in ap.load("who_and_when", root=str(root), limit=2)] == [
        "Algorithm-Generated/1", "Algorithm-Generated/2"]


def test_a_limited_load_without_the_hf_extra_fails_at_call_time(cache, monkeypatch):
    def no_hub():
        raise ImportError("loading a Hugging Face dataset needs the hf extra: pip install 'agentpile[hf]'")

    monkeypatch.setattr(base, "require_hub", no_hub)
    with pytest.raises(ImportError, match="hf extra"):
        ap.load("who_and_when", limit=2)


def test_a_finished_download_answers_a_limited_load_without_the_hub(cache, monkeypatch):
    ds = ap.datasets.get("who_and_when")
    local = _local(cache, ds)
    local.mkdir(parents=True)
    for p in (FIXTURES / "who_and_when").rglob("*.json"):
        target = local / p.relative_to(FIXTURES / "who_and_when")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    (local / MARKER).write_text(json.dumps(
        {"repo_id": ds.repo_id, "revision": ds.revision, "allow_patterns": ds.allow_patterns}) + "\n")

    def no_hub():
        raise AssertionError("the hub was used")

    monkeypatch.setattr(base, "require_hub", no_hub)
    assert [r.id for r in ap.load("who_and_when", limit=2, verify=False)] == [
        "Algorithm-Generated/1", "Algorithm-Generated/2"]


def test_a_dataset_that_cannot_fetch_one_file_at_a_time_downloads_everything(tmp_path, monkeypatch):
    class Plain(Dataset):
        name = "plain"
        format = "plain-jsonl"
        manifest = {"hf_id": "someone/plain", "revision": REV}

        def records(self, root, **options):
            for f in sorted(root.glob("*.jsonl")):
                for i, line in enumerate(f.read_text().splitlines()):
                    yield f"{f.name}:{i}", json.loads(line), {"file": f.name}

        def convert(self, key, raw, where):
            return ap.Run(messages_to_trajectory(raw["messages"], trajectory_id=key))

    for n in range(2):
        row = {"messages": [{"role": "user", "content": str(n)}]}
        (tmp_path / f"{n}.jsonl").write_text(json.dumps(row))
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return tmp_path

    monkeypatch.setattr(base, "fetch", fetch)
    assert not Plain.incremental
    assert [r.id for r in Plain().load(limit=1)] == ["0.jsonl:0"]
    assert len(calls) == 1
