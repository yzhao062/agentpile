import pytest

import agentpile as ap
from agentpile import cache


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv(cache.ENV, str(tmp_path / "apcache"))
    return tmp_path / "apcache"


def _fill(name, rev, size=10):
    d = cache.dataset_dir(name, rev)
    d.mkdir(parents=True)
    (d / "data.json").write_bytes(b"x" * size)
    return d


def test_layout_and_info(root):
    _fill("who_and_when", "abc", 10)
    _fill("mast", "def", 5)
    info = ap.cache_info()
    assert [(i["dataset"], i["revision"], i["bytes"]) for i in info] == [("mast", "def", 5), ("who_and_when", "abc", 10)]
    assert all(i["path"].startswith(str(root)) for i in info)


def test_clear_one_dataset_then_all(root):
    _fill("who_and_when", "abc")
    _fill("mast", "def")
    assert len(ap.clear_cache("mast")) == 1
    assert [i["dataset"] for i in ap.cache_info()] == ["who_and_when"]
    ap.clear_cache()
    assert ap.cache_info() == []


def test_bad_names_cannot_escape_the_root(root):
    for bad in ("..", "../x", "a/b", ""):
        with pytest.raises(ValueError):
            cache.dataset_dir(bad, "abc")
    with pytest.raises(ValueError):
        ap.clear_cache(revision="abc")


def test_clear_on_missing_root_is_a_no_op(root):
    assert ap.clear_cache() == []
