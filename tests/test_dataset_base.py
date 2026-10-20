import json

import pytest

import agentpile as ap
from agentpile.corpora._base import CountMismatch, Dataset
from agentpile.formats.messages import messages_to_trajectory


class Toy(Dataset):
    name = "toy"
    format = "toy-jsonl"

    def __init__(self, expected):
        self._expected = expected

    @property
    def manifest(self):
        return {"hf_id": "someone/toy", "revision": "0" * 40, "counts": {"runs_after_load": {"value": self._expected}}}

    def records(self, root, **options):
        for f in sorted(root.glob("*.jsonl")):
            for i, line in enumerate(f.read_text().splitlines()):
                yield f"{f.name}:{i}", json.loads(line), {"file": f.name, "split": "train"}

    def convert(self, key, raw, where):
        ext = {"source": self.source(key, where)}
        return ap.Run(messages_to_trajectory(raw["messages"], trajectory_id=key, ext=ext))


@pytest.fixture
def data(tmp_path):
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(3)]
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_root_load_and_source_fields(data):
    runs = list(Toy(3).load(root=str(data)))
    assert [r.id for r in runs] == ["a.jsonl:0", "a.jsonl:1", "a.jsonl:2"]
    src = runs[0].source
    assert src["dataset"] == "someone/toy" and src["revision"] == "0" * 40 and src["file"] == "a.jsonl"
    assert ap.validate(runs[0]) == []


def test_count_check_on_request(data):
    assert len(list(Toy(4).load(root=str(data)))) == 3  # root copies are not checked by default
    with pytest.raises(CountMismatch, match="read 3 runs"):
        list(Toy(4).load(root=str(data), verify=True))
    assert len(list(Toy(3).load(root=str(data), verify=True))) == 3


def test_limit_skips_the_count_check(data):
    assert len(list(Toy(99).load(root=str(data), limit=2, verify=True))) == 2
