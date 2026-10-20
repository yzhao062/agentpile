"""The ATIF part of every record validates against Harbor's own models, pinned at harbor 0.22.0.

harbor 0.22.0 ships the ATIF v1.7 models byte-identical to harbor-framework/harbor
commit bc5594425140fe02ee94a360f62f981ac5d08fcb. Install them without Harbor's
runtime stack: ``pip install --no-deps harbor==0.22.0 'pydantic>=2.11.7'``.
The tests skip when the package is absent, so the core suite stays dependency-free.
"""

import json
from pathlib import Path

import pytest

import agentpile as ap
from agentpile.formats.messages import messages_to_trajectory

trajectories = pytest.importorskip("harbor.models.trajectories")

FIX = Path(__file__).parent / "fixtures"
CORPORA = Path(__file__).parent / "corpora" / "fixtures"


def _harbor_valid(record):
    trajectories.Trajectory.model_validate(record)


@pytest.mark.parametrize("path", sorted(p.name for p in FIX.glob("atif_*")))
def test_fixture_trajectories(path):
    for run in ap.load(FIX / path):
        _harbor_valid(run.record)


@pytest.mark.parametrize("name", sorted(p.name for p in CORPORA.iterdir() if p.is_dir() and p.name in ap.datasets))
def test_dataset_loaders(name):
    try:
        runs = list(ap.load(name, root=str(CORPORA / name)))
    except ImportError as exc:  # a loader whose fixture needs pyarrow
        pytest.skip(str(exc))
    assert runs
    for run in runs:
        if run.kind == "trajectory":
            _harbor_valid(run.record)


def test_message_parser_output():
    from test_messages import CONVERSATION

    _harbor_valid(messages_to_trajectory(CONVERSATION, trajectory_id="t"))
    _harbor_valid(messages_to_trajectory([], trajectory_id="empty"))


def test_harbor_and_our_validator_agree_on_a_cross_step_result():
    record = json.loads((FIX / "atif_multiagent.json").read_text())
    record["steps"][3]["observation"]["results"][0]["source_call_id"] = "c1"  # a call on step 3, not step 4
    assert any("matches no call on this step" in e for e in ap.validate(record))
    with pytest.raises(Exception):
        _harbor_valid(record)
