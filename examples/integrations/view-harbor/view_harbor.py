"""Open runs in Harbor's viewer.

`harbor view` reads job folders of ATIF v1.7 trajectories, and an agentpile run is one, so `run.to_json` writes the
trajectory file as it is. This writes the runs as a Harbor job, one trial per run, with the config.json and result.json
files the viewer lists. A trial's reward is the run's recorded outcome (1.0 or 0.0; none if the run records none). The
viewer does not draw the agentpile labels; they stay in the file under extra["agentpile"].

    pip install -e ".[hf]" -r examples/integrations/view-harbor/requirements.txt
    python examples/integrations/view-harbor/view_harbor.py                        # the bundled fixtures, with the checks
    python examples/integrations/view-harbor/view_harbor.py tau_bench --limit 20   # your runs: a dataset name or a trace file
    harbor view DIR                                                                # DIR is the folder the script prints
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import agentpile as ap
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trajectories import Trajectory
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "corpora" / "fixtures"
DATASETS = ("tau_bench", "trail", "who_and_when")
JOB = "agentpile-runs"


def trial_name(run):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", run.id)


def write_job(runs, jobs_dir):
    """The agentpile part: each run's own JSON is a trial's trajectory; the rest is Harbor bookkeeping."""
    job_dir = jobs_dir / JOB
    job_dir.mkdir(parents=True)
    results = []
    for run in runs:
        name = trial_name(run)
        (job_dir / name / "agent").mkdir(parents=True)
        run.to_json(job_dir / name / "agent" / "trajectory.json")
        agent, success = run.record["agent"], run.outcome.get("success")
        task = TaskConfig(path=Path("agentpile") / name)
        result = TrialResult(
            task_name=run.id, trial_name=name, trial_uri=(job_dir / name).as_uri(), task_id=task.get_task_id(),
            source=run.source.get("dataset"), task_checksum="none",
            config=TrialConfig(task=task, trial_name=name, trials_dir=job_dir),
            agent_info=AgentInfo(name=agent["name"], version=agent["version"],
                                 model_info=ModelInfo(name=agent["model_name"]) if agent.get("model_name") else None),
            verifier_result=None if success is None else VerifierResult(rewards={"success": float(success)}))
        (job_dir / name / "result.json").write_text(result.model_dump_json(indent=2))
        results.append(result)
    now = datetime.now(timezone.utc)
    (job_dir / "config.json").write_text(JobConfig(job_name=JOB, jobs_dir=jobs_dir, agents=[]).model_dump_json(indent=2))
    (job_dir / "result.json").write_text(JobResult(
        id=uuid.uuid4(), started_at=now, finished_at=now, n_total_trials=len(runs),
        stats=JobStats.from_trial_results(results, n_total_trials=len(runs))).model_dump_json(indent=2))


def check_viewer(runs, jobs_dir):
    """Start `harbor view` on a free port, read what the browser reads, and stop it."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback only, whatever the proxy settings

    def get(path):
        with opener.open(f"http://127.0.0.1:{port}{path}", timeout=10) as response:
            return json.load(response)

    with tempfile.TemporaryFile("w+") as log:
        server = subprocess.Popen([sys.executable, "-m", "harbor.cli.main", "view", str(jobs_dir), "--jobs",
                                   "--no-build", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while True:
                try:
                    get("/api/health")
                    break
                except OSError:
                    log.seek(0)
                    assert server.poll() is None and time.monotonic() < deadline, f"harbor view did not start:\n{log.read()}"
                    time.sleep(0.5)
            listing = get("/api/jobs")
            assert [(j["name"], j["n_total_trials"], j["n_completed_trials"]) for j in listing["items"]] \
                == [(JOB, len(runs), len(runs))], f"job listing: {listing}"
            trials = get(f"/api/jobs/{JOB}/trials")
            assert trials["total"] == len(runs) and {t["name"] for t in trials["items"]} == {trial_name(r) for r in runs}, \
                "trial listing"
            for run in runs:
                assert get(f"/api/jobs/{JOB}/trials/{trial_name(run)}/trajectory") == run.record, \
                    f"{run.id}: served trajectory"
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()


def check(runs, jobs_dir, viewer=True):
    """Load every file back with Harbor's own models; each trajectory must also equal the agentpile record."""
    job_dir = jobs_dir / JOB
    for run in runs:
        name = trial_name(run)
        text = (job_dir / name / "agent" / "trajectory.json").read_text(encoding="utf-8")
        trajectory = Trajectory.model_validate_json(text)
        assert trajectory.schema_version == "ATIF-v1.7", f"{run.id}: schema {trajectory.schema_version}"
        assert json.loads(text) == run.record, f"{run.id}: trajectory.json differs from the agentpile record"
        result = TrialResult.model_validate_json((job_dir / name / "result.json").read_text())
        assert result.trial_name == name and result.task_name == run.id, f"{run.id}: trial result"
        assert (result.verifier_result is None) == (run.outcome.get("success") is None), f"{run.id}: reward"
    JobConfig.model_validate_json((job_dir / "config.json").read_text())
    job = JobResult.model_validate_json((job_dir / "result.json").read_text())
    assert job.n_total_trials == job.stats.n_completed_trials == len(runs), "job result counts"
    if viewer:
        check_viewer(runs, jobs_dir)


def load_runs(sources, limit):
    """The first `limit` runs of each source; a dataset name takes limit= and downloads only that part, a file path does not."""
    return [run for source in sources
            for run in islice(ap.load(source, **({"limit": limit} if source in ap.datasets.names() else {})), limit)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", nargs="*", help="a dataset name or a trace file (default: the bundled fixtures, with the checks)")
    parser.add_argument("--limit", type=int, default=20, help="runs per source (default: 20)")
    parser.add_argument("--out", type=Path, help="directory for the job (default: a new temporary one)")
    parser.add_argument("--no-viewer", action="store_true", help="skip starting `harbor view` in the checks")
    args = parser.parse_args()
    if args.source:
        runs = load_runs(args.source, args.limit)
    else:
        runs = [run for name in DATASETS for run in ap.load(name, root=FIXTURES / name)]
    jobs_dir = (args.out or Path(tempfile.mkdtemp(prefix="agentpile-harbor-"))).resolve()
    write_job(runs, jobs_dir)
    print(f"wrote {len(runs)} trials to {jobs_dir}")
    print(f"open with: harbor view {jobs_dir}")
    if not args.source:
        check(runs, jobs_dir, viewer=not args.no_viewer)
        print("all checks passed")


if __name__ == "__main__":
    main()
