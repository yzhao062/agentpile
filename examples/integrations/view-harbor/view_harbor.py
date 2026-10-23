"""Browse agentpile runs in Harbor's local viewer (`harbor view`), which reads ATIF v1.7 trajectories.

agentpile's run record is an ATIF v1.7 trajectory, so `run.to_json` writes the file Harbor shows as it is. This example
loads the tau_bench, trail and who_and_when fixtures from the checkout (16 runs; no network, API key or model call),
writes them as a Harbor job (one trial per run, plus the config.json and result.json files the viewer lists), loads
every file back with Harbor's own pydantic models, then starts `harbor view` on a free local port, reads its job,
trial and trajectory endpoints, and stops it. A trial's reward is the run's recorded outcome (1.0 or 0.0; TRAIL
records none). The viewer does not draw the agentpile labels; they stay in the file under extra["agentpile"].

    pip install -e ".[hf]" -r examples/integrations/view-harbor/requirements.txt
    python examples/integrations/view-harbor/view_harbor.py [OUT_DIR] [--no-viewer] [--hub]

OUT_DIR (default: a new temporary directory) keeps the job, so `harbor view OUT_DIR` browses it afterwards.
--no-viewer skips the server check; --hub loads 3 runs per dataset from Hugging Face (TRAIL is gated) instead.
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


def check(ok, what):
    if not ok:
        sys.exit(f"CHECK FAILED: {what}")


def write_job(runs, jobs_dir):
    """The agentpile part: each run's own JSON becomes a trial's trajectory; the rest is Harbor bookkeeping."""
    job_dir = jobs_dir / JOB
    check(not job_dir.exists(), f"{job_dir} exists; pass a new OUT_DIR so old and new trials never mix")
    job_dir.mkdir(parents=True)
    names, results = [], []
    for run in runs:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", run.id)
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
        names.append(name)
        results.append(result)
    now = datetime.now(timezone.utc)
    (job_dir / "config.json").write_text(JobConfig(job_name=JOB, jobs_dir=jobs_dir).model_dump_json(indent=2))
    (job_dir / "result.json").write_text(JobResult(
        id=uuid.uuid4(), started_at=now, finished_at=now, n_total_trials=len(runs),
        stats=JobStats.from_trial_results(results, n_total_trials=len(runs))).model_dump_json(indent=2))
    return job_dir, names


def check_files(runs, job_dir, names):
    """Load every file back with Harbor's own models; each trajectory must also equal the agentpile record."""
    steps = 0
    for run, name in zip(runs, names):
        text = (job_dir / name / "agent" / "trajectory.json").read_text(encoding="utf-8")
        trajectory = Trajectory.model_validate_json(text)
        check(trajectory.schema_version == "ATIF-v1.7", f"{run.id}: schema {trajectory.schema_version}")
        check(json.loads(text) == run.record, f"{run.id}: trajectory.json differs from the agentpile record")
        steps += len(trajectory.steps)
        result = TrialResult.model_validate_json((job_dir / name / "result.json").read_text())
        check(result.trial_name == name and result.task_name == run.id, f"{run.id}: trial result")
        check((result.verifier_result is None) == (run.outcome.get("success") is None), f"{run.id}: reward")
    JobConfig.model_validate_json((job_dir / "config.json").read_text())
    job = JobResult.model_validate_json((job_dir / "result.json").read_text())
    check(job.n_total_trials == job.stats.n_completed_trials == len(runs), "job result counts")
    return steps


def check_viewer(runs, jobs_dir, names):
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
                    check(server.poll() is None and time.monotonic() < deadline, f"harbor view did not start:\n{log.read()}")
                    time.sleep(0.5)
            listing = get("/api/jobs")
            check([(j["name"], j["n_total_trials"], j["n_completed_trials"]) for j in listing["items"]]
                  == [(JOB, len(runs), len(runs))], f"job listing: {listing}")
            trials = get(f"/api/jobs/{JOB}/trials")
            check(trials["total"] == len(runs) and {t["name"] for t in trials["items"]} == set(names), "trial listing")
            for run, name in zip(runs, names):
                check(get(f"/api/jobs/{JOB}/trials/{name}/trajectory") == run.record, f"{run.id}: served trajectory")
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
    return port


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", nargs="?", type=Path, help="jobs directory to write (default: a new temporary one)")
    parser.add_argument("--no-viewer", action="store_true", help="skip starting `harbor view`")
    parser.add_argument("--hub", action="store_true", help="load 3 runs per dataset from Hugging Face (network)")
    args = parser.parse_args()
    runs = [run for name in DATASETS
            for run in (ap.load(name, limit=3) if args.hub else ap.load(name, root=FIXTURES / name))]
    jobs_dir = (args.out_dir or Path(tempfile.mkdtemp(prefix="agentpile-harbor-"))).resolve()
    job_dir, names = write_job(runs, jobs_dir)
    steps = check_files(runs, job_dir, names)
    empty = sum(1 for run in runs if not any(True for _ in run.events()))
    print(f"wrote {len(runs)} trials ({steps} steps, {empty} runs without events) to {job_dir}")
    print(f"loaded back with harbor.models: {len(runs)} ATIF-v1.7 trajectories equal to the agentpile records, "
          f"{len(runs)} trial results, 1 job config, 1 job result")
    if args.no_viewer:
        print("skipped `harbor view` (--no-viewer)")
    else:
        port = check_viewer(runs, jobs_dir, names)
        print(f"harbor view on port {port}: 1 job, {len(runs)} trials and {len(runs)} trajectories served; stopped")
    print(f"browse it: harbor view {jobs_dir}\nOK")


if __name__ == "__main__":
    main()
