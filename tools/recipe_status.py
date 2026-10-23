"""Status of the integration recipes, for the README's Works With table; run by .github/workflows/recipes.yml.

    python tools/recipe_status.py record --row train --outcome success --requirements examples/integrations/train/requirements.txt \\
        --covers src/agentpile --covers examples/integrations/train --out status-train.json
    python tools/recipe_status.py update results/ status/ --root main-checkout   # prints the rows that failed

``record`` writes one check's result: the row, the outcome, the agentpile revision, the date, the CI run and attempt,
the installed version of every package its requirements pin or ``--package`` names (plus any ``--version NAME=VERSION``
for a platform that is not a Python package, such as Jaeger), and a fingerprint of the code the check covered (``--covers``). ``update``
folds results into a status directory (the ``recipe-status`` branch). For each row it writes a shields.io endpoint
badge (``<row>.json``: passing or stale), a badge with the date of the last passing check (``<row>-last.json``) and
the full record (``<row>.record.json``). A row is stale when its last check failed, or when the code its last passing
check covered differs from that code in ``--root``, a checkout of the current main: the code changed and the check has
not passed on it since. Every fold judges every passing row against the current main, so a run that finishes late,
from an older checkout, can neither turn a row green on old code nor mark a newer pass stale. A result from an older
CI run than the one a row already records is ignored, so a late run cannot overwrite a newer failure either. A row
named with ``--expect`` that has no record yet shows "not run yet". The standard library is enough.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from importlib import metadata
from pathlib import Path


def _pinned(paths):
    """Installed versions of the packages the requirement files name."""
    versions = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            name = re.split(r"[=<>!~;\[ ]", line.split("#", 1)[0].strip(), maxsplit=1)[0]
            if name and not name.startswith("-"):
                versions[name] = _installed(name)
    return versions


def _installed(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _fingerprint(paths, root="."):
    """SHA-256 over the relative path and bytes of every file under ``paths`` (cache files left out)."""
    digest = hashlib.sha256()
    base = Path(root)
    for top in sorted(paths):
        found = [base / top] if (base / top).is_file() else sorted((base / top).rglob("*"))
        for path in found:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                digest.update(path.relative_to(base).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def record(args):
    env = os.environ
    keys = ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    run = "{}/{}/actions/runs/{}".format(*(env[k] for k in keys)) if all(k in env for k in keys) else None
    out = {"row": args.row, "outcome": args.outcome, "revision": env.get("GITHUB_SHA"),
           "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), "run": run,
           "run_id": int(env.get("GITHUB_RUN_ID", 0)), "run_attempt": int(env.get("GITHUB_RUN_ATTEMPT", 0)),
           "event": env.get("GITHUB_EVENT_NAME"),
           "versions": {**_pinned(args.requirements), **{name: _installed(name) for name in args.package},
                        **dict(v.split("=", 1) for v in args.version)},
           "covers": sorted(args.covers), "fingerprint": _fingerprint(args.covers)}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def _badge(label, message, color):
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def _write(status, rec):
    row, last = rec["row"], (rec["last_passing_run"] or {}).get("date")
    color = {"passing": "brightgreen", "not run yet": "lightgrey"}.get(rec["status"], "orange")
    for name, body in ((f"{row}.json", _badge("check", rec["status"], color)),
                       (f"{row}-last.json", _badge("last pass", last or "never", "blue" if last else "lightgrey")),
                       (f"{row}.record.json", rec)):
        (status / name).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def _order(run):
    return (run.get("run_id") or 0, run.get("run_attempt") or 0)


def update(args):
    results, status = Path(args.results), Path(args.status)
    status.mkdir(parents=True, exist_ok=True)
    failed = []
    for path in sorted(results.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        row = result["row"]
        kept = status / f"{row}.record.json"
        previous = json.loads(kept.read_text(encoding="utf-8")) if kept.exists() else {}
        if previous and _order(result) < _order(previous.get("last_run") or {}):
            continue  # an older run than the one this row records
        passed = result["outcome"] == "success"
        rec = {"row": row, "status": "passing" if passed else "stale", "reason": None if passed else "check failed",
               "last_run": result, "last_passing_run": result if passed else previous.get("last_passing_run")}
        _write(status, rec)
        if not passed:
            failed.append(row)
    for row in args.expect:  # a row whose check has never run shows that, instead of a badge that fails to load
        if not (status / f"{row}.record.json").exists():
            _write(status, {"row": row, "status": "not run yet", "reason": "no check has run", "last_run": None,
                            "last_passing_run": None})
    for kept in sorted(status.glob("*.record.json")):  # every passing row, this run's too: did it pass on main's code?
        rec = json.loads(kept.read_text(encoding="utf-8"))
        checked = rec.get("last_passing_run") or {}
        if rec["status"] != "passing" or not checked.get("covers"):
            continue
        if _fingerprint(checked["covers"], args.root) != checked.get("fingerprint"):
            rec.update(status="stale", reason="the code it covers changed after its last check")
            _write(status, rec)
    print("\n".join(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--row", required=True)
    rec.add_argument("--outcome", required=True, choices=["success", "failure", "cancelled"])
    rec.add_argument("--requirements", action="append", default=[])
    rec.add_argument("--package", action="append", default=[], help="an installed package whose version to record")
    rec.add_argument("--version", action="append", default=[], help="NAME=VERSION of a platform pip does not install")
    rec.add_argument("--covers", action="append", default=[], help="a file or directory whose change makes the row stale")
    rec.add_argument("--out", required=True)
    upd = sub.add_parser("update")
    upd.add_argument("results")
    upd.add_argument("status")
    upd.add_argument("--root", default=".", help="a checkout of the current main, which every passing row is judged by")
    upd.add_argument("--expect", action="append", default=[], help="a row to show as not run yet when it has no record")
    args = parser.parse_args(argv)
    (record if args.command == "record" else update)(args)


if __name__ == "__main__":
    sys.exit(main())
