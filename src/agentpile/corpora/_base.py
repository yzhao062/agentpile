"""Shared machinery for dataset loaders: pinned downloads, manifests and count checks."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..cache import dataset_dir
from ..errors import AgentpileError
from ..record import Run


class CountMismatch(AgentpileError):
    """A full load produced a different number of runs than the dataset's manifest records."""


@lru_cache(maxsize=None)
def _manifest_text(name: str) -> str:
    return resources.files("agentpile.corpora").joinpath(f"manifests/{name}.json").read_text(encoding="utf-8")


def manifest(name: str) -> Dict[str, Any]:
    """The bundled manifest of one dataset loader (a fresh dict each call; the file is read once)."""
    return json.loads(_manifest_text(name))


def require_hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError("loading a Hugging Face dataset needs the hf extra: pip install 'agentpile[hf]'") from exc
    return huggingface_hub


def fetch(name: str, repo_id: str, revision: str, allow_patterns: Optional[List[str]] = None,
          max_workers: int = 4) -> Path:
    """Download the pinned files of a hub dataset into ``<cache root>/<name>/<revision>/``.

    Files go straight into that directory (``local_dir``), not into the shared
    Hugging Face cache. A marker file records the patterns of a finished
    download, so a second call does no network work.
    """
    local = dataset_dir(name, revision)
    marker = local / ".agentpile-complete.json"
    want = {"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns}
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == want:
                return local
        except ValueError:
            pass
    hub = require_hub()
    hub.snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                          allow_patterns=allow_patterns, local_dir=str(local), max_workers=max_workers)
    marker.write_text(json.dumps(want) + "\n", encoding="utf-8")
    return local


class Dataset:
    """A dataset pinned to one revision and read into runs by a dataset-specific converter.

    Subclasses set ``name`` (the loader name, which is also the manifest file
    name) and implement ``records(root, **options)``, yielding ``(record_key,
    raw_record, where)``, and ``convert(record_key, raw_record, where)``,
    returning a ``Run``. ``where`` holds the file, config and split a record
    came from. The repository id, revision, download patterns and expected
    count come from the manifest.
    """

    name: str = ""
    format: str = ""
    reader_version: str = "1"

    @property
    def manifest(self) -> Dict[str, Any]:
        cached = self.__dict__.get("_manifest")
        if cached is None:
            cached = self.__dict__["_manifest"] = manifest(self.name)
        return cached

    @property
    def repo_id(self) -> Optional[str]:
        return self.manifest.get("hf_id")

    @property
    def revision(self) -> str:
        return self.manifest["revision"]

    @property
    def allow_patterns(self) -> Optional[List[str]]:
        return self.manifest.get("allow_patterns")

    @property
    def expected_runs(self) -> Optional[int]:
        return ((self.manifest.get("counts") or {}).get("runs_after_load") or {}).get("value")

    def files(self) -> Path:
        return fetch(self.name, self.repo_id, self.revision, self.allow_patterns)

    def source(self, key: str, where: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": self.format, "dataset": self.repo_id, "revision": self.revision,
                "config": where.get("config"), "split": where.get("split"), "file": where.get("file"),
                "record_key": key, "reader": self.name, "reader_version": self.reader_version,
                "lost_fields": []}

    def records(self, root: Path, **options: Any) -> Iterator[Tuple[str, Any, Dict[str, Any]]]:
        raise NotImplementedError

    def convert(self, key: str, raw: Any, where: Dict[str, Any]) -> Run:
        raise NotImplementedError

    def load(self, limit: Optional[int] = None, verify: Optional[bool] = None, root: Optional[str] = None,
             **options: Any) -> Iterator[Run]:
        """Runs, read lazily. ``root`` reads a local directory laid out like the pinned
        files instead of downloading. A complete, unfiltered load of the pinned
        files is checked against the manifest count; ``verify=True`` also checks
        a ``root`` copy, and ``verify=False`` skips the check.

        The files are fetched here, before the iterator is returned, so a
        missing ``hf`` extra or a failed download is raised by this call.
        """
        base = Path(root) if root is not None else self.files()
        return self._runs(base, limit, verify, root, options)

    def _runs(self, base: Path, limit: Optional[int], verify: Optional[bool], root: Optional[str],
              options: Dict[str, Any]) -> Iterator[Run]:
        n = 0
        for key, raw, where in self.records(base, **options):
            yield self.convert(key, raw, where)
            n += 1
            if limit is not None and n >= limit:
                return
        expected = self.expected_runs
        check = verify if verify is not None else root is None
        if check and not options and expected is not None and n != expected:
            raise CountMismatch(f"{self.name}: read {n} runs at {self.revision[:12]}, manifest records {expected}")

    def __repr__(self) -> str:
        return f"<agentpile dataset {self.name} {self.repo_id}@{self.revision[:12]}>"
