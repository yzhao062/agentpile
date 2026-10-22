"""Shared machinery for dataset loaders: pinned downloads, manifests and count checks."""

from __future__ import annotations

import copy
import fnmatch
import json
import re
from contextlib import contextmanager
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..cache import dataset_dir
from ..errors import AgentpileError, GatedDataset
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


@contextmanager
def gated(name: str, repo_id: str) -> Iterator[None]:
    """Report the hub's refusal of a gated dataset with the steps that grant access; the hub's error is the cause."""
    from huggingface_hub.utils import GatedRepoError

    try:
        yield
    except GatedRepoError as exc:
        raise GatedDataset(
            f"{name}: the Hugging Face dataset {repo_id} is gated. Accept its terms at "
            f"https://huggingface.co/datasets/{repo_id} while signed in, then give huggingface_hub a token of that "
            f"account: run `hf auth login`, or set HF_TOKEN (in Google Colab, add HF_TOKEN under Secrets, enable its "
            f"notebook access and restart the session, since the hub reads Colab secrets once)") from exc


def finished(name: str, repo_id: str, revision: str, allow_patterns: Optional[List[str]] = None) -> Optional[Path]:
    """The cache directory of a finished download of these patterns, or ``None``."""
    local = dataset_dir(name, revision)
    marker = local / ".agentpile-complete.json"
    want = {"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns}
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == want:
                return local
        except ValueError:
            pass
    return None


def fetch(name: str, repo_id: str, revision: str, allow_patterns: Optional[List[str]] = None,
          max_workers: int = 4) -> Path:
    """Download the pinned files of a hub dataset into ``<cache root>/<name>/<revision>/``.

    Files go straight into that directory (``local_dir``), not into the shared
    Hugging Face cache. A marker file records the patterns of a finished
    download, so a second call does no network work. Files an earlier partial
    load left there are reused, not fetched again. The marker is written only
    when every file the hub lists for these patterns is on disk, so a download
    that fell back to a partial copy (offline, say) is never taken as complete.
    """
    done = finished(name, repo_id, revision, allow_patterns)
    if done is not None:
        return done
    local = dataset_dir(name, revision)
    want = {"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns}
    hub = require_hub()
    pinned = Partial(name, repo_id, revision, allow_patterns).listing
    with gated(name, repo_id):
        hub.snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                              allow_patterns=allow_patterns, local_dir=str(local), max_workers=max_workers)
    missing = [path for path in pinned if not (local / path).is_file()]
    if missing:
        raise FileNotFoundError(f"{repo_id}@{revision[:12]}: {len(missing)} pinned file(s) still missing after the "
                                f"download, e.g. {missing[0]}; the download was not marked complete")
    (local / ".agentpile-complete.json").write_text(json.dumps(want) + "\n", encoding="utf-8")
    return local


class Partial:
    """The pinned files of a hub dataset, fetched one at a time as a reader reaches them.

    One listing of the dataset at its revision, narrowed by ``allow_patterns``
    with huggingface_hub's own matching, gives the file names. A file is
    downloaded into the same ``<cache root>/<name>/<revision>/`` directory a
    full download uses, the first time ``get`` is asked for it; a file already
    there is not fetched again. The completion marker is never written, so a
    later full load still checks the whole set and reuses what is present.
    """

    def __init__(self, name: str, repo_id: str, revision: str, allow_patterns: Optional[List[str]] = None):
        self.hub = require_hub()
        from huggingface_hub.utils import filter_repo_objects

        self.name, self.repo_id, self.revision = name, repo_id, revision
        self.local = dataset_dir(name, revision)
        with gated(name, repo_id):
            entries = self.hub.HfApi().list_repo_tree(repo_id, recursive=True, revision=revision, repo_type="dataset")
            names = [e.path for e in entries if getattr(e, "size", None) is not None]  # files only; folders: no size
        self.listing = sorted(filter_repo_objects(names, allow_patterns=allow_patterns))
        self._names = set(self.listing)

    def match(self, pattern: str) -> List[str]:
        """The listed files that ``Path.glob(pattern)`` would find; ``*``, ``?`` and ``[...]`` act within one path part."""
        if "**" in pattern:
            raise ValueError(f"recursive pattern {pattern!r} is not supported")
        parts = [re.compile(fnmatch.translate(p)) for p in pattern.split("/")]
        found = []
        for name in self.listing:
            got = name.split("/")
            if len(got) == len(parts) and all(p.match(g) for p, g in zip(parts, got)):
                found.append(name)
        return found

    def get(self, path: Path) -> Path:
        """``path``, downloaded first when it is not on disk."""
        if not path.is_file():
            name = path.relative_to(self.local).as_posix()
            if name not in self._names:
                raise FileNotFoundError(f"{name} is not among the pinned files of {self.repo_id}@{self.revision[:12]}")
            with gated(self.name, self.repo_id):
                self.hub.hf_hub_download(repo_id=self.repo_id, filename=name, repo_type="dataset",
                                         revision=self.revision, local_dir=str(self.local))
            if not path.is_file():
                raise FileNotFoundError(f"the download of {name} did not produce {path}")
        return path


class Dataset:
    """A dataset pinned to one revision and read into runs by a dataset-specific converter.

    Subclasses set ``name`` (the loader name, which is also the manifest file
    name) and implement ``records(root, **options)``, yielding ``(record_key,
    raw_record, where)``, and ``convert(record_key, raw_record, where)``,
    returning a ``Run``. ``where`` holds the file, config and split a record
    came from. The repository id, revision, download patterns and expected
    count come from the manifest.

    A reader that finds its data files with ``self.paths(root, pattern, key)``
    and opens each through ``self.ready(path)`` sets ``incremental``, so a load
    with ``limit=`` downloads only the files it reaches. A file the reader
    opens by a fixed name before its first run (an annotation table, a tool
    list) goes in ``upfront``. Any other dataset keeps the full download.
    """

    name: str = ""
    format: str = ""
    reader_version: str = "1"
    incremental: bool = False
    upfront: Tuple[str, ...] = ()

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

    def paths(self, root: Path, pattern: str, key: Optional[Callable[[Path], Any]] = None) -> List[Path]:
        """The data files under ``root`` that ``Path.glob(pattern)`` finds, sorted by ``key``.

        In a load that fetches files one at a time, the names come from the
        hub listing of the pinned revision instead of the disk, in the same
        order, and nothing is downloaded until ``ready`` is called.
        """
        partial = self.__dict__.get("_partial")
        if partial is None:
            return sorted(Path(root).glob(pattern), key=key)
        return sorted((partial.local / name for name in partial.match(pattern)), key=key)

    def ready(self, path: Path) -> Path:
        """``path`` with its file present: downloaded now in a load that fetches files one at a time."""
        partial = self.__dict__.get("_partial")
        return path if partial is None else partial.get(path)

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

        A full load fetches its files here, before the iterator is returned, so
        a missing ``hf`` extra or a failed download is raised by this call.

        With ``limit`` on an ``incremental`` dataset that is not fully
        downloaded yet, only the hub listing and the ``upfront`` files are
        fetched here; each data file is downloaded when the reader reaches it,
        and no file after the ``limit``-th run is. The runs and their order
        are those of a full load. Such a load never marks the download
        complete, so a later full load fetches the rest and reuses these files.
        """
        if root is not None:
            return self._runs(Path(root), limit, verify, root, options)
        if limit is None or not self.incremental:
            return self._runs(self.files(), limit, verify, root, options)
        done = finished(self.name, self.repo_id, self.revision, self.allow_patterns)
        if done is not None:
            return self._runs(done, limit, verify, root, options)
        view = copy.copy(self)  # carries this load's file listing, so concurrent loads do not share it
        partial = view.__dict__["_partial"] = Partial(self.name, self.repo_id, self.revision, self.allow_patterns)
        for name in self.upfront:
            partial.get(partial.local / name)
        return view._runs(partial.local, limit, verify, root, options)

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
