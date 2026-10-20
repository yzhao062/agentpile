"""Where agentpile keeps downloaded dataset files.

Everything lives under one root, one directory per dataset and pinned
revision: ``<root>/<dataset>/<revision>/``. The root is ``$AGENTPILE_CACHE``
when set, else ``$XDG_CACHE_HOME/agentpile``, else ``~/.cache/agentpile``.
Hub files are downloaded straight into that directory, not into the shared
Hugging Face cache, so ``clear_cache()`` (or deleting the root) removes all
of it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

ENV = "AGENTPILE_CACHE"


def cache_root() -> Path:
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "agentpile"


def dataset_dir(name: str, revision: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid dataset name {name!r}")
    if not revision or "/" in revision or revision.startswith("."):
        raise ValueError(f"invalid revision {revision!r}")
    return cache_root() / name / revision


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cache_info() -> List[Dict[str, object]]:
    """One entry per cached dataset revision: name, revision, bytes and path."""
    root = cache_root()
    if not root.is_dir():
        return []
    out = []
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        for rev in sorted(p for p in ds.iterdir() if p.is_dir()):
            out.append({"dataset": ds.name, "revision": rev.name, "bytes": _size(rev), "path": str(rev)})
    return out


def clear_cache(dataset: Optional[str] = None, revision: Optional[str] = None) -> List[str]:
    """Delete cached files and return the directories removed.

    With no arguments, every dataset under the cache root goes; with
    ``dataset``, only that dataset; with both, one revision. Nothing outside
    the cache root is ever touched.
    """
    root = cache_root().resolve()
    if revision is not None and dataset is None:
        raise ValueError("revision needs dataset")
    if dataset is None:
        targets = [p for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    elif revision is None:
        targets = [root / dataset]
    else:
        targets = [dataset_dir(dataset, revision)]
    removed = []
    for target in targets:
        target = target.resolve()
        if root not in target.parents:
            raise ValueError(f"refusing to delete {target}: it is not inside the cache root {root}")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(str(target))
    return removed
