"""``agentpile.load``: one entry point for dataset names, files and in-memory records."""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any, Iterator, List, Optional

from .errors import AmbiguousSource, UnrecognizedSource
from .record import Run
from .registry import datasets, readers


def _describe(source: Any) -> str:
    if isinstance(source, Path):
        return f"file {str(source)!r}"
    return f"{type(source).__name__} object"


def _matching_readers(source: Any) -> List[str]:
    matches = []
    for name in readers.names():
        try:
            reader = readers.get(name)
        except ImportError:
            continue  # a broken plugin must not stop the built-in readers
        try:
            ok = reader.sniff(source)
        except Exception:
            ok = False
        if ok:
            matches.append(name)
    return matches


def load(source: Any, reader: Optional[str] = None, **options: Any) -> Iterator[Run]:
    """Load runs from a registered dataset name, a file path, or a record already in memory.

    ``ap.load("who_and_when")`` reads a dataset at its pinned revision.
    ``ap.load("trace.json")`` asks every reader whether it recognizes the
    file; exactly one must, or the call fails and names the candidates, so a
    format is never guessed. ``reader=`` skips detection. Options are passed to
    the dataset loader or reader. Errors about the input are raised here, at
    call time; the returned iterator yields ``Run`` objects lazily.

    ``limit=N`` on a built-in Hugging Face dataset that is not downloaded yet
    fetches only the files its first N runs come from (plus any table every
    run needs), and a later load without ``limit`` fetches the rest.
    """
    if isinstance(source, Run):
        return iter([source])
    if isinstance(source, str) and reader is None and source in datasets:
        return iter(datasets.get(source).load(**options))
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if not path.exists():
            if not isinstance(source, str) or os.sep in source or "/" in source or path.suffix:
                raise FileNotFoundError(f"no such file: {str(source)!r}")
            known = ", ".join(datasets.names()) or "none registered"
            close = difflib.get_close_matches(source, datasets.names(), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise FileNotFoundError(f"{source!r} is neither a registered dataset ({known}) nor an existing file{hint}")
        source = path
    if reader is not None:
        return iter(readers.get(reader).read(source, **options))
    matches = _matching_readers(source)
    if not matches:
        raise UnrecognizedSource(
            f"no reader recognizes {_describe(source)} (tried: {', '.join(readers.names())}); "
            "pass reader=... or register a reader")
    if len(matches) > 1:
        raise AmbiguousSource(f"{_describe(source)} is recognized by {', '.join(matches)}; pass reader=... to choose")
    return iter(readers.get(matches[0]).read(source, **options))
