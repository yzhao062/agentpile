"""Reader for ATIF trajectories (``.json``, ``.jsonl``, a dict, or a list of dicts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .._io import first_document, iter_documents
from ..errors import UnsupportedVersion
from ..record import ATIF_VERSION, TRAJECTORY, Run

name = "atif"

# Versions this release was tested against. Anything else is reported as
# unsupported rather than read as if it were compatible.
TESTED_VERSIONS = (ATIF_VERSION,)


def _looks_like_atif(doc: Any) -> bool:
    if isinstance(doc, list):
        doc = doc[0] if doc else None
    return (isinstance(doc, dict) and isinstance(doc.get("schema_version"), str)
            and doc["schema_version"].startswith("ATIF-") and "steps" in doc)


def sniff(source: Any) -> bool:
    return _looks_like_atif(first_document(source))


def read(source: Any, **options: Any) -> Iterator[Run]:
    if options:
        raise TypeError(f"atif reader takes no options, got {sorted(options)}")
    if isinstance(source, Path):
        docs = iter_documents(source)
    elif isinstance(source, list):
        docs = iter(source)
    else:
        docs = iter([source])
    for doc in docs:
        version = doc.get("schema_version")
        if version not in TESTED_VERSIONS:
            raise UnsupportedVersion(f"ATIF {version!r} is not a tested version (tested: {', '.join(TESTED_VERSIONS)})")
        yield Run(doc, TRAJECTORY)
