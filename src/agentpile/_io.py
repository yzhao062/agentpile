"""Small file helpers shared by the built-in readers (standard library only)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Optional


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _is_jsonl(path: Path) -> bool:
    suffixes = [s for s in path.suffixes if s != ".gz"]
    return bool(suffixes) and suffixes[-1] in (".jsonl", ".ndjson")


def iter_documents(path: Path) -> Iterator[Any]:
    """Yield the JSON documents in a file: one per line for JSONL, else the whole file.

    A JSON file whose top level is a list yields its items, so a file holding
    several trajectories reads like a JSONL file.
    """
    with _open_text(path) as fh:
        if _is_jsonl(path):
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
        doc = json.load(fh)
    if isinstance(doc, list):
        yield from doc
    else:
        yield doc


def peek_document(path: Path) -> Optional[Any]:
    """The first JSON document in a file, or None when the file is not JSON."""
    try:
        return next(iter_documents(path), None)
    except (ValueError, UnicodeDecodeError, OSError):
        return None


def first_document(source: Any) -> Optional[Any]:
    """What a reader's ``sniff`` inspects: the object itself, or the first document of a file."""
    if isinstance(source, Path):
        return peek_document(source) if source.is_file() else None
    return source
