"""Reader for agentpile config records (``agentpile-config/0`` JSON)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .._io import first_document, iter_documents
from ..errors import UnsupportedVersion
from ..record import CONFIG, CONFIG_SCHEMA, Run

name = "agentpile-config"


def _looks_like_config(doc: Any) -> bool:
    if isinstance(doc, list):
        doc = doc[0] if doc else None
    return isinstance(doc, dict) and str(doc.get("schema", "")).startswith("agentpile-config/")


def sniff(source: Any) -> bool:
    return _looks_like_config(first_document(source))


def read(source: Any, **options: Any) -> Iterator[Run]:
    if options:
        raise TypeError(f"agentpile-config reader takes no options, got {sorted(options)}")
    if isinstance(source, Path):
        docs = iter_documents(source)
    elif isinstance(source, list):
        docs = iter(source)
    else:
        docs = iter([source])
    for doc in docs:
        if doc.get("schema") != CONFIG_SCHEMA:
            raise UnsupportedVersion(f"config schema {doc.get('schema')!r} is not supported (expected {CONFIG_SCHEMA!r})")
        yield Run(doc, CONFIG)
