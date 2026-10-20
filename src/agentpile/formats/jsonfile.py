"""Write a run as JSON: an ATIF trajectory, or an agentpile-config record."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..record import Run

name = "json"


def write(run: Run, dest: Union[str, Path]) -> None:
    run.to_json(dest, indent=2)
