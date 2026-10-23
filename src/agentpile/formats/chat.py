"""Write a trajectory as one chat row in JSON Lines, as ``ap.to_chat(run, dest)`` does."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .._chat import to_chat
from ..record import Run

name = "chat-jsonl"


def write(run: Run, dest: Union[str, Path]) -> None:
    to_chat(run, dest)
