"""Plugin registries for readers, datasets and writers.

Each registry is filled from its entry-point group on first use, so an
installed plugin package needs no import by the user. ``register()`` adds a
plugin at run time (a notebook, a test) and wins over an entry point of the
same name. A value may be a ``"module:attr"`` string, imported on first
``get()``, which keeps optional dependencies of built-in datasets out of
``import agentpile``.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Dict, Iterator, List, Tuple

READERS_GROUP = "agentpile.readers"
DATASETS_GROUP = "agentpile.datasets"
WRITERS_GROUP = "agentpile.writers"


def _entry_points(group: str) -> List[Any]:
    eps = metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return list(eps.select(group=group))
    return list(eps.get(group, []))  # Python 3.9 returns a dict


def _resolve(value: Any) -> Any:
    """Load a ``"module:attr.path"`` string the way an entry point is loaded, nested attributes included."""
    if isinstance(value, str):
        return metadata.EntryPoint(name="", value=value, group="").load()
    return value


class Registry:
    """Name -> plugin object, in registration order."""

    def __init__(self, group: str, required: Tuple[str, ...]):
        self.group = group
        self.required = required
        self._items: Dict[str, Any] = {}
        self._broken: Dict[str, str] = {}
        self._discovered = False

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        for ep in _entry_points(self.group):
            if ep.name in self._items:
                continue  # built-ins and run-time registrations win over entry points
            self._items[ep.name] = ep.value

    def register(self, name: str, obj: Any, replace: bool = False) -> Any:
        self._discover()
        if name in self._items and not replace:
            raise ValueError(f"{self.group}: {name!r} is already registered; pass replace=True to override")
        if not isinstance(obj, str):
            missing = [a for a in self.required if not hasattr(obj, a)]
            if missing:
                raise TypeError(f"{self.group}: {name!r} lacks {', '.join(missing)}")
        self._items[name] = obj
        self._broken.pop(name, None)
        return obj

    def _builtin(self, name: str, value: str) -> None:
        """Add a built-in before entry points are read, so a plugin of the same name cannot shadow it."""
        self._items[name] = value

    def unregister(self, name: str) -> None:
        self._discover()
        self._items.pop(name, None)

    def get(self, name: str) -> Any:
        self._discover()
        if name not in self._items:
            known = ", ".join(self.names()) or "none"
            raise KeyError(f"{self.group}: no plugin named {name!r} (known: {known})")
        value = self._items[name]
        if isinstance(value, str):
            try:
                value = _resolve(value)
            except Exception as exc:  # report which plugin failed, not a bare ImportError
                self._broken[name] = repr(exc)
                raise ImportError(f"{self.group}: {name!r} could not be loaded from {self._items[name]!r}: {exc}") from exc
            self._items[name] = value
        return value

    def names(self) -> List[str]:
        self._discover()
        return list(self._items)

    def items(self) -> Iterator[Tuple[str, Any]]:
        for name in self.names():
            yield name, self.get(name)

    def __contains__(self, name: object) -> bool:
        self._discover()
        return name in self._items


readers = Registry(READERS_GROUP, ("sniff", "read"))
datasets = Registry(DATASETS_GROUP, ("load",))
writers = Registry(WRITERS_GROUP, ("write",))

_KINDS = {"reader": readers, "dataset": datasets, "writer": writers}


def register(name: str, obj: Any, kind: str = "reader", replace: bool = False) -> Any:
    """Register a reader, dataset or writer for this process.

    A reader has ``sniff(source) -> bool`` and ``read(source, **options)``
    yielding ``Run`` objects; a dataset has ``load(**options)``; a writer has
    ``write(run, dest)``. Packages register the same objects through the
    entry-point groups ``agentpile.readers``, ``agentpile.datasets`` and
    ``agentpile.writers``.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    return _KINDS[kind].register(name, obj, replace=replace)
