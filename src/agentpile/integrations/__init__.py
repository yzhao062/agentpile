"""Framework integrations behind ``agentpile.capture`` and ``agentpile.config_from``.

Each integration module imports its framework at the top, so a module is
imported only when an agent of that framework is passed in, and
``import agentpile`` loads none of them. A module provides

- ``config_record(agents) -> dict``: an ``agentpile-config/0`` record for the
  agents and everything reachable from them (tools, handoffs);
- ``Binding(agent, session)`` with ``enter()`` and ``exit()``: start and stop
  feeding the capture session, restoring whatever it changed.

Supported: ``pydantic-ai`` (``pydantic_ai.Agent``) and ``openai-agents``
(``agents.Agent``).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

# framework name -> (module, pip extra)
FRAMEWORKS: Dict[str, Tuple[str, str]] = {
    "pydantic-ai": ("agentpile.integrations.pydantic_ai", "pydantic-ai"),
    "openai-agents": ("agentpile.integrations.openai_agents", "openai-agents"),
}

_TOP_LEVEL = {"pydantic_ai": "pydantic-ai", "agents": "openai-agents"}


def framework_of(agent: Any) -> str:
    """Name the framework an agent object belongs to, from its class hierarchy (the framework is not imported)."""
    for cls in type(agent).__mro__:
        top = (getattr(cls, "__module__", "") or "").split(".", 1)[0]
        if top in _TOP_LEVEL:
            return _TOP_LEVEL[top]
    raise TypeError(
        f"cannot tell which framework {type(agent).__name__!r} belongs to; "
        f"supported agents: {', '.join(sorted(FRAMEWORKS))}")


def load(framework: str) -> Any:
    """Import an integration module; an absent framework is reported with the extra that installs it."""
    from .._capture import CaptureError

    module, extra = FRAMEWORKS[framework]
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise CaptureError(
            f"capturing a {framework} agent needs its packages ({exc}); "
            f"install them with: pip install 'agentpile[{extra}]'") from exc
