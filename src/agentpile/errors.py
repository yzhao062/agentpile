"""Exceptions raised by agentpile."""


class AgentpileError(Exception):
    """Base class for agentpile errors."""


class UnsupportedVersion(AgentpileError):
    """The input declares a format version this release was not tested against."""


class AmbiguousSource(AgentpileError, ValueError):
    """More than one reader recognizes the input; pass ``reader=`` to choose."""


class UnrecognizedSource(AgentpileError, ValueError):
    """No reader recognizes the input."""
