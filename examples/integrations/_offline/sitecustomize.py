"""Block the network for a recipe run: put this directory first on PYTHONPATH and every connection beyond this
machine raises.

The recipes CI job runs each recipe this way, so a recipe that reaches for the network fails instead of passing on a
download. Loopback stays open for a viewer the recipe starts locally. Installation happens before, with the network on.
DeepEval and Ragas send telemetry unless told not to, so this also opts them out.
"""

import os
import socket

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_connect, _connect_ex, _getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo


def _local(address):
    return not isinstance(address, tuple) or str(address[0]) in LOOPBACK  # a Unix socket path, or loopback


def _check(address):
    if not _local(address):
        raise OSError(f"network blocked for an offline recipe run (examples/integrations/_offline): {address!r}")


def connect(self, address):
    _check(address)
    return _connect(self, address)


def connect_ex(self, address):
    _check(address)
    return _connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    if host is not None and (host.decode() if isinstance(host, bytes) else str(host)) not in LOOPBACK:
        raise OSError(f"network blocked for an offline recipe run (examples/integrations/_offline): {host!r}")
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
