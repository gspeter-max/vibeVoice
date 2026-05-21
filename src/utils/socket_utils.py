"""Socket creation and connection utilities.

This module provides standardized helpers for creating and connecting different
types of sockets (Unix, TCP, UDP). It abstracts the boilerplate of socket
initialization and connection setup.
"""

from __future__ import annotations

import socket
from typing import Callable

def create_socket(
    family: int,
    socket_type: int,
    address: str | tuple[str, int] | None = None,
    timeout_seconds: float | None = None,
    socket_factory: Callable[..., socket.socket] | None = None,
) -> socket.socket:
    """Create a socket with a specific family and type, optional timeout, and optional connection address.

    Args:
        family: Address family (e.g., socket.AF_UNIX, socket.AF_INET).
        socket_type: Socket type (e.g., socket.SOCK_STREAM, socket.SOCK_DGRAM).
        address: Optional address to connect to.
        timeout_seconds: Optional timeout in seconds.
        socket_factory: Optional factory for creating the socket object (used for mocking).

    Returns:
        An open socket (and optionally connected if address is provided).
    """
    if socket_factory is None:
        socket_factory = socket.socket

    sock = socket_factory(family, socket_type)
    if timeout_seconds is not None:
        sock.settimeout(timeout_seconds)
    if address is not None:
        sock.connect(address)
    return sock

