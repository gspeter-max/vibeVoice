"""Socket transport helpers for the Ear-to-Brain IPC layer.

This module handles the low-level socket mechanics only. It does not define
or interpret protocol/command bytes — callers are responsible for constructing
those. The two public surfaces are:

- ``create_socket`` – a context manager that opens and closes a Unix socket.
- ``send_message`` – a one-shot helper that sends a pre-built byte
  payload over a short-lived connection.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

from src import log
from src.utils.settings import settings


@dataclass(frozen=True)
class SocketConfig:
    """Immutable configuration for a Unix socket connection.

    Attributes:
        family: Socket address family (default: ``AF_UNIX``).
        socket_type: Socket type (default: ``SOCK_STREAM``).
        address: Filesystem path to the Unix socket (default: ``settings.ear_to_brain_socket_path``).
        timeout: Optional socket timeout in seconds. ``None`` means blocking mode.
    """

    family: int = socket.AF_UNIX
    socket_type: int = socket.SOCK_STREAM
    address: str = settings.ear_to_brain_socket_path
    timeout: float | None = None


@contextmanager
def create_socket(cfg: SocketConfig) -> Callable:
    """Open a Unix socket connection and yield it as a context manager.

    Creates a socket using the family and type from *cfg*, optionally sets a
    timeout, connects to *cfg.address*, and yields the connected socket.
    The socket is always closed in the ``finally`` block, regardless of
    whether an exception was raised inside the ``with`` block.

    Args:
        cfg: A ``SocketConfig`` instance describing the socket parameters.

    Yields:
        socket.socket: A connected, ready-to-use socket.

    Raises:
        FileExistsError: If the socket file at *cfg.address* does not exist.
    """
    try:
        if not os.path.exists(cfg.address):
            raise FileExistsError(" cfg.address is not exists ")

        sock = socket.socket(cfg.family, cfg.socket_type)
        if cfg.timeout is not None:
            sock.settimeout(cfg.timeout)
        if cfg.address is not None:
            sock.connect(cfg.address)

        yield sock

    finally:
        sock.close()


def send_message(message_bytes: bytes, sock: Optional[socket.socket] = None) -> bool:
    """Send one complete message over a short-lived Unix socket connection.

    Opens a new socket, sends *message_bytes* in full, then performs a
    half-close (``SHUT_WR``) before the connection is torn down by
    ``create_socket``'s ``finally`` block.

    Args:
        message_bytes: The raw byte payload to transmit. Must be non-empty.
        timeout: Socket timeout in seconds applied to the connection.
            Defaults to ``5.0``.

    Returns:
        ``True`` if the message was sent successfully.
        ``False`` if *message_bytes* is empty or any ``OSError`` is raised
        during the socket operation, allowing callers to keep their existing
        error-handling paths without catching exceptions here.
    """
    if not message_bytes:
        return False
    try:
        if sock is None:
            cfg = SocketConfig()
            sock = create_socket(cfg)

        with sock as sock:
            if isinstance(sock, Exception):
                raise sock

            sock.sendall(message_bytes)
            sock.shutdown(socket.SHUT_WR)

        return True

    except OSError as e:
        log.error(f"Failed Send message over {cfg.address} : |{e}|")

        return False
