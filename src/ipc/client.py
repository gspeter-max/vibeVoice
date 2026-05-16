"""Socket transport helpers for Ear-to-Brain IPC.

This module owns the mechanical socket work. It does not decide command bytes.
Callers build protocol messages separately and then ask this module to send
them or manage a long-lived raw-audio stream.
"""

from __future__ import annotations

import socket
from typing import Callable

from src.utils.socket_utils import create_and_connect_unix_socket


SOCKET_PATH = "/tmp/parakeet.sock"


def send_message_to_brain(
    message_bytes: bytes,
    timeout_seconds: float = 5.0,
    socket_path: str = SOCKET_PATH,
    socket_factory: Callable[..., socket.socket] | None = None,
) -> bool:
    """Send one complete message over a short-lived Unix socket connection.

    The caller provides already-formatted bytes. This helper only opens the
    socket, sends the bytes, shuts down the write side, and closes the socket.
    If the payload is empty or any socket step fails, the function returns
    `False` so the caller can preserve existing error handling.
    """

    if not message_bytes:
        return False

    try:
        # Use our shared utility to create and connect the socket
        with create_and_connect_unix_socket(
            socket_path=socket_path,
            timeout_seconds=timeout_seconds,
            socket_factory=socket_factory
        ) as client_socket:
            client_socket.sendall(message_bytes)
            client_socket.shutdown(socket.SHUT_WR)
        return True
    except Exception:
        return False


def open_raw_audio_stream_to_brain(
    timeout_seconds: float = 5.0,
    socket_path: str = SOCKET_PATH,
    socket_factory: Callable[..., socket.socket] | None = None,
) -> socket.socket | None:
    """Open the long-lived raw-audio stream used by no-streaming mode.

    This helper returns an open socket object on success. It returns `None`
    when the socket path is missing or the connection attempt fails.
    """

    try:
        # Use our shared utility to create and connect the socket
        return create_and_connect_unix_socket(
            socket_path=socket_path,
            timeout_seconds=timeout_seconds,
            socket_factory=socket_factory
        )
    except Exception:
        return None


def send_raw_audio_stream_chunk(
    raw_stream_socket: socket.socket | None,
    chunk_bytes: bytes,
) -> bool:
    """Send one raw audio chunk on an already-open stream socket."""

    if raw_stream_socket is None:
        return False

    try:
        raw_stream_socket.sendall(chunk_bytes)
        return True
    except Exception:
        return False


def close_raw_audio_stream_to_brain(raw_stream_socket: socket.socket | None) -> None:
    """Shut down and close an open raw-audio stream socket."""

    if raw_stream_socket is None:
        return

    try:
        raw_stream_socket.shutdown(socket.SHUT_WR)
    except Exception:
        pass

    try:
        raw_stream_socket.close()
    except Exception:
        pass
