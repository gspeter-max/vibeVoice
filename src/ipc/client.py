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
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from rich.console import Capture
from scipy.constants import audio

from src import log
from src.ipc.protocol import (
    fmt_audio,
    fmt_commit,
    fmt_event,
)
from src.streaming.capture_session import CaptureSession
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
def create_socket(cfg: SocketConfig) -> Iterator[socket.socket]:
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
        FileNotFoundError: If the socket server is not running (file not found).
        ConnectionRefusedError: If the socket file exists but the server is offline.
        OSError: For other socket-level errors.
    """
    sock = socket.socket(cfg.family, cfg.socket_type)
    try:
        if cfg.timeout is not None:
            sock.settimeout(cfg.timeout)
        if cfg.address is not None:
            sock.connect(cfg.address)

        yield sock

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Socket connection failed: '{cfg.address}' does not exist. Is the server running?"
        ) from error
    except ConnectionRefusedError as error:
        raise ConnectionRefusedError(
            f"Connection refused: No active listener on '{cfg.address}'."
        ) from error
    except OSError:
        sock.close()
        raise
    finally:
        sock.close()


def send_message(message_bytes: bytes, cfg: SocketConfig | None = None) -> bool:
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
    if cfg is None:
        cfg = SocketConfig()
    try:
        with create_socket(cfg) as sock:
            if isinstance(sock, Exception):
                raise sock

            sock.sendall(message_bytes)

        return True

    except OSError as e:
        log.error(f"Failed to Send message over {cfg.address} : |{e}|")

        return False


def commit_stop(session: CaptureSession, cfg: SocketConfig | None = None):
    if not session.session_id:
        return False

    fmt_msg = fmt_commit(session_id=session.session_id, recording_index=session.rec_idx)

    sent = send_message(fmt_msg, cfg)

    if sent:
        log.info(
            "✅ Session recording stop committed",
            session=session.session_id[:8],
            recording=session.rec_idx,
        )
        session.commit()
        return True

    log.error("❌ Failed to commit session recording stop")
    return False


def send_event(
    session: CaptureSession,
    telemetry_enabled: bool,
    event_type: str,
    fields: dict | None = None,
) -> bool:
    """Send an Ear runtime telemetry event over the Telemetry Brain socket."""
    if not telemetry_enabled or not session.session_id:
        return False

    payload = {"type": event_type}
    if fields:
        payload.update(fields)
    message_bytes = fmt_event(
        session.session_id,
        session.rec_idx,
        payload,
    )
    sent = send_message(message_bytes, SocketConfig(timeout=5.0))
    if not sent:
        log.info(f"[Ear] ❌ Failed to send telemetry event '{event_type}' to telemetry brain")
    return sent


def send_audio(
    session: CaptureSession,
    audio_bytes: bytes,
    telemetry_enabled: bool,
    cfg: SocketConfig | None = None,
) -> bool:
    if not audio_bytes or not session.session_id:
        return False

    fmt_msg = fmt_audio(session.session_id, session.rec_idx, session.mark_sent(), audio_bytes)
    sent = send_message(fmt_msg, cfg)
    if sent:
        send_event(
            session=session,
            telemetry_enabled=telemetry_enabled,
            event_type="chunk_sent_to_brain",
            fields={
                "chunk_index": session.mark_sent(),
                "audio_bytes": len(audio_bytes),
            },
        )
        return True

    log.error("❌ Failed to send chunk")
    return False
