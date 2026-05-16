"""Ear-facing HUD client helpers.

This module contains only the lightweight socket clients that Ear uses to tell
the HUD about state changes and live microphone levels.
"""

from __future__ import annotations

import socket
import threading
import time

from src import log
from src.utils.socket_utils import create_and_connect_tcp_socket, create_udp_socket


HUD_HOST = "127.0.0.1"
HUD_PORT = 57234
VOLUME_PORT = 57235


def send_hud_command(
    command_text: str,
    *,
    host: str = HUD_HOST,
    port: int = HUD_PORT,
    timeout_seconds: float = 0.2,
    socket_factory=None,
) -> bool:
    """Send one HUD state command over TCP.

    The command vocabulary stays unchanged. This helper only handles the local
    socket send and returns `True` on success or `False` on failure.
    """

    try:
        # Use our shared utility to create and connect the TCP socket
        hud_socket = create_and_connect_tcp_socket(
            host=host,
            port=port,
            timeout_seconds=timeout_seconds,
            socket_factory=socket_factory
        )
        hud_socket.sendall(command_text.encode())
        hud_socket.close()
        return True
    except Exception:
        return False


def start_volume_sender_thread(
    ear_state,
    *,
    host: str = HUD_HOST,
    volume_port: int = VOLUME_PORT,
    socket_factory=None,
    send_interval_seconds: float = 0.04,
):
    """Start the background UDP sender that streams Ear volume information.

    The `ear_state` object must expose `_lock`, `is_recording`, `last_rms`,
    and `last_frequency_bands`, which matches the current Ear runtime fields.
    The helper returns the created thread so tests can join it deterministically.
    """

    # Use our shared utility to create the UDP socket
    udp_socket = create_udp_socket(socket_factory=socket_factory)

    def _sender():
        packets_sent = 0
        while True:
            with ear_state._lock:
                if not ear_state.is_recording:
                    log.info(f"[Ear] Volume sender stopped (sent {packets_sent} packets)")
                    break
                rms = ear_state.last_rms
                frequency_bands = ear_state.last_frequency_bands

            try:
                message = (
                    f"vol:{rms:.4f},bass:{frequency_bands['bass']:.3f},"
                    f"mid:{frequency_bands['mid']:.3f},treble:{frequency_bands['treble']:.3f}"
                )
                udp_socket.sendto(message.encode(), (host, volume_port))
                packets_sent += 1
            except Exception as error:
                log.info(f"[Ear] ❌ Failed to send volume: {error}")
            time.sleep(send_interval_seconds)

        udp_socket.close()

    sender_thread = threading.Thread(target=_sender, daemon=True)
    sender_thread.start()
    return sender_thread
