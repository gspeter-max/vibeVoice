"""Ear-facing HUD client helpers.

This module contains only the lightweight socket clients that Ear uses to tell
the HUD about state changes and live microphone levels.
"""

from __future__ import annotations

import socket
import threading
import time

from src import log
from src.utils.settings import settings
from src.utils.socket_utils import create_socket

def send_hud_command(
    command_text: str,
    *,
    host: str = settings.hud_host,
    port: int = settings.hud_port,
    timeout_seconds: float = 0.2,
    socket_factory=None,
) -> bool:
    """Send one HUD state command over TCP.

    The command vocabulary stays unchanged. This helper only handles the local
    socket send and returns `True` on success or `False` on failure.
    """

    try:
        # Use our shared utility to create and connect the TCP socket
        hud_socket = create_socket(
            family=socket.AF_INET,
            socket_type=socket.SOCK_STREAM,
            address=(host, port),
            timeout_seconds=timeout_seconds,
            socket_factory=socket_factory
        )
        hud_socket.sendall(command_text.encode())
        hud_socket.close()
        return True
    except OSError:
        return False


def change_ui_status(
    command_text: str,
    *,
    host: str = settings.hud_host,
    port: int = settings.hud_port,
    timeout_seconds: float = 0.2,
    socket_factory=None,
):
    """Start one daemon thread that sends a single HUD command.

    Ear triggers these fire-and-forget HUD state changes from multiple places.
    Keeping the thread launch here avoids repeating the same small threading
    boilerplate in the runtime controller.
    """

    sender_thread = threading.Thread(
        target=send_hud_command,
        kwargs={
            "command_text": command_text,
            "host": host,
            "port": port,
            "timeout_seconds": timeout_seconds,
            "socket_factory": socket_factory,
        },
        daemon=True,
    )
    sender_thread.start()
    return sender_thread


def ui_wave_input(
    ear_state,
    *,
    host: str = settings.hud_host,
    volume_port: int = settings.vol_port,
    socket_factory=None,
    send_interval_seconds: float = 0.04,
):
    """Start the background UDP sender that streams Ear volume information.

    The `ear_state` object must expose `_lock`, `is_recording`, `last_rms`,
    and `last_frequency_bands`, which matches the current Ear runtime fields.
    The helper returns the created thread so tests can join it deterministically.
    """

    # Use our shared utility to create the UDP socket
    udp_socket = create_socket(
        family=socket.AF_INET,
        socket_type=socket.SOCK_DGRAM,
        socket_factory=socket_factory
    )

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
            except OSError as error:
                log.info("[Ear] Failed to send volume: %s", error)
            time.sleep(send_interval_seconds)

        udp_socket.close()

    sender_thread = threading.Thread(target=_sender, daemon=True)
    sender_thread.start()
    return sender_thread
